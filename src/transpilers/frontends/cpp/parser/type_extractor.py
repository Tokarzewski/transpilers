"""C++ AST type ground-truth extractor (issue #50).

Walks a libclang ``TranslationUnit`` and collects resolved types for
*every* node the HIR->MIR pass cares about. The resulting
``TypeGroundTruth`` map is then handed to a downstream pass that
substitutes concrete types for ``UnknownT`` holes in MIR.

Why we need this
----------------

The C++ frontend records the *type the user wrote* in HIR (e.g. the
declared type of a parameter). For most C/C++ code that is enough. For
real C++ -- typedefs, ``auto``, ``decltype``, templated call returns,
``std::vector<T>::size_type`` -- the declared type is sometimes wrong
or the inference pass can't bridge the gap. Clang, on the other hand,
*already knows* the canonical type: it has the full type system, has
run the preprocessor, and instantiates templates on the fly. We just
ask it.

Keyed by source location
------------------------

Every HIR node carries an optional ``source_loc`` field (``file:line:col``
in libclang's format). The extractor stores entries keyed by that
``SourceLocation`` so the ground truth is looked up at the same
physical source position the HIR node came from. A function return
type is keyed by both the function's location *and* its fully-qualified
name (mangled or simple), so call sites can resolve ``f()`` -> ``int``
without needing per-call-site locations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import clang.cindex as ci

from transpilers.ir.types import (
    BoolT,
    FloatT,
    IntT,
    ListT,
    NoneT,
    StrT,
    StructT,
    Type,
    UnknownT,
)

if TYPE_CHECKING:
    from clang.cindex import Cursor

from .types import (
    CPP_TYPE_ALIASES,
    SIMD_TYPE_ALIASES,
)

_log = logging.getLogger(__name__)

# Cursor kinds that always mark the boundary between a complete
# declaration and its body. When we walk the AST we use these as the
# points at which to record "this name has this type".
_DECL_KINDS: frozenset[ci.CursorKind] = frozenset(
    k
    for k in (
        ci.CursorKind.VAR_DECL,
        ci.CursorKind.PARM_DECL,
        ci.CursorKind.FIELD_DECL,
        ci.CursorKind.FUNCTION_DECL,
        ci.CursorKind.CXX_METHOD,
        ci.CursorKind.CONSTRUCTOR,
        ci.CursorKind.DESTRUCTOR,
        ci.CursorKind.STRUCT_DECL,
        ci.CursorKind.CLASS_DECL,
        ci.CursorKind.ENUM_DECL,
        ci.CursorKind.TYPEDEF_DECL,
        ci.CursorKind.TYPE_ALIAS_DECL,
        # FUNCTION_TEMPLATE: the per-kind check below is what actually
        # decides what to record, but adding it to _DECL_KINDS keeps
        # the dispatch logic uniform.
        ci.CursorKind.FUNCTION_TEMPLATE,
    )
    if k is not None
)


def _loc(cursor: "Cursor") -> str:
    """Canonical ``file:line:col`` key for cursor lookups. libclang
    returns ``Cursor.location`` whose ``file.name`` is the unsaved-file
    name we passed at parse time (``"input.cpp"``), so the prefix is
    stable across runs."""
    loc = cursor.location
    return f"{loc.file.name if loc.file else '?'}:{loc.line}:{loc.column}"


def _spelling(cursor: "Cursor") -> str:
    """Best-effort bare name for a cursor. For CXX_METHOD we keep the
    method name only (no class qualifier) so callers don't have to
    know whether they are looking up a free function or a method."""
    try:
        return cursor.spelling or ""
    except Exception:
        return ""


def _qualified_name(cursor: "Cursor") -> str:
    """``"mehara::sort::bubble_sort"``-shaped name for a function. Used
    to disambiguate overloads and to look up the return type from a
    call site that already has the qualified name."""
    parts: list[str] = []
    cur: "Cursor | None" = cursor
    while cur is not None and cur.kind != ci.CursorKind.TRANSLATION_UNIT:
        s = _spelling(cur)
        if s:
            parts.append(s)
        cur = cur.semantic_parent
    return "::".join(reversed(parts))


def _to_type(t: "ci.Type | None") -> Type:
    """Convert a libclang ``Type`` to a transpiler ``Type`` instance.

    The frontend's ``_type_text`` returns *string* annotations (used
    by the HIR layer for backward compatibility); the ground-truth
    table needs *typed* values so ``isinstance(ty, UnknownT)`` checks
    in the MIR pass work. This function maps the same libclang
    spellings onto the dataclass ``Type`` lattice instead.
    """
    if t is None:
        return UnknownT()
    spelling = t.spelling
    # Strip cv-qualifiers and references for the alias lookup. We
    # don't recurse through references/pointers here -- the
    # caller (the AST walker) handles that for free by giving us
    # the canonical / pointee type when the user wrote
    # ``std::vector<int>&``.
    cleaned = (
        spelling.replace("const ", "")
        .replace("volatile ", "")
        .strip()
        .rstrip("&")
        .strip()
    )
    if cleaned in CPP_TYPE_ALIASES:
        # Map the C++ type-text string back to a Type instance.
        from ._to_mir_type import text_to_type

        return text_to_type(CPP_TYPE_ALIASES[cleaned])
    # Templates: spelled ``std::vector<int>`` -- collapse to ListT.
    # We recurse through the libclang type to get the *actual*
    # template argument (e.g. `T` for an uninstantiated template,
    # `int` for a concrete instantiation). This is the difference
    # between ``ListT(elem=StructT("T"))`` (informative) and
    # ``ListT(elem=UnknownT())`` (a hole).
    #
    # NB: the `template_argument_type` accessor doesn't work on
    # reference / pointer kinds (libclang returns -1). For a
    # ``std::vector<T>&`` the recursion below peels the reference
    # off first and queries the template arguments on the pointee.
    if cleaned.startswith(("vector<", "std::vector<")) and cleaned.endswith(">"):
        target = t
        # Unwrap references / pointers so the template-argument
        # accessor returns a non-negative count.
        if target.kind in (
            ci.TypeKind.LVALUEREFERENCE,
            ci.TypeKind.RVALUEREFERENCE,
            ci.TypeKind.POINTER,
        ):
            try:
                target = target.get_pointee()
            except Exception:
                pass
        try:
            num_args = target.get_num_template_arguments()
            if num_args > 0:
                arg_ty = target.get_template_argument_type(0)
                elem = _to_type(arg_ty)
                # If the arg is a template type parameter (UNEXPOSED
                # with spelling == the param name like 'T'), keep the
                # *spelling* as a StructT (a real Type, not a bare str --
                # every backend's type-lattice walk expects a Type
                # instance) so the ground truth reads
                # `ListT(elem=StructT("T"))` rather than collapsing to
                # `ListT(elem=UnknownT())`. This is the difference
                # between an informative ground truth and a hole.
                if isinstance(elem, UnknownT) and arg_ty.spelling:
                    return ListT(elem=StructT(name=arg_ty.spelling))
                if not isinstance(elem, UnknownT):
                    return ListT(elem=elem)
        except Exception:
            pass
        return ListT(elem=UnknownT(hint="vector element"))
    # String shapes: ``std::string``, ``std::string_view``.
    if cleaned in (
        "string",
        "std::string",
        "string_view",
        "std::string_view",
        "basic_string",
        "basic_string_view",
        "std::basic_string",
        "std::basic_string_view",
    ):
        return StrT()
    if cleaned in SIMD_TYPE_ALIASES:
        from ._to_mir_type import text_to_type

        return text_to_type(SIMD_TYPE_ALIASES[cleaned])
    # Fallback: try the libclang kind, with a canonical-type recursion
    # for AUTO / ELABORATED / TYPEDEF that hides the underlying type.
    try:
        kind = t.kind
        if kind == ci.TypeKind.VOID:
            return NoneT()
        if kind in (ci.TypeKind.BOOL,):
            return BoolT()
        INTEGER_KINDS = (
            ci.TypeKind.INT,
            ci.TypeKind.LONG,
            ci.TypeKind.LONGLONG,
            ci.TypeKind.SHORT,
            ci.TypeKind.SCHAR,
            ci.TypeKind.UCHAR,
            ci.TypeKind.CHAR_S,
            ci.TypeKind.CHAR_U,
            ci.TypeKind.UINT,
            ci.TypeKind.ULONG,
            ci.TypeKind.ULONGLONG,
            ci.TypeKind.USHORT,
        )
        if kind in INTEGER_KINDS:
            return IntT()
        if kind in (ci.TypeKind.FLOAT, ci.TypeKind.DOUBLE, ci.TypeKind.LONGDOUBLE):
            return FloatT()
        if kind in (ci.TypeKind.AUTO, ci.TypeKind.ELABORATED, ci.TypeKind.UNEXPOSED):
            canonical = t.get_canonical()
            if canonical.kind != kind:
                return _to_type(canonical)
        if kind == ci.TypeKind.TYPEDEF:
            canonical = t.get_canonical()
            if canonical.kind != kind:
                return _to_type(canonical)
    except Exception:
        pass
    return UnknownT()


@dataclass
class TypeGroundTruth:
    """Resolved-type ground truth extracted from a clang AST.

    The four index dimensions are:

    * ``var_types`` -- ``source_loc`` -> type, for ``VAR_DECL``,
      ``PARM_DECL`` and ``FIELD_DECL`` cursors.
    * ``func_returns`` -- qualified function name -> return type.
    * ``func_params`` -- qualified function name -> ordered param types.
    * ``call_returns`` -- ``source_loc`` of a ``CALL_EXPR`` -> the
      *canonical* return type clang computed for that specific call
      (taking template-argument deduction and implicit conversions
      into account).
    """

    var_types: dict[str, Type] = field(default_factory=dict)
    func_returns: dict[str, Type] = field(default_factory=dict)
    func_params: dict[str, list[Type]] = field(default_factory=dict)
    call_returns: dict[str, Type] = field(default_factory=dict)
    # Auxiliary: qualified name -> source loc of its decl. Lets later
    # passes look up a function's parameter/return types from a call
    # site by the callee spelling.
    decl_locs: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_tu(cls, tu, *, only_input: bool = True) -> "TypeGroundTruth":
        """Walk *tu*'s AST and build the ground truth.

        When *only_input* is true (the default), decls from system
        headers are skipped so the table reflects only what the user's
        code actually contains. Set it to false when you want the
        full project -- e.g. to look up ``std::vector::size``'s return
        type -- but be aware the table gets much larger.
        """
        self = cls()
        if tu is None or tu.cursor is None:
            return self
        self._walk(tu.cursor, only_input=only_input)
        return self

    def _walk(self, cursor, *, only_input: bool) -> None:
        """Recursive AST walk that records decls and calls."""

        def is_input(c) -> bool:
            if not only_input:
                return True
            f = c.location.file
            if f is None or f.name != "input.cpp":
                return False
            # Skip the parser preamble -- it's the first
            # ``PARSER_PREAMBLE + _project_preamble`` lines of the
            # unsaved file. ``_compute_user_first_line`` is computed
            # lazily here so a single TU walk does the lookup once.
            try:
                first = self._user_first_line
            except AttributeError:
                from .core import _compute_user_first_line

                first = _compute_user_first_line()
                self._user_first_line = first
            return c.location.line > first

        kind = cursor.kind
        if kind in _DECL_KINDS and is_input(cursor):
            self._record_decl(cursor)
        if kind in (ci.CursorKind.CALL_EXPR,) and is_input(cursor):
            self._record_call(cursor)
        for child in cursor.get_children():
            self._walk(child, only_input=only_input)

    def _record_decl(self, cursor) -> None:
        loc = _loc(cursor)
        name = _spelling(cursor)
        if not name:
            return
        if cursor.kind in (
            ci.CursorKind.VAR_DECL,
            ci.CursorKind.PARM_DECL,
            ci.CursorKind.FIELD_DECL,
            ci.CursorKind.TYPEDEF_DECL,
            ci.CursorKind.TYPE_ALIAS_DECL,
        ):
            self.var_types[loc] = _to_type(cursor.type)
            return
        if cursor.kind in (
            ci.CursorKind.FUNCTION_DECL,
            ci.CursorKind.CXX_METHOD,
            ci.CursorKind.CONSTRUCTOR,
            ci.CursorKind.DESTRUCTOR,
            # FUNCTION_TEMPLATE: a template definition is a function
            # for the purpose of *signature lookup* (param/return types).
            # The IR doesn't instantiate templates, so the body is a
            # HirRaw hole -- but call sites that hit the template name
            # can still pull `void bubble_sort(vector<T>&)` from here.
            ci.CursorKind.FUNCTION_TEMPLATE,
        ):
            # Skip non-definitions: bare declarations of an existing
            # function shouldn't overwrite the definition's signature.
            if not cursor.is_definition():
                return
            qname = _qualified_name(cursor) or name
            self.func_returns[qname] = _to_type(cursor.result_type)
            self.func_params[qname] = [
                _to_type(c.type)
                for c in cursor.get_children()
                if c.kind == ci.CursorKind.PARM_DECL
            ]
            self.decl_locs[qname] = loc
            return
        if cursor.kind in (
            ci.CursorKind.STRUCT_DECL,
            ci.CursorKind.CLASS_DECL,
        ):
            # For class/struct decls we record a synthetic StructT so
            # downstream HIR->MIR can resolve ``Point`` -> ``StructT("Point")``
            # without re-walking the AST.
            self.var_types[loc] = StructT(name=name)

    def _record_call(self, cursor) -> None:
        try:
            ret = cursor.type
        except Exception:
            return
        if ret is None:
            return
        self.call_returns[_loc(cursor)] = _to_type(ret)

    def is_empty(self) -> bool:
        """True when the table carries no entries at all. Used as a
        fast-path guard by ``apply_ground_truth`` so non-C++ frontends
        (Python, Rust, ...) can pass through the same pipeline
        without paying the walk cost."""
        return not (
            self.var_types
            or self.func_returns
            or self.func_params
            or self.call_returns
            or self.decl_locs
        )


__all__ = ["TypeGroundTruth"]
