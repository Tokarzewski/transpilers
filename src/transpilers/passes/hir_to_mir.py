"""HIR -> MIR.

Algorithmic for the annotated subset. Where annotations are missing, we leave
UnknownT holes for a later inference pass (algorithmic dataflow first, LLM as
fallback). The boundary is non-negotiable: algorithmic passes must not invent
types they don't know.
"""

from __future__ import annotations

from transpilers.ir import hir, mir
from transpilers.ir.types import (
    BoolT,
    DictT,
    FloatT,
    TupleT,
    IntT,
    ListT,
    NoneT,
    RangeT,
    SimdT,
    StrT,
    StructT,
    Type,
    UnknownT,
)


# -- provenance helper -------------------------------------------------------
# Every MIR node built from a HIR node carries the HIR node's id in
# ``_hir_provenance_id``.  This helper sets it at construction time.


def _prov(mir_node: mir.MirNode, hir_node: hir.HirNode) -> mir.MirNode:
    """Copy ``hir_node._hir_node_id`` to ``mir_node._hir_provenance_id``."""
    mir_node._hir_provenance_id = hir_node._hir_node_id
    return mir_node


PYTHON_TYPE_MAP: dict[str, Type] = {
    "int": IntT(),
    "float": FloatT(),
    "bool": BoolT(),
    "str": StrT(),
    "None": NoneT(),
}


# Module-level registry of user-defined struct names. Populated during
# hir_to_mir so type-annotation resolution can recognize `Point` as
# StructT("Point") rather than UnknownT.
_KNOWN_STRUCTS: set[str] = set()
_STRUCT_FIELD_NAMES: dict[str, list[str]] = {}
_STRUCT_FIELD_TYPES: dict[str, dict[str, Type]] = {}
# A struct's explicit `__init__` param types (self excluded), keyed by struct
# name -- only present when the struct declares a real constructor. Used by
# the HirStructInit trailing-field padding below to pad to the
# *constructor's* declared arity rather than the *field* count: a
# constructor that computes some fields internally from fewer params than
# the struct has fields (e.g. OCCT's `gp_EulerSequence_Parameters(int
# theAx1, bool, bool, bool)` deriving 3 of its 6 fields from `theAx1`) must
# not have its call sites padded with fabricated extra args just because
# the field count is larger.
_STRUCT_CTOR_PARAM_TYPES: dict[str, list[Type]] = {}
# Module-level constants (`NAME: T = <literal>` at module scope). A free-name
# reference inside a function inlines the constant's value, so self-contained
# numeric modules transpile without a module-const declaration concept in the
# IR. Keyed name -> the HIR value expression (re-lowered per reference).
_MODULE_CONSTS: dict[str, "hir.HirNode"] = {}


def _default_init_for(ty: Type) -> mir.MirNode:
    if isinstance(ty, IntT):
        return mir.MirIntLiteral(value=0, ty=IntT())
    if isinstance(ty, FloatT):
        return mir.MirFloatLiteral(value=0.0, ty=FloatT())
    if isinstance(ty, BoolT):
        return mir.MirBoolLiteral(value=False, ty=BoolT())
    if isinstance(ty, StrT):
        return mir.MirStringLiteral(value="", ty=StrT())
    if isinstance(ty, ListT):
        # empty typed list (e.g. a std::stack/vector member's default ctor)
        return mir.MirList(elements=[], ty=ty)
    # Unknown / struct / dict defaults aren't expressible as one literal, and
    # this is an algorithmic pass -- it must not invent a value (a fabricated
    # `IntT(0)` here would previously masquerade as a real default for a
    # `dict[...]`/nested-struct field, so any later `self.field["key"]`-style
    # use would silently index an integer). Emit a `MirRaw` hole instead: every
    # backend already renders that as a `TODO[port]` stub rather than wrong
    # code.
    return mir.MirRaw(snippet="<default-init>", ty=UnknownT(hint=f"default init for {type(ty).__name__}"))


def hir_to_mir(module: hir.HirModule) -> mir.MirModule:
    functions: list[mir.MirFunction] = []
    structs: list[mir.MirStruct] = []
    _KNOWN_STRUCTS.clear()
    _STRUCT_FIELD_NAMES.clear()
    _STRUCT_FIELD_TYPES.clear()
    _STRUCT_CTOR_PARAM_TYPES.clear()
    # Reset per-module so synthesised foreach index names are deterministic:
    # a given module always lowers to byte-identical output regardless of how
    # many modules were processed earlier in the same process.
    global _FOREACH_INDEX_COUNTER
    _FOREACH_INDEX_COUNTER = 0
    _MODULE_CONSTS.clear()
    # First pass: register struct names so methods and fields can reference
    # them in their annotations, and collect module-level constants so function
    # bodies can inline free-name references to them.
    for node in module.body:
        if isinstance(node, hir.HirStruct):
            _KNOWN_STRUCTS.add(node.name)
        elif isinstance(node, hir.HirAssign) and node.augmented_op is None:
            _MODULE_CONSTS[node.target] = node.value
    for node in module.body:
        if isinstance(node, hir.HirFunction):
            functions.append(_lower_function(node))
        elif isinstance(node, hir.HirStruct):
            structs.append(_lower_struct(node))
        elif isinstance(node, hir.HirRaw):
            # Top-level unsupported construct (issue #50: a C++
            # template definition that the IR doesn't model). Emit a
            # placeholder function carrying the snippet, so the
            # downstream emit pass renders a TODO[port] stub. The
            # function name is best-effort ("raw_<n>") and the ground
            # truth still records the real signature under the
            # template's qualified name.
            functions.append(
                mir.MirFunction(
                    name=f"__raw_{len(functions)}",
                    params=[],
                    return_type=UnknownT(hint="top-level raw hole"),
                    body=[mir.MirRaw(snippet=node.snippet)],
                )
            )
    return mir.MirModule(functions=functions, structs=structs)


# -- provenance-aware lowering helpers ---------------------------------------
# Every ``_lower_*`` function receives a HIR node and assigns its id to the
# returned MIR sub-tree.  This wrapper sets ``_hir_provenance_id`` on a
# single MIR node; synthetic nodes that have no corresponding HIR node
# (generated indices, default inits) remain at id 0.


def _p(node: mir.MirNode, hir_node: hir.HirNode) -> mir.MirNode:
    node._hir_provenance_id = hir_node._hir_node_id
    return node


def _lower_struct(s: hir.HirStruct) -> mir.MirStruct:
    fields = [_p(mir.MirParam(name=f.name, ty=_resolve_annotation(f.annotation)), f) for f in s.fields]
    methods = [_lower_function(m) for m in s.methods]
    _STRUCT_FIELD_NAMES[s.name] = [f.name for f in fields]
    _STRUCT_FIELD_TYPES[s.name] = {f.name: f.ty for f in fields}
    ctor = next((m for m in methods if m.name == "__init__"), None)
    if ctor is not None:
        _STRUCT_CTOR_PARAM_TYPES[s.name] = [p.ty for p in ctor.params[1:]]
    return _p(mir.MirStruct(name=s.name, fields=fields, methods=methods), s)


def _lower_function(fn: hir.HirFunction) -> mir.MirFunction:
    params = [_p(mir.MirParam(name=p.name, ty=_resolve_annotation(p.annotation)), p) for p in fn.params]
    ret_ty = _resolve_annotation(fn.return_annotation)
    env: dict[str, Type] = {p.name: p.ty for p in params}
    body = [_lower_stmt(n, env) for n in fn.body]
    return _p(mir.MirFunction(name=fn.name, params=params, return_type=ret_ty, body=body,
                              is_static=fn.is_static), fn)


def _lower_stmt(node: hir.HirNode, env: dict[str, Type]) -> mir.MirNode:
    if isinstance(node, hir.HirRaw):
        return _p(mir.MirRaw(snippet=node.snippet), node)
    if isinstance(node, hir.HirReturn):
        return _p(mir.MirReturn(value=_lower_expr(node.value, env) if node.value else None), node)
    if isinstance(node, hir.HirBreak):
        return _p(mir.MirBreak(), node)
    if isinstance(node, hir.HirContinue):
        return _p(mir.MirContinue(), node)
    if isinstance(node, hir.HirFieldAssign):
        return _p(mir.MirFieldAssign(
            obj=_lower_expr(node.obj, env),
            field=node.field,
            value=_lower_expr(node.value, env),
        ), node)
    if isinstance(node, hir.HirSubscriptAssign):
        return _p(mir.MirSubscriptAssign(
            obj=_lower_expr(node.obj, env),
            index=_lower_expr(node.index, env),
            value=_lower_expr(node.value, env),
        ), node)
    if isinstance(node, hir.HirAssign):
        value = _lower_expr(node.value, env)
        ann_ty = _resolve_annotation(node.annotation) if node.annotation else None
        # Augmented ops desugar to `target = target <op> value`; we keep the
        # original op symbol on MIR so emission can pick `+=` form for readability.
        ty = ann_ty if ann_ty is not None and not isinstance(ann_ty, UnknownT) else _type_of(value)
        # Propagate the binding's list element type onto an empty list literal,
        # so each backend's MIR→LIR pass can render the element type even when
        # `[]` carries no inference signal of its own.
        from transpilers.ir.types import ListT
        if (
            isinstance(value, mir.MirList)
            and isinstance(ty, ListT)
            and isinstance(value.ty, ListT)
            and isinstance(value.ty.elem, UnknownT)
        ):
            value.ty = ty
        if node.target not in env:
            env[node.target] = ty
        return _p(mir.MirAssign(target=node.target, value=value, ty=ty, augmented_op=node.augmented_op), node)
    if isinstance(node, hir.HirIf):
        return _p(mir.MirIf(
            test=_lower_expr(node.test, env),
            body=[_lower_stmt(n, env) for n in node.body],
            orelse=[_lower_stmt(n, env) for n in node.orelse],
        ), node)
    if isinstance(node, hir.HirWhile):
        return _p(mir.MirWhile(
            test=_lower_expr(node.test, env),
            body=[_lower_stmt(n, env) for n in node.body],
        ), node)
    if isinstance(node, hir.HirFor):
        return _lower_for(node, env)
    if isinstance(node, hir.HirForEach):
        return _lower_foreach(node, env)
    # Bare expression statement (rare in our subset, but legal): wrap as no-op-ish.
    return _lower_expr(node, env)


def _lower_for(node: hir.HirFor, env: dict[str, Type]) -> mir.MirForRange:
    if not (isinstance(node.iter, hir.HirCall) and node.iter.func == "range"):
        raise NotImplementedError("for-loop iter must be range(...) in the initial subset")
    args = [_lower_expr(a, env) for a in node.iter.args]
    if len(args) == 1:
        start = mir.MirIntLiteral(value=0, ty=IntT())
        stop, step = args[0], None
    elif len(args) == 2:
        start, stop, step = args[0], args[1], None
    elif len(args) == 3:
        start, stop, step = args[0], args[1], args[2]
    else:
        raise NotImplementedError(f"range arity {len(args)}")
    env[node.target] = IntT()
    body = [_lower_stmt(n, env) for n in node.body]
    # The for-loop node itself carries provenance from the HirFor.
    return _p(mir.MirForRange(target=node.target, start=start, stop=stop, step=step, body=body), node)


# Monotonic counter for synthesised plain-foreach index names. A plain index is
# named uniquely so nested foreach loops can't shadow each other.
_FOREACH_INDEX_COUNTER = 0


def _lower_foreach(node: hir.HirForEach, env: dict[str, Type]) -> mir.MirForRange:
    """Desugar `for v in seq` / `for i, v in enumerate(seq)` to an indexed
    `range(len(seq))` loop with a typed `v = seq[i]` binding at the body head.

    The binding's type is the iterable's element type resolved from `env`, so it
    is typed by construction — every backend (not just type-inferring ones)
    sees a concrete element type when the iterable's element type is known.
    """
    global _FOREACH_INDEX_COUNTER
    seq_ty = env.get(node.iterable, UnknownT(hint=f"name {node.iterable}"))
    elem_ty: Type = (
        seq_ty.elem
        if isinstance(seq_ty, ListT)
        else UnknownT(hint=f"foreach over non-list {node.iterable}")
    )

    if node.index_name is not None:
        index_name = node.index_name
    else:
        index_name = f"__xpile_idx_{_FOREACH_INDEX_COUNTER}"
        _FOREACH_INDEX_COUNTER += 1
    env[index_name] = IntT()

    # `value = seq[index]` — typed binding synthesised at the body head.
    # Synthetic nodes carry the ForEach's provenance id.
    binding = _p(mir.MirAssign(
        target=node.value_name,
        value=_p(mir.MirSubscript(
            value=_p(mir.MirName(name=node.iterable, ty=seq_ty), node),
            index=_p(mir.MirName(name=index_name, ty=IntT()), node),
            ty=elem_ty,
        ), node),
        ty=elem_ty,
        augmented_op=None,
    ), node)
    env[node.value_name] = elem_ty
    body = [binding, *(_lower_stmt(n, env) for n in node.body)]

    stop = _p(mir.MirCall(
        func="len", args=[_p(mir.MirName(name=node.iterable, ty=seq_ty), node)], ty=IntT()
    ), node)
    return _p(mir.MirForRange(
        target=index_name,
        start=_p(mir.MirIntLiteral(value=0, ty=IntT()), node),
        stop=stop,
        step=None,
        body=body,
    ), node)


def _lower_expr(node: hir.HirNode, env: dict[str, Type]) -> mir.MirNode:
    if isinstance(node, hir.HirRaw):
        return _p(mir.MirRaw(snippet=node.snippet), node)
    if isinstance(node, hir.HirBinOp):
        left = _lower_expr(node.left, env)
        right = _lower_expr(node.right, env)
        return _p(mir.MirBinOp(op=node.op, left=left, right=right, ty=_binop_type(node.op, _type_of(left), _type_of(right))), node)
    if isinstance(node, hir.HirCompare):
        return _p(mir.MirCompare(
            op=node.op, left=_lower_expr(node.left, env), right=_lower_expr(node.right, env), ty=BoolT()
        ), node)
    if isinstance(node, hir.HirBoolOp):
        return _p(mir.MirBoolOp(
            op=node.op, left=_lower_expr(node.left, env), right=_lower_expr(node.right, env), ty=BoolT()
        ), node)
    if isinstance(node, hir.HirUnaryOp):
        operand = _lower_expr(node.operand, env)
        ty = BoolT() if node.op == "not" else _type_of(operand)
        return _p(mir.MirUnaryOp(op=node.op, operand=operand, ty=ty), node)
    if isinstance(node, hir.HirName):
        # A free name (not a local/param) that matches a module-level constant
        # inlines that constant's value — re-lowered here so it carries its
        # literal type. Locals/params shadow constants (checked via `env`).
        if node.name not in env and node.name in _MODULE_CONSTS:
            return _lower_expr(_MODULE_CONSTS[node.name], env)
        return _p(mir.MirName(name=node.name, ty=env.get(node.name, UnknownT(hint=f"name {node.name}"))), node)
    if isinstance(node, hir.HirIntLiteral):
        return _p(mir.MirIntLiteral(value=node.value, ty=IntT()), node)
    if isinstance(node, hir.HirFloatLiteral):
        return _p(mir.MirFloatLiteral(value=node.value, ty=FloatT()), node)
    if isinstance(node, hir.HirBoolLiteral):
        return _p(mir.MirBoolLiteral(value=node.value, ty=BoolT()), node)
    if isinstance(node, hir.HirStringLiteral):
        return _p(mir.MirStringLiteral(value=node.value, ty=StrT()), node)
    if isinstance(node, hir.HirNullLiteral):
        return _p(mir.MirNullLiteral(), node)
    if isinstance(node, hir.HirCall):
        return _lower_call(node, env)
    if isinstance(node, hir.HirList):
        elements = [_lower_expr(e, env) for e in node.elements]
        elem_ty = _type_of(elements[0]) if elements else UnknownT(hint="empty list literal")
        return _p(mir.MirList(elements=elements, ty=ListT(elem=elem_ty)), node)
    if isinstance(node, hir.HirSubscript):
        value = _lower_expr(node.value, env)
        index = _lower_expr(node.index, env)
        value_ty = _type_of(value)
        ty: Type = value_ty.elem if isinstance(value_ty, ListT) else UnknownT(hint="subscript on non-list")
        return _p(mir.MirSubscript(value=value, index=index, ty=ty), node)
    if isinstance(node, hir.HirFieldAccess):
        # Type of a field access stays Unknown for now — a field-resolution
        # pass would look up the struct's field types. Out of scope for the
        # minimal struct slice; the LIR emitters handle Unknown by leaving
        # the field bare and letting target inference work.
        return _p(mir.MirFieldAccess(value=_lower_expr(node.value, env), field=node.field), node)
    if isinstance(node, hir.HirMethodCall):
        return _p(mir.MirMethodCall(
            receiver=_lower_expr(node.receiver, env),
            method=node.method,
            args=[_lower_expr(a, env) for a in node.args],
        ), node)
    if isinstance(node, hir.HirStructInit):
        # Pair positional ctor args with the struct's declared field names so
        # every target's emitter can render named-field form when it wants.
        field_names = _STRUCT_FIELD_NAMES.get(node.name, [])
        field_types = _STRUCT_FIELD_TYPES.get(node.name, {})
        lowered_args = [_lower_expr(a, env) for a in node.args]
        # A struct with a real user-defined constructor (member-init list
        # computing some fields from fewer params than the struct has
        # fields -- e.g. OCCT's `gp_EulerSequence_Parameters`, whose ctor
        # derives 3 of its 6 fields from a single `theAx1` argument) must be
        # padded to *its own declared arity*, not the field count: padding
        # to field count fabricates extra args the constructor was never
        # declared to take. Structs without an explicit ctor (the common
        # per-field aggregate case, e.g. `gp_XYZ(x, y, z)`) fall back to the
        # field-count expectation, unchanged from before.
        ctor_param_types = _STRUCT_CTOR_PARAM_TYPES.get(node.name)
        expected_types = ctor_param_types if ctor_param_types is not None else [
            field_types.get(n, UnknownT()) for n in field_names
        ]
        pairs: list[tuple[str, mir.MirNode]] = []
        # Pair positionally; when field names aren't registered yet (e.g. a
        # struct's own method constructs the struct before its fields are
        # recorded), keep the args positionally (name "") rather than dropping
        # them. Trailing fields with no arg get a default (C++ partial init).
        for i in range(max(len(expected_types), len(lowered_args))):
            fname = field_names[i] if i < len(field_names) else ""
            if i < len(lowered_args):
                pairs.append((fname, lowered_args[i]))
            else:
                ty = expected_types[i] if i < len(expected_types) else UnknownT()
                pairs.append((fname, _default_init_for(ty)))
        return _p(mir.MirStructInit(
            name=node.name, field_values=pairs, ty=StructT(name=node.name)
        ), node)
    raise NotImplementedError(f"HIR expr {type(node).__name__}")


def _numeric_result_ty(arg_tys: list[Type]) -> Type:
    """Result type of a type-preserving numeric builtin (abs/min/max). Float
    dominates int; an all-int set stays int; otherwise preserve the first known
    numeric arg, falling back to a hole."""
    if any(isinstance(t, FloatT) for t in arg_tys):
        return FloatT()
    if arg_tys and all(isinstance(t, IntT) for t in arg_tys):
        return IntT()
    for t in arg_tys:
        if isinstance(t, (IntT, FloatT)):
            return t
    return UnknownT(hint="numeric builtin result")


def _lower_call(node: hir.HirCall, env: dict[str, Type]) -> mir.MirNode:
    args = [_lower_expr(a, env) for a in node.args]
    if node.func == "len":
        return _p(mir.MirCall(func="len", args=args, ty=IntT()), node)
    if node.func == "range":
        return _p(mir.MirCall(func="range", args=args, ty=RangeT()), node)
    # Common builtins across source languages — assign a plausible return
    # type so call results flow through inference rather than blocking on
    # UnknownT. Heuristic, not exhaustive; the stdlib_maps/ tables are the
    # long-term home for richer mappings.
    if node.func in ("abs", "min", "max"):
        # Type-preserving: abs(float)->float, max(float, float)->float. A single
        # list arg (min([...])/max([...])) yields its element type.
        tys = [_type_of(a) for a in args]
        if node.func in ("min", "max") and len(tys) == 1 and isinstance(tys[0], ListT):
            ty: Type = tys[0].elem
        else:
            ty = _numeric_result_ty(tys)
        return _p(mir.MirCall(func=node.func, args=args, ty=ty), node)
    if node.func == "sum":
        # sum of a list -> the list's element type (sum([floats]) is float).
        t0 = _type_of(args[0]) if args else IntT()
        ty = t0.elem if isinstance(t0, ListT) else IntT()
        return _p(mir.MirCall(func="sum", args=args, ty=ty), node)
    if node.func in ("int", "ord", "id", "hash"):
        return _p(mir.MirCall(func=node.func, args=args, ty=IntT()), node)
    if node.func in ("float", "round"):
        return _p(mir.MirCall(func=node.func, args=args, ty=FloatT()), node)
    if node.func == "bool":
        return _p(mir.MirCall(func=node.func, args=args, ty=BoolT()), node)
    if node.func in ("str", "repr", "format"):
        return _p(mir.MirCall(func=node.func, args=args, ty=StrT()), node)
    if node.func in ("print", "println"):
        return _p(mir.MirCall(func=node.func, args=args, ty=NoneT()), node)
    return _p(mir.MirCall(func=node.func, args=args, ty=UnknownT(hint=f"call {node.func}")), node)


def _resolve_annotation(ann: str | None) -> Type:
    if ann is None:
        return UnknownT(hint="missing Python annotation")
    if ann in PYTHON_TYPE_MAP:
        return PYTHON_TYPE_MAP[ann]
    if ann.startswith("list[") and ann.endswith("]"):
        inner = ann[len("list[") : -1]
        return ListT(elem=_resolve_annotation(inner))
    if ann.startswith("tuple[") and ann.endswith("]"):
        inner = ann[len("tuple[") : -1]
        parts, depth, last = [], 0, 0
        for i, ch in enumerate(inner):
            if ch in "[<":
                depth += 1
            elif ch in "]>":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append(inner[last:i].strip())
                last = i + 1
        parts.append(inner[last:].strip())
        return TupleT(elems=tuple(_resolve_annotation(p) for p in parts))
    if ann.startswith("dict[") and ann.endswith("]"):
        inner = ann[len("dict[") : -1]
        depth = 0
        for i, ch in enumerate(inner):  # split K, V at the top-level comma
            if ch in "[<":
                depth += 1
            elif ch in "]>":
                depth -= 1
            elif ch == "," and depth == 0:
                return DictT(key=_resolve_annotation(inner[:i].strip()),
                             value=_resolve_annotation(inner[i + 1:].strip()))
        return UnknownT(hint=f"malformed dict annotation {ann!r}")
    if ann.startswith("simd[") and ann.endswith("]"):
        # `simd[<elem>, <lanes>]` — a SIMD vector type. The C++ frontend
        # produces these strings when it sees Intel/ARM vector typedefs.
        inner = ann[len("simd[") : -1]
        elem_text, _, lanes_text = inner.rpartition(",")
        elem = _resolve_annotation(elem_text.strip())
        lanes = int(lanes_text.strip())
        return SimdT(elem=elem, lanes=lanes)
    if ann in _KNOWN_STRUCTS:
        return StructT(name=ann)
    # Unknown identifier-shaped annotation looking like a class name
    # (e.g., third-party `Charset`, `Scanner`) — treat as opaque struct.
    # Lossy: we don't know the actual layout, but downstream emission can
    # render the name and let the target's typechecker decide.
    if ann.isidentifier() and ann[0].isupper():
        return StructT(name=ann)
    return UnknownT(hint=f"unknown annotation {ann!r}")


def _type_of(node: mir.MirNode) -> Type:
    return getattr(node, "ty", UnknownT())


def _binop_type(op: str, lt: Type, rt: Type) -> Type:
    if isinstance(lt, IntT) and isinstance(rt, IntT):
        return IntT()
    if isinstance(lt, (IntT, FloatT)) and isinstance(rt, (IntT, FloatT)):
        return FloatT()
    if op == "+" and isinstance(lt, StrT) and isinstance(rt, StrT):
        return StrT()
    return UnknownT(hint=f"binop {op}")
