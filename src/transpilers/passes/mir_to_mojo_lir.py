"""MIR -> Mojo LIR.

Mojo's binding model is `var`-only (no `let`/`mut` split). Every locally
introduced name emits `var ...`; reassignments emit plain `<name> = ...`.
Loop variables are scoped to the loop body and don't need a declaration.

String concat: Mojo `String` supports `+`, so we can emit it directly without
the format!() detour Rust needs.

Structure is shared with the other backends via ``_mir_lower_base``; this
module supplies the Mojo spec plus the cmath/SIMD/pow_N call rewrites and the
`import math` bookkeeping.
"""

from __future__ import annotations

import re

from transpilers.ir import lir, mir
from transpilers.ir.types import (
    BoolT,
    DictT,
    FloatT,
    IntT,
    ListT,
    TupleT,
    NoneT,
    SimdT,
    StrT,
    StructT,
    Type,
    UnknownT,
)

from ._mir_lower_base import MirLoweringBase


# <cmath> functions that live in Mojo's `math` module -> emit as `math.<name>`
# (with an `import math` header). Names not here that are Mojo builtins map to
# the builtin directly (no import).
# NOTE: `fmod` is intentionally NOT here — Mojo's `math` module has no `fmod`,
# and Mojo's `%` operator follows Python sign-of-divisor semantics, not C
# `fmod`'s sign-of-dividend. It is lowered to `a - b*trunc(a/b)` below.
_MATH_FNS = frozenset({
    "exp", "log", "log2", "log10", "sqrt", "cbrt", "pow", "hypot",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "ceil", "floor", "trunc",
    "expm1", "log1p", "erf", "copysign",  # used in EnergyPlus; present in Mojo std.math
})
# Python numeric constructors map to Mojo's scalar types. `float`/`int`/`bool`
# are not Mojo builtins (`use of unknown declaration 'float'`); `Float64(x)` /
# `Int(x)` / `Bool(x)` are. `int(float)` truncates toward zero in both Python
# and Mojo, so the cast semantics match.
_BUILTIN_MAP = {
    "fabs": "abs", "fmin": "min", "fmax": "max",
    "float": "Float64", "int": "Int", "bool": "Bool",
}


def _foreign_struct_names(module: mir.MirModule) -> set[str]:
    """Every `StructT` name reachable from a function/method signature or a
    struct field's declared type -- covers params, return types, and
    fields, which is where a type the parser resolved via a project-
    preamble shim (see docs/occt_preamble.hpp) but the user's own code
    never declared as a struct will actually show up. Doesn't walk into
    local-variable declarations inside function bodies; that would need a
    param/field/return type carrying the same name too in every practical
    case this is meant to catch."""
    names: set[str] = set()

    def note(ty) -> None:
        if isinstance(ty, StructT):
            names.add(ty.name)

    def scan_fn(fn: mir.MirFunction) -> None:
        note(fn.return_type)
        for p in fn.params:
            note(p.ty)

    for fn in module.functions:
        scan_fn(fn)
    for struct in module.structs:
        for f in struct.fields:
            note(f.ty)
        for m in struct.methods:
            scan_fn(m)
    return names


def _uses_dict(fn) -> bool:
    """True if the function signature or body involves a Dict (whose subscript
    read raises in Mojo, requiring `raises`)."""
    if isinstance(getattr(fn, "return_type", None), DictT):
        return True
    if any(isinstance(p.ty, DictT) for p in fn.params):
        return True

    def walk(nodes) -> bool:
        for n in nodes:
            if isinstance(getattr(n, "ty", None), DictT):
                return True
            for attr in ("body", "orelse"):
                sub = getattr(n, attr, None)
                if isinstance(sub, list) and walk(sub):
                    return True
        return False

    return walk(fn.body)


def _zero_literal(elem: str):
    """Default element value for a sized container (vector<T>(n))."""
    if elem in ("Float64", "Float32", "Float16"):
        return lir.MojoFloatLiteral(value=0.0)
    if elem == "Bool":
        return lir.MojoBoolLiteral(value=False)
    return lir.MojoIntLiteral(value=0)


class _MojoLowering(MirLoweringBase):
    prefix = "Mojo"
    module_cls = lir.MojoModule

    def __init__(self) -> None:
        super().__init__()
        self._used_math: set[str] = set()
        self._field_types: dict[str, dict[str, Type]] = {}
        self._method_return_types: dict[str, dict[str, Type]] = {}
        self._mutating_names: frozenset[str] = frozenset()

    def type_str(self, ty: Type) -> str:
        return _mojo_type(ty)

    def _resolved_ty(self, node: mir.MirNode) -> Type:
        """`node.ty` if infer_types resolved it, else best-effort lookup for
        node shapes whose *own* type infer_types never fills in: a field
        access (no struct-field table to consult) or a method call (MIR
        carries no method-signature table either -- `hir_to_mir` never sets
        `ty=` when lowering a HirMethodCall at all). Recurses through
        receiver chains (`a.Direction().coord`: field-access-of-method-call)
        so a copy gets inserted for the field of a temporary just as
        reliably as for the field of a plain name.
        """
        ty = getattr(node, "ty", None)
        if ty is not None and not isinstance(ty, UnknownT):
            return ty
        if isinstance(node, mir.MirFieldAccess):
            recv_ty = self._resolved_ty(node.value)
            if isinstance(recv_ty, StructT):
                return self._field_types.get(recv_ty.name, {}).get(node.field, UnknownT())
        if isinstance(node, mir.MirMethodCall):
            recv_ty = self._resolved_ty(node.receiver)
            if isinstance(recv_ty, StructT):
                return self._method_return_types.get(recv_ty.name, {}).get(node.method, UnknownT())
        return ty if ty is not None else UnknownT()

    def lower_arg(self, node: mir.MirNode):
        # Passing a bare struct-typed name/field ("f(self.field)", "f(v)")
        # into a call hits the identical ImplicitlyCopyable restriction as
        # returning/assigning one -- a call argument is a by-value hand-off
        # too, whichever of the callee's own parameter conventions
        # (default borrow, `var`, `mut`) ends up receiving it. Unlike
        # return/assign, this needs no `ret`/declared-type context: any
        # bare reference to an existing struct value reaching a call
        # boundary needs the copy, full stop.
        val = self.lower_expr(node)
        if isinstance(val, (lir.MojoName, lir.MojoFieldAccess)) and isinstance(self._resolved_ty(node), StructT):
            return lir.MojoMethodCall(receiver=val, method="copy", args=[])
        return val

    def lower_module(self, module: mir.MirModule) -> lir.MojoModule:
        self._used_math.clear()
        self._field_types = {s.name: {f.name: f.ty for f in s.fields} for s in module.structs}
        self._method_return_types = {
            s.name: {m.name: m.return_type for m in s.methods} for s in module.structs
        }
        self._mutating_names = _compute_mir_mutating_method_names(module)
        items: list[lir.LirNode] = []
        # Struct types the parser resolved (e.g. via a project-preamble shim
        # for a third-party library -- see docs/occt_preamble.hpp) but that
        # the user's own code never declared: emit a minimal opaque
        # placeholder for each, or every reference to it is a compile-time
        # "use of unknown declaration" in the *output*, even though libclang
        # itself was perfectly happy to resolve it during parsing. One dummy
        # field (rather than none) matters: an empty struct doesn't satisfy
        # `ImplicitlyCopyable` the same way in this Mojo version.
        declared = {s.name for s in module.structs}
        for name in sorted(_foreign_struct_names(module) - declared):
            items.append(lir.MojoStruct(name=name, fields=[("_opaque", "Int")], methods=[]))
        for struct in module.structs:
            items.extend(self.lower_struct_items(struct))
        for fn in module.functions:
            items.append(self.lower_function(fn))
        # Mojo 1.0 requires the `std.` prefix: `from std.math import sqrt, exp`
        # (bare `from math import` is deprecated and warns). Explicit names, not
        # `import std.math` + qualified access. Only actually-used names imported.
        imports = ([f"from std.math import {', '.join(sorted(self._used_math))}"]
                   if self._used_math else [])
        return lir.MojoModule(items=items, imports=imports)

    # -- function signature: var/mut param decoration --------------------- #

    def lower_params(self, fn: mir.MirFunction):
        # Mojo args are read-only by default; reassigning to `n` errors unless
        # the param is declared `var n: …`. Subscript-assigning to `xs` (e.g.
        # `xs[i] = v`) requires `mut xs: …` instead. Scan the body for both.
        param_names = {p.name for p in fn.params}
        var_params, mut_params = _params_reassigned(fn.body, param_names, self._mutating_names)
        params = []
        for p in fn.params:
            if p.name in mut_params:
                params.append((f"mut {p.name}", _mojo_type(p.ty)))
            elif p.name in var_params:
                params.append((f"var {p.name}", _mojo_type(p.ty)))
            else:
                params.append((p.name, _mojo_type(p.ty)))
        return params

    # -- statements -------------------------------------------------------- #

    def lower_stmt(self, node: mir.MirNode, declared: set[str], mut: set[str]):
        # Drop bare `assert(...)` statements: Mojo has no free `assert` function,
        # and EnergyPlus uses C `assert` only as a debug-build precondition check
        # (compiled out under NDEBUG) — it never affects the returned value. The
        # behavioral oracle is built with -DNDEBUG so this matches the C++ result.
        if isinstance(node, mir.MirCall) and node.func == "assert":
            return None
        return super().lower_stmt(node, declared, mut)

    # -- assign: var vs bare reassign ------------------------------------- #

    def lower_assign(self, node: mir.MirAssign, declared: set[str], mut: set[str]):
        if node.augmented_op is not None:
            rhs = lir.MojoBinOp(
                op=node.augmented_op,
                left=lir.MojoName(name=node.target),
                right=self.lower_expr(node.value),
            )
            return lir.MojoReassign(name=node.target, value=rhs)
        if node.target in declared:
            # Same ImplicitlyCopyable restriction as the fresh-declaration
            # branch below (and lower_return/lower_field_assign): a plain
            # reassignment (`Tloc = T.loc;` reached a second time, e.g.
            # across if/switch branches sharing one declared local) is just
            # as much a by-value hand-off of an existing reference as a
            # var-decl initializer is. This branch had no copy-insertion at
            # all before -- every reassignment from a bare struct-typed
            # name/field silently fell through to the same compile error.
            value = self.lower_arg(node.value)
            return lir.MojoReassign(name=node.target, value=value)
        declared.add(node.target)
        ty = _mojo_type(node.ty) if not isinstance(node.ty, UnknownT) else None
        value = self._lower_container_ctor(node, ty)
        if value is None:
            # Same "bare name of a non-ImplicitlyCopyable type needs an
            # explicit .copy()" rule as lower_return/lower_arg above --
            # `Mat aCopy = *this;` / `Mat aCopy = otherMat;` hit the
            # identical Mojo 1.0.0b2 restriction a struct-returning
            # function or a call argument does, just via a var-decl.
            value = self.lower_arg(node.value)
        return lir.MojoVar(name=node.target, ty=ty, value=value)

    def lower_field_assign(self, node: mir.MirFieldAssign):
        # `self.vydir = otherDirField` hits the identical ImplicitlyCopyable
        # restriction as a var-decl, reassign, return, or call-argument
        # hand-off of a bare struct-typed name/field (see lower_assign /
        # lower_return / lower_arg above) -- the base implementation has no
        # copy-insertion at all (correct for Rust/Zig, which don't need
        # it), so Mojo needs its own override.
        value = self.lower_arg(node.value)
        return lir.MojoFieldAssign(obj=self.lower_expr(node.obj), field=node.field, value=value)

    def _lower_container_ctor(self, node, ty):
        """C++ container construction -> Mojo, using the declared var type.
        `vector<T> v;` -> List[T](); `vector<T> v(n[, fill])` -> [fill_or_0] * n.
        """
        v = node.value
        if not (ty and isinstance(v, mir.MirCall)):
            return None
        if v.func == "__cpp_overloaded_op__" and not v.args:
            return lir.MojoCall(func=ty, args=[])          # empty default-construct
        if v.func == "__vector_fill__" and v.args:
            elem = ty[5:-1] if ty.startswith("List[") else ""
            size = self.lower_expr(v.args[0])
            fill = self.lower_expr(v.args[1]) if len(v.args) >= 2 else _zero_literal(elem)
            return lir.MojoBinOp(op="*", left=lir.MojoList(elements=[fill]), right=size)
        return None

    # -- expressions ------------------------------------------------------- #

    def lower_expr_special(self, node: mir.MirNode):
        if isinstance(node, mir.MirBinOp) and node.op == "//":
            return lir.MojoBinOp(op="//", left=self.lower_expr(node.left), right=self.lower_expr(node.right))
        return None

    def lower_binop(self, node: mir.MirBinOp):
        return lir.MojoBinOp(op=node.op, left=self.lower_expr(node.left), right=self.lower_expr(node.right))

    def lower_boolop(self, node: mir.MirBoolOp):
        return lir.MojoBoolOp(op=node.op, left=self.lower_expr(node.left), right=self.lower_expr(node.right))

    def lower_unary(self, node: mir.MirUnaryOp):
        op = "not" if node.op == "not" else "-"
        return lir.MojoUnary(op=op, operand=self.lower_expr(node.operand))

    def lower_null(self, node: mir.MirNullLiteral):
        return lir.MojoName(name="None")

    def lower_list(self, node: mir.MirList):
        return lir.MojoList(elements=[self.lower_expr(e) for e in node.elements])

    def lower_subscript(self, node: mir.MirSubscript):
        # Mojo String indexing is `s[byte=i]` (returns a StringSlice), not `s[i]`.
        byte = isinstance(getattr(node.value, "ty", None), StrT)
        return lir.MojoIndex(value=self.lower_expr(node.value),
                             index=self.lower_expr(node.index), byte=byte)

    def lower_function(self, fn: mir.MirFunction):
        self._cur_ret = self.return_type(fn)
        return super().lower_function(fn)

    def make_function(self, fn, params, ret, body):
        f = super().make_function(fn, params, ret, body)
        # Mojo Dict subscript reads raise (KeyError); a function using a Dict
        # needs `raises`. Over-declaring raises is harmless.
        f.raises = _uses_dict(fn)
        f.is_static = getattr(fn, "is_static", False)
        return f

    def lower_return(self, node: mir.MirReturn):
        ret = getattr(self, "_cur_ret", None) or ""
        # `return {};` -> typed empty container constructor (List[T]()/Dict[K,V]()).
        if (isinstance(node.value, mir.MirCall) and node.value.func == "__cpp_overloaded_op__"
                and not node.value.args
                and (ret.startswith("List[") or ret.startswith("Dict[") or ret == "String")):
            return lir.MojoReturn(value=lir.MojoCall(func=ret, args=[]))
        val = self.lower_expr(node.value) if node.value else None
        # `return {a, b};` for a tuple/pair return -> (a, b).
        if isinstance(val, lir.MojoList) and ret.startswith("Tuple["):
            return lir.MojoReturn(value=lir.MojoTuple(elements=val.elements))
        # `return {a, b};` where the function returns a struct -> Type(a, b). The
        # init-list lowers to a MojoList; a struct return type (not a container /
        # primitive) means it's really a fieldwise constructor.
        if (isinstance(val, lir.MojoList) and ret
                and not ret.startswith(("List[", "Dict[", "Tuple[", "SIMD["))
                and ret not in ("Int", "Float64", "Bool", "String", "None")):
            return lir.MojoReturn(value=lir.MojoStructInit(
                name=ret, field_values=[("", e) for e in val.elements]))
        # List/Dict/String/struct values aren't ImplicitlyCopyable in this Mojo
        # version: `return localVar` needs an explicit copy (or `^`). Only a
        # bare name needs it — rvalues (`[0]*n`, calls, struct-init) are
        # already owned temporaries. Struct-ness is checked off the returned
        # value's own resolved type (not the return-type spelling), so this
        # also covers passing a same-typed parameter straight through
        # (confirmed against the real Mojo 1.0.0b2 compiler: even a plain
        # `def f(v: Vec) -> Vec: return v` needs this, not just container/
        # String types — a struct with real fields hits the exact same
        # "cannot be implicitly copied, does not conform to
        # 'ImplicitlyCopyable'" error).
        needs_copy = (
            ret.startswith("List[") or ret.startswith("Dict[") or ret == "String"
            or isinstance(self._resolved_ty(node.value), StructT)
        )
        # A bare field access (`return self.vdir`) is exactly as much a
        # "reference to an existing owned value" as a bare local name is —
        # same ImplicitlyCopyable restriction, same fix. Only MojoName was
        # covered before, which missed the very common "return one of my
        # own struct-typed fields" pattern entirely.
        if isinstance(val, (lir.MojoName, lir.MojoFieldAccess)) and needs_copy:
            val = lir.MojoMethodCall(receiver=val, method="copy", args=[])
        return lir.MojoReturn(value=val)

    def lower_method_call(self, node: mir.MirMethodCall):
        # These STL-container method rewrites key off the method NAME, so they
        # must NOT fire when the receiver is a user struct that happens to have a
        # method of the same name (push/top/empty/size/...). Guard by receiver type.
        is_struct = isinstance(getattr(node.receiver, "ty", None), StructT)
        if not is_struct:
            # Mojo containers/String have no `.size()`/`.length()` — use `len(x)`.
            if node.method in ("size", "length") and not node.args:
                return lir.MojoCall(func="len", args=[self.lower_expr(node.receiver)])
            # map/set membership: m.count(k) -> `k in m`.
            if node.method == "count" and len(node.args) == 1:
                return lir.MojoCompare(op="in", left=self.lower_expr(node.args[0]),
                                       right=self.lower_expr(node.receiver))
            # vector::push_back / emplace_back / stack::push -> List.append
            if node.method in ("push_back", "emplace_back", "push") and len(node.args) == 1:
                return lir.MojoMethodCall(
                    receiver=self.lower_expr(node.receiver), method="append",
                    args=[self.lower_arg(node.args[0])])
            # stack/queue: top()/back() -> v[len(v)-1]; front() -> v[0]
            # (Mojo has no negative indexing on List).
            if node.method in ("top", "back") and not node.args:
                recv = self.lower_expr(node.receiver)
                return lir.MojoIndex(value=recv, index=lir.MojoBinOp(
                    op="-", left=lir.MojoCall(func="len", args=[recv]), right=lir.MojoIntLiteral(value=1)))
            if node.method == "front" and not node.args:
                return lir.MojoIndex(value=self.lower_expr(node.receiver), index=lir.MojoIntLiteral(value=0))
            # empty() -> len(v) == 0
            if node.method == "empty" and not node.args:
                return lir.MojoCompare(op="==", left=lir.MojoCall(
                    func="len", args=[self.lower_expr(node.receiver)]), right=lir.MojoIntLiteral(value=0))
        return super().lower_method_call(node)

    def lower_call(self, node: mir.MirCall):
        args = [self.lower_arg(a) for a in node.args]
        if node.func == "__ternary__" and len(args) == 3:
            return _MojoIfExpr(test=args[0], then_=args[1], else_=args[2])
        # std::to_string(x) -> String(x) (int/float -> string, common in EP output)
        if node.func == "to_string" and len(args) == 1:
            return lir.MojoCall(func="String", args=args)
        # vector(it1, it2) iterator-range ctor -> c[lo:hi]
        if node.func == "__vector_slice__" and len(args) == 3:
            return lir.MojoSlice(value=args[0], lo=args[1], hi=args[2])
        # std::min({a,b,c}) / max -> fold to nested min(a, min(b, c)) (Mojo is 2-arg)
        if node.func in ("min", "max") and len(node.args) == 1 and isinstance(node.args[0], mir.MirList):
            elems = [self.lower_expr(e) for e in node.args[0].elements]
            if elems:
                acc = elems[-1]
                for e in reversed(elems[:-1]):
                    acc = lir.MojoCall(func=node.func, args=[e, acc])
                return acc
        # tuple/pair construction: frontend emits tuple(MirList([...])) -> (a, b)
        if node.func == "tuple" and len(node.args) == 1 and isinstance(node.args[0], mir.MirList):
            return lir.MojoTuple(elements=[self.lower_expr(e) for e in node.args[0].elements])
        # std::vector sized ctor as an expression (e.g. the inner fill of a 2D
        # `vector<vector<int>>(m, vector<int>(n,0))`): `[fill] * size`. The
        # type-aware assign path handles the outer one; this enables nesting.
        if node.func == "__vector_fill__" and args:
            fill = args[1] if len(args) >= 2 else lir.MojoIntLiteral(value=0)
            return lir.MojoBinOp(op="*", left=lir.MojoList(elements=[fill]), right=args[0])
        # std::sort(v.begin(), v.end()) -> Mojo `sort(v)` (in-place, prelude builtin)
        if node.func == "sort" and len(node.args) == 2:
            a0 = node.args[0]
            if isinstance(a0, mir.MirMethodCall) and a0.method == "begin":
                return lir.MojoCall(func="sort", args=[self.lower_expr(a0.receiver)])
        # ObjexxFCL integer-power helpers (pervasive in EnergyPlus): pow_2(x) -> x**2.
        _pn = re.fullmatch(r"pow_(\d+)", node.func)
        if _pn and len(args) == 1:
            return lir.MojoBinOp(op="**", left=args[0],
                                 right=lir.MojoIntLiteral(value=int(_pn.group(1))))
        # ObjexxFCL integer-root helpers: root_4(x)==x^(1/4)==sqrt(sqrt(x)),
        # root_8(x)==x^(1/8)==sqrt(sqrt(sqrt(x))) (Fmath.hh). Lower to nested
        # sqrt rather than `** 0.25` to bit-match ObjexxFCL's implementation.
        _rn = re.fullmatch(r"root_([48])", node.func)
        if _rn and len(args) == 1:
            self._used_math.add("sqrt")
            depth = 2 if _rn.group(1) == "4" else 3
            expr = args[0]
            for _ in range(depth):
                expr = lir.MojoCall(func="sqrt", args=[expr])
            return expr
        # ObjexxFCL/Fortran scalar intrinsics.
        if node.func == "mod" and len(args) == 2:          # mod(a, b) -> a % b
            return lir.MojoBinOp(op="%", left=args[0], right=args[1])
        if node.func == "fmod" and len(args) == 2:
            # C fmod(a, b) = a - b*trunc(a/b) (result takes sign of a). Mojo has
            # no math.fmod and its `%` takes the divisor's sign, so build the
            # truncation form explicitly. (Args are pure scalars in this domain,
            # so referencing them twice is side-effect-free.)
            self._used_math.add("trunc")
            quotient = lir.MojoBinOp(op="/", left=args[0], right=args[1])
            truncq = lir.MojoCall(func="trunc", args=[quotient])
            return lir.MojoBinOp(op="-", left=args[0],
                                 right=lir.MojoBinOp(op="*", left=args[1], right=truncq))
        if node.func == "sign" and len(args) == 2:         # Fortran SIGN(a,b) == copysign
            self._used_math.add("copysign")
            return lir.MojoCall(func="copysign", args=args)
        if node.func == "len":
            if len(args) != 1:
                raise ValueError("len() takes exactly one argument")
            return lir.MojoCall(func="len", args=args)
        # SIMD lifting — the C++ frontend translates Intel intrinsics into
        # synthetic `simd_pack` / `simd_splat` / `simd_zero` / `simd_sqrt`
        # calls. Map them to Mojo's portable `SIMD[DType.X, N](...)` form.
        if node.func == "simd_pack" and args:
            return lir.MojoCall(func=f"SIMD[DType.float64, {len(args)}]", args=args)
        if node.func == "simd_splat" and len(args) == 1:
            return lir.MojoCall(func="SIMD[DType.float64, 4]", args=args)
        if node.func == "simd_zero" and not args:
            return lir.MojoCall(
                func="SIMD[DType.float64, 4]",
                args=[lir.MojoFloatLiteral(value=0.0)],
            )
        if node.func == "simd_sqrt" and len(args) == 1:
            return lir.MojoMethodCall(receiver=args[0], method="sqrt", args=[])
        if node.func == "simd_abs" and len(args) == 1:
            return lir.MojoCall(func="abs", args=args)
        # std::clamp(x, lo, hi) -> Mojo `x.clamp(lo, hi)` (method on SIMD/Float).
        if node.func == "clamp" and len(args) == 3:
            return lir.MojoMethodCall(receiver=args[0], method="clamp", args=args[1:])
        # <cmath> intrinsics -> Mojo math fns via `from math import <name>`.
        if node.func in _MATH_FNS:
            self._used_math.add(node.func)
            return lir.MojoCall(func=node.func, args=args)
        if node.func in _BUILTIN_MAP:
            return lir.MojoCall(func=_BUILTIN_MAP[node.func], args=args)
        # Print/abs/min/max identity (already Mojo builtins).
        return lir.MojoCall(func=node.func, args=args)


_LOWERING = _MojoLowering()


def mir_to_mojo_lir(module: mir.MirModule) -> lir.MojoModule:
    return _LOWERING.lower_module(module)


_BUILTIN_MUTATING_METHODS = frozenset({
    "append", "push_back", "emplace_back", "clear", "pop_back", "resize",
})


def _touches_mir_self(node: mir.MirNode) -> bool:
    """A receiver/target rooted at `self` (self, self.field, self.field[i])."""
    if isinstance(node, mir.MirName) and node.name == "self":
        return True
    if isinstance(node, mir.MirFieldAccess):
        return _touches_mir_self(node.value)
    if isinstance(node, mir.MirSubscript):
        return _touches_mir_self(node.value)
    return False


def _mir_mutates_self(nodes, mutating_names: frozenset[str]) -> bool:
    """MIR counterpart of the Mojo backend's own `_mutates_self` (see
    backends/mojo/emit.py) -- needed one IR tier earlier, during MIR->LIR
    lowering, so a *parameter* mutated via a call to one of these methods
    (see _params_reassigned below) can be recognized too. Duplicated rather
    than shared because the two IR tiers use distinct node types
    (MirFieldAssign vs. MojoFieldAssign, etc.) with no common walk."""
    import dataclasses
    if isinstance(nodes, list):
        return any(_mir_mutates_self(n, mutating_names) for n in nodes)
    if not dataclasses.is_dataclass(nodes):
        return False
    if isinstance(nodes, mir.MirFieldAssign) and _touches_mir_self(nodes.obj):
        return True
    if isinstance(nodes, mir.MirSubscriptAssign) and _touches_mir_self(nodes.obj):
        return True
    if isinstance(nodes, mir.MirAssign) and nodes.target == "self":
        # `self = expr` -- the "replace my whole value" idiom for a
        # mutate-via-copy-assign method (C++'s `(*this) = Multiplied(...);`,
        # see `_is_this_deref` in the C++ frontend). Mirrors the same check
        # in emit.py's `_mutates_self`.
        return True
    if (isinstance(nodes, mir.MirMethodCall) and nodes.method in mutating_names
            and _touches_mir_self(nodes.receiver)):
        return True
    return any(
        _mir_mutates_self(getattr(nodes, f.name), mutating_names) for f in dataclasses.fields(nodes)
    )


def _compute_mir_mutating_method_names(module: mir.MirModule) -> frozenset[str]:
    """Fixed-point closure of method names that mutate their receiver,
    computed on MIR (before lowering) so `_params_reassigned` below can
    recognize a reference/by-value parameter mutated via a call to one of
    these -- not just self (see backends/mojo/emit.py's
    `_compute_mutating_method_names`, the LIR-level sibling of this
    function, computed too late in the pipeline for `lower_params` to use).
    Matches by bare method name, not per-struct: conservative, but at worst
    marks an unrelated same-named method/param `mut`/`var` too, which Mojo
    permits on one that doesn't strictly need it.
    """
    names = set(_BUILTIN_MUTATING_METHODS)
    methods = [m for s in module.structs for m in s.methods]
    changed = True
    while changed:
        changed = False
        for m in methods:
            if m.name in names:
                continue
            if _mir_mutates_self(m.body, frozenset(names)):
                names.add(m.name)
                changed = True
    return frozenset(names)


def _params_reassigned(
    body: list[mir.MirNode], param_names: set[str], mutating_names: frozenset[str] = _BUILTIN_MUTATING_METHODS
) -> tuple[set[str], set[str]]:
    """Return two sets: (var_params, mut_params).

    var_params: params that are scalar-reassigned (need `var` prefix).
    mut_params: params that are subscript-assigned (need `mut` prefix).
    """
    # `self` has its own separate, correct handling: `out self` for
    # __init__ (emit.py), `mut self` when the body mutates self (also
    # emit.py's _mutates_self). It must never be a candidate here --
    # before mutating_names included user-defined methods, a body calling
    # a *builtin*-named mutating method directly on `self` (rare) was the
    # only way to trigger this; now that it includes every user method
    # that mutates self too, an __init__ body that calls a self-mutating
    # setter (`self.SetXYZ(...)`) does it constantly. Treating `self` as
    # an ordinary var/mut-param candidate then baked a stray "var "/"mut "
    # prefix straight into the *parameter name string* here, which
    # _emit_fn's own `n == "self"` special-case check (by then comparing
    # against "var self", not "self") no longer matched -- producing the
    # nonsensical, real-compiler-rejected `def __init__(var self: T, ...)`.
    param_names = param_names - {"self"}
    var_out: set[str] = set()
    mut_out: set[str] = set()

    def _param_of(node):
        return node.name if isinstance(node, mir.MirName) and node.name in param_names else None

    def _scan(nodes: list[mir.MirNode]) -> None:
        for n in nodes:
            if isinstance(n, mir.MirAssign) and n.target in param_names:
                var_out.add(n.target)
            elif (
                isinstance(n, mir.MirSubscriptAssign)
                and isinstance(n.obj, mir.MirName)
                and n.obj.name in param_names
            ):
                mut_out.add(n.obj.name)
            # In-place mutation via call: `v.append(x)` (an STL builtin) or
            # `theCoord.Add(x)` (a user-defined method whose own body
            # mutates self, per the closure above) on a param -> needs an
            # owned `var` to allow the call at all. This is a compile-only
            # approximation, same as the plain-reassignment case just
            # above: Mojo's `var` gives a *local* mutable copy regardless
            # of whether the original C++ parameter was pass-by-value or
            # pass-by-reference, so a reference out-parameter's mutation
            # won't actually propagate back to the caller in the emitted
            # target -- accepted here since fully modeling C++ reference
            # parameters isn't; the alternative is refusing to compile at
            # all rather than compiling with a narrower behavioral gap.
            elif isinstance(n, mir.MirMethodCall) and n.method in mutating_names and _param_of(n.receiver):
                var_out.add(_param_of(n.receiver))
            elif (isinstance(n, mir.MirCall) and n.func == "sort" and n.args
                  and isinstance(n.args[0], mir.MirMethodCall)
                  and _param_of(n.args[0].receiver)):
                var_out.add(_param_of(n.args[0].receiver))
            elif isinstance(n, mir.MirIf):
                _scan(n.body)
                _scan(n.orelse)
            elif isinstance(n, mir.MirWhile):
                _scan(n.body)
            elif isinstance(n, mir.MirForRange):
                _scan(n.body)

    _scan(body)
    return var_out, mut_out


class _MojoIfExpr(lir.LirNode):
    """`<then> if <test> else <else>` — Python/Mojo ternary syntax."""

    def __init__(self, test: lir.LirNode, then_: lir.LirNode, else_: lir.LirNode) -> None:
        self.test = test
        self.then_ = then_
        self.else_ = else_


def _mojo_type(ty: Type) -> str:
    if isinstance(ty, IntT):
        # Mojo's `Int` is the idiomatic default — platform-sized but our
        # IR uses 64-bit, which matches on 64-bit targets. Use `Int` for
        # readability; `Int64` is also accepted.
        if ty.bits == 64 and ty.signed:
            return "Int"
        return f"{'Int' if ty.signed else 'UInt'}{ty.bits}"
    if isinstance(ty, FloatT):
        return f"Float{ty.bits}"
    if isinstance(ty, BoolT):
        return "Bool"
    if isinstance(ty, StrT):
        return "String"
    if isinstance(ty, NoneT):
        return "None"
    if isinstance(ty, ListT):
        return f"List[{_mojo_type(ty.elem)}]"
    if isinstance(ty, DictT):
        return f"Dict[{_mojo_type(ty.key)}, {_mojo_type(ty.value)}]"
    if isinstance(ty, TupleT):
        return "Tuple[" + ", ".join(_mojo_type(e) for e in ty.elems) + "]"
    if isinstance(ty, StructT):
        return ty.name
    if isinstance(ty, SimdT):
        # `SIMD[DType.<elem>, <lanes>]` — Mojo's portable SIMD type. The
        # compiler picks AVX2/AVX-512/NEON/SVE per target ISA.
        return f"SIMD[DType.{_dtype_for(ty.elem)}, {ty.lanes}]"
    if isinstance(ty, UnknownT):
        raise ValueError(f"unresolved type hole: {ty.hint}")
    raise NotImplementedError(f"type {type(ty).__name__}")


def _dtype_for(elem: Type) -> str:
    if isinstance(elem, FloatT):
        return f"float{elem.bits}"
    if isinstance(elem, IntT):
        prefix = "int" if elem.signed else "uint"
        return f"{prefix}{elem.bits}"
    if isinstance(elem, BoolT):
        return "bool"
    raise NotImplementedError(f"no DType for {type(elem).__name__}")
