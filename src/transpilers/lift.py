"""Never-refuse C++ -> Python "lift" (Phase-1 whole-program migration).

Unlike the strict HIR transpiler (which REFUSES unsupported constructs), this
lifter translates a whole file/snippet 1:1 — architecture preserved — and emits
a `# TODO[lift]` stub for anything it can't yet handle, so it ALWAYS produces
output and scales to a whole repo. Coverage grows by adding node handlers.

Lift conventions:
  * struct/class            -> Python class; fields -> __init__ defaults
  * `a->b` / `a.b`          -> `a.b`  (unique_ptr / ref / value collapse)
  * member chains           -> recovered from TOKENS when the AST is name-degraded
                               (so state.dataX->field works even without a full
                               type-resolving parse)
  * C-style for(i;c;inc)    -> init; while c: body; inc
  * (T)x / static_cast/T(x) -> value passed through
  * x.size()                -> len(x)
  * unknown node            -> `# TODO[lift]` + raw snippet
"""

from __future__ import annotations

import keyword
import os
import re
import clang.cindex as ci

from transpilers.frontends.cpp import parser as _cfg  # noqa: F401  (configures libclang)

K = ci.CursorKind
IND = "    "


_SOFT_KW = {"match", "case", "type"}  # contextual keywords / common shadows


_OP_DUNDER = {
    "operator==": "__eq__",
    "operator!=": "__ne__",
    "operator<": "__lt__",
    "operator<=": "__le__",
    "operator>": "__gt__",
    "operator>=": "__ge__",
    "operator()": "__call__",
    "operator[]": "__getitem__",
    "operator+": "__add__",
    "operator-": "__sub__",
    "operator*": "__mul__",
    "operator/": "__truediv__",
    "operator%": "__mod__",
    "operator+=": "__iadd__",
    "operator-=": "__isub__",
    "operator*=": "__imul__",
    "operator bool": "__bool__",
    "operator<<": "__lshift__",
}


def _py_method_name(spelling: str) -> str:
    """C++ method name -> Python identifier, mapping operator overloads to the
    matching dunder (`operator==` -> `__eq__`) so the emitted `def` is valid."""
    if spelling in _OP_DUNDER:
        return _OP_DUNDER[spelling]
    if spelling.startswith("operator"):  # unmapped operator -> safe name
        return "_op_" + re.sub(r"\W+", "_", spelling[len("operator") :]).strip("_")
    return snake(spelling)


def snake(name: str) -> str:
    name = name.split("::")[-1]
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    s = s.lower() if s else name
    # C++ identifiers like `is`, `in`, `class`, `lambda` are Python keywords —
    # suffix `_` so the emitted name/attribute is valid Python.
    if keyword.iskeyword(s) or s in _SOFT_KW:
        s += "_"
    return s


def toks(c) -> list[str]:
    return [t.spelling for t in c.get_tokens()]


def src(c) -> str:
    try:
        e = c.extent
        raw = open(e.start.file.name, "rb").read()[e.start.offset : e.end.offset]
        return raw.decode("utf-8", "replace")
    except Exception:
        return c.spelling or "?"


# tokens we can faithfully re-emit from a degraded AST node (no type info)
_TOK_OK = re.compile(
    r"[A-Za-z_]\w*"  # identifiers
    r"|->|\.|::|\(|\)|\[|\]|,"  # access / call / subscript
    r"|&&|\|\||!"  # logical
    r"|==|!=|<=|>=|<|>"  # comparison
    r"|\+|-|\*|/|%"  # arithmetic
    r"|[0-9][0-9.eExXa-fA-F+]*"  # numeric literal
    r'|".*"|\'.*\''  # string / char literal
)
# binary operators that want surrounding whitespace in the emitted Python
_SPACED = {"==", "!=", "<=", ">=", "<", ">", "+", "-", "*", "/", "%"}
_LOGIC = {"&&": " and ", "||": " or "}
# operators `e()` may legitimately emit for a BINARY_OPERATOR; anything else
# means a degraded/macro-mangled node — fall back rather than emit garbage.
_PY_BINOPS = {
    "+",
    "-",
    "*",
    "/",
    "%",
    "//",
    "**",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "and",
    "or",
    "&",
    "|",
    "^",
    "<<",
    ">>",
}


# Python binary-operator precedence (higher binds tighter). Emission decisions
# use PYTHON's table (not C++'s): clang already resolved the C++ parse into the
# AST shape; what matters is how the emitted text re-parses in Python.
_PY_PREC = {
    "or": 1,
    "and": 2,
    "==": 4,
    "!=": 4,
    "<": 4,
    "<=": 4,
    ">": 4,
    ">=": 4,
    "in": 4,
    "is": 4,
    "|": 5,
    "^": 6,
    "&": 7,
    "<<": 8,
    ">>": 8,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "//": 10,
    "%": 10,
    "**": 12,
}
_CMP_PREC = 4  # Python CHAINS comparisons; C++ nests them left-to-right
_UNARY_MINUS_PREC = 11  # binds tighter than every binary operator except **


def _min_top_prec(s: str):
    """Lowest precedence of any binary operator at paren/bracket/quote depth 0
    of an emitted Python expression (the lifter emits binary operators
    space-delimited), or None if the expression is atomic at the top level."""
    best = None
    depth, quote, i, n = 0, "", 0, len(s)
    while i < n:
        ch = s[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and ch == " ":
            j = s.find(" ", i + 1)
            p = _PY_PREC.get(s[i + 1 : j]) if j != -1 else None
            if p is not None and (best is None or p < best):
                best = p
        i += 1
    if s.startswith("not ") and (best is None or best > 3):
        best = 3
    if s.startswith("lambda "):
        best = 0
    return best


def _paren_operand(s: str, parent_op: str, right: bool = False) -> str:
    """Re-parenthesize an emitted operand whose top-level operator binds looser
    than `parent_op` — restores the grouping dropped when PAREN_EXPR nodes are
    unwrapped (#58: `(Tamb - Tsurf) * cosTilt` must not become
    `tamb - tsurf * cos_tilt`).

    Equal precedence: a LEFT operand reproduces C++'s left-associative parse
    unparenthesized, but a RIGHT operand at equal precedence only exists
    because the source had explicit parens (`a - (b - c)`, `a / (b * c)`), so
    it keeps them; and comparisons chain in Python where C++ nests, so a
    comparison child of a comparison parent is always wrapped."""
    pp = _PY_PREC.get(parent_op)
    if pp is None:
        return s
    cp = _min_top_prec(s)
    if cp is None:
        return s
    if cp < pp or (cp == pp and (right or pp == _CMP_PREC)):
        return f"({s})"
    return s


_CAST_NAMES = {"static_cast", "dynamic_cast", "reinterpret_cast", "const_cast"}
_PRIM_TYPES = {
    "int",
    "long",
    "short",
    "char",
    "double",
    "float",
    "bool",
    "unsigned",
    "signed",
    "size_t",
    "void",
    "const",
    "Real64",
    "Real32",
    "Int",
    "Int64",
    "int64_t",
    "int32_t",
    "intptr_t",
    "uintptr_t",
    "ptrdiff_t",
    "wchar_t",
    "uint8_t",
    "uint32_t",
    "uint64_t",
    "int8_t",
    "int16_t",
    "uint16_t",
}
_ASSIGN_OPS = {"=", "+=", "-=", "*=", "/=", "%="}


def _strip_casts(ts: list[str]) -> list[str]:
    """Drop cast wrappers from a token stream so they don't re-emit as Python
    comparisons/garbage: `dynamic_cast < T * > ( e )` -> `( e )` and the C-style
    `( int ) x` -> `x` (only when the parens wrap primitive type tokens)."""
    out, i, n = [], 0, len(ts)
    while i < n:
        if ts[i] in _CAST_NAMES and i + 1 < n and ts[i + 1] == "<":
            depth, j = 0, i + 1
            while j < n:
                if ts[j] == "<":
                    depth += 1
                elif ts[j] == ">":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            i = j + 1  # skip past the closing '>' ; the following (expr) stays
            continue
        # C-style cast `( <primitive-types> ) expr` -> `expr`
        if ts[i] == "(":
            j = i + 1
            while j < n and ts[j] in _PRIM_TYPES or (j < n and ts[j] == "*"):
                j += 1
            if (
                j > i + 1
                and j < n
                and ts[j] == ")"
                and all(t in _PRIM_TYPES or t == "*" for t in ts[i + 1 : j])
            ):
                i = j + 1  # drop `( types )`, keep the cast operand
                continue
        out.append(ts[i])
        i += 1
    return out


def _num(t: str) -> str:
    """Strip C numeric suffixes, but NOT from hex literals — `F`/`f` are hex
    digits there, so `0xFF`.rstrip('uUlLfF') would wrongly become `0x`."""
    if t[:2].lower() == "0x":
        return t.rstrip("uUlL")  # only u/U/l/L are hex suffixes
    return t.rstrip("uUlLfF")


def _balanced(ts: list[str]) -> bool:
    """True iff `()` and `[]` are balanced and never close before opening.

    Degraded macro expansions (e.g. ObjexxFCL `EP_SIZE_CHECK(...)`) recover into
    token streams like `nsides ) 0` — every token is individually translatable
    but the result is unbalanced garbage. Reject those so the lifter emits a
    valid TODO instead of unparseable Python."""
    p = b = 0
    for t in ts:
        if t == "(":
            p += 1
        elif t == ")":
            p -= 1
            if p < 0:
                return False
        elif t == "[":
            b += 1
        elif t == "]":
            b -= 1
            if b < 0:
                return False
    return p == 0 and b == 0


def _valid_target(s: str) -> bool:
    """True iff `s` is a legal Python assignment target (name / attribute /
    subscript / tuple). Degraded ObjexxFCL LHS recovery can yield `k + 1` or
    `1` (array name lost) — reject those so we don't emit `k + 1 = v`."""
    if not s:
        return False
    try:
        compile(f"{s} = 0", "<lhs>", "exec")
        return True
    except SyntaxError, ValueError:
        return False


def _to_lvalue(s: str) -> str:
    """Rewrite a trailing call `…name(idx)` -> subscript `…name[idx]` so a
    token-recovered expression is valid as an assignment target (ObjexxFCL
    1-based `()` indexing). Shared by the AST and token paths."""
    if not s.endswith(")"):
        return s
    depth = 0
    for i in range(len(s) - 1, -1, -1):
        if s[i] == ")":
            depth += 1
        elif s[i] == "(":
            depth -= 1
            if depth == 0:
                inner = s[i + 1 : -1]
                return s[:i] + "[" + inner + "]" if inner else s
    return s


def _toks_to_py(ts: list[str]) -> str:
    ts = _strip_casts(ts)
    # Top-level assignment in a token-recovered statement: the LHS may be a
    # call-style ObjexxFCL element (`arr(i) += 1`) — fix it to a subscript.
    depth = 0
    for idx, t in enumerate(ts):
        if t in "([":
            depth += 1
        elif t in ")]":
            depth -= 1
        elif depth == 0 and t in _ASSIGN_OPS:
            lhs = _to_lvalue(_toks_to_py(ts[:idx]))
            rhs = _toks_to_py(ts[idx + 1 :])
            return f"{lhs} {t} {rhs}"
    """Best-effort: turn a degraded operand token stream into Python.

    Handles identifier/member chains (`state.dataX->Field`), ObjexxFCL
    array-member access (`Arr(i).Field` -> `arr(i).field`, the 1-based
    subscript is kept faithfully as a call for Phase-1), namespaced calls
    (`Util::SameString(a,b)` -> `same_string(a, b)`, `std::abs(x)` ->
    `abs(x)`; the namespace qualifier is dropped, only the final identifier
    is kept + snaked), and embedded operators/literals.
    """
    out = []
    i, n = 0, len(ts)
    while i < n:
        t = ts[i]
        nxt = ts[i + 1] if i + 1 < n else None
        if t in ("->", "."):
            out.append(".")
        elif t == "::":
            # namespace qualifier: drop it, the previous identifier too
            if out and re.fullmatch(r"[A-Za-z_]\w*", out[-1]):
                out.pop()
        elif t == "this":
            out.append("self")
        elif t in ("true", "false"):
            out.append("True" if t == "true" else "False")
        elif t in ("nullptr", "NULL"):
            out.append("None")
        elif t == "!":
            out.append("not ")
        elif t in _LOGIC:
            out.append(_LOGIC[t])
        elif t == ",":
            out.append(", ")
        elif t in ("*", "&") and (
            not out
            or out[-1].strip()
            in (
                "",
                "(",
                "[",
                ",",
                "+",
                "-",
                "*",
                "/",
                "%",
                "==",
                "!=",
                "<",
                "<=",
                ">",
                ">=",
                "and",
                "or",
                "not",
                "return",
                "=",
            )
        ):
            pass  # prefix `*`/`&` (deref / address-of) — no-op once pointers
            # collapse to values; drop so `(*ptr).f` -> `(ptr).f`
        elif t in _SPACED:
            out.append(f" {t} ")
        elif re.fullmatch(r"[A-Za-z_]\w*", t):
            # snake unless it's a namespace qualifier (next token is `::`)
            out.append(t if nxt == "::" else snake(t))
        elif t and (t[0].isdigit()):
            out.append(_num(t))
        else:
            out.append(t)
        i += 1
    return "".join(out)


class _Lifter:
    def __init__(self):
        self.nodes = 0
        self.todo = 0

    def todo_expr(self, c, why="") -> str:
        self.todo += 1
        # Expression position: a `#` comment would swallow the rest of an
        # enclosing line (e.g. other call args), producing invalid Python. Emit
        # a sentinel CALL that preserves the snippet and stays syntactically
        # valid. `__todo__` is defined in the lifted module preamble.
        snip = " ".join(src(c).split())[:80].replace("\\", "\\\\").replace('"', '\\"')
        tag = f"{c.kind.name}{(' ' + why) if why else ''}: {snip}"
        return f'__todo__("{tag}")'

    # ----------------------------- expressions ----------------------------- #
    def e(self, c) -> str:
        self.nodes += 1
        k = c.kind
        if k in (
            K.UNEXPOSED_EXPR,
            K.PAREN_EXPR,
            K.CXX_STATIC_CAST_EXPR,
            K.CSTYLE_CAST_EXPR,
            K.CXX_FUNCTIONAL_CAST_EXPR,
            K.CXX_DYNAMIC_CAST_EXPR,
            K.CXX_CONST_CAST_EXPR,
            K.CXX_REINTERPRET_CAST_EXPR,
        ):
            kids = [x for x in c.get_children() if x.kind != K.TYPE_REF]
            return self.e(kids[-1]) if kids else self._tok_fallback(c)
        if k in (K.INTEGER_LITERAL, K.FLOATING_LITERAL):
            t = toks(c)
            return _num(t[0]) if t else "0"
        if k == K.CXX_BOOL_LITERAL_EXPR:
            return "True" if (toks(c) or ["false"])[0] == "true" else "False"
        if k == K.STRING_LITERAL:
            return c.spelling
        if k in (K.GNU_NULL_EXPR, K.CXX_NULL_PTR_LITERAL_EXPR):
            return "None"
        if k == K.CXX_THIS_EXPR:
            return "self"
        if k == K.DECL_REF_EXPR:
            return snake(c.spelling) if c.spelling else self._tok_fallback(c)
        if k == K.MEMBER_REF_EXPR:
            field = c.spelling
            kids = list(c.get_children())
            if field and kids:
                return f"{self.e(kids[0])}.{snake(field)}"
            # name-degraded chain: reconstruct from tokens (recovers state.dataX->f)
            return self._tok_fallback(c)
        if k == K.ARRAY_SUBSCRIPT_EXPR:
            a, i = list(c.get_children())
            return f"{self.e(a)}[{self.e(i)}]"
        if k == K.INIT_LIST_EXPR:
            # C++ brace-init `{a, b, c}` -> Python list (faithful Phase-1 form;
            # works for aggregate/vector/array init and {key, value} pairs).
            return "[" + ", ".join(self.e(x) for x in c.get_children()) + "]"
        if k in (K.BINARY_OPERATOR, K.COMPOUND_ASSIGNMENT_OPERATOR):
            kids = list(c.get_children())
            if len(kids) == 2:
                op = self._binop(c)
                if op in _PY_BINOPS:
                    lhs = _paren_operand(self.e(kids[0]), op)
                    rhs = _paren_operand(self.e(kids[1]), op, right=True)
                    return f"{lhs} {op} {rhs}"
                # degraded operator (e.g. macro expansion gives `)`): don't emit
                # garbage like `nsides ) 0` — recover from tokens or TODO.
                return self._tok_fallback(c)
        if k == K.UNARY_OPERATOR:
            kids = list(c.get_children())
            op = (toks(c) or ["?"])[0]
            inner = self.e(kids[0]) if kids else "None"
            if op == "!":
                return f"(not {inner})"
            if op == "-":
                # `-(a + b)` must not emit `(-a + b)`: unary minus binds
                # tighter than the unwrapped binary, so re-wrap the operand.
                cp = _min_top_prec(inner)
                if cp is not None and cp < _UNARY_MINUS_PREC:
                    inner = f"({inner})"
                return f"(-{inner})"
            return inner
        if k == K.CONDITIONAL_OPERATOR:
            kids = list(c.get_children())
            if len(kids) == 3:
                return (
                    f"({self.e(kids[1])} if {self.e(kids[0])} else {self.e(kids[2])})"
                )
        if k == K.CALL_EXPR:
            return self._call(c)
        if k == K.LAMBDA_EXPR:
            return self._lambda(c)
        return self._tok_fallback(c)

    def _lambda(self, c) -> str:
        """C++ lambda -> Python `lambda`. Faithful only for the single-`return`
        body shape (pervasive in EnergyPlus root-finder callbacks, e.g.
        `[&](Real64 x){ return f(x); }`); multi-statement bodies degrade to a
        TODO so we never emit invalid syntax."""
        params = [
            snake(a.spelling) or "arg"
            for a in c.get_children()
            if a.kind == K.PARM_DECL
        ]
        body = next((x for x in c.get_children() if x.kind == K.COMPOUND_STMT), None)
        stmts = [x for x in body.get_children()] if body is not None else []
        if len(stmts) == 1 and stmts[0].kind == K.RETURN_STMT:
            rk = list(stmts[0].get_children())
            expr = self.e(rk[0]) if rk else "None"
            return f"lambda {', '.join(params)}: {expr}"
        return self.todo_expr(c)

    def _tok_fallback(self, c) -> str:
        ts = toks(c)
        # only re-emit when EVERY token is one we know how to translate, and
        # the expression isn't an assignment (handled structurally elsewhere)
        if (
            ts
            and "=" not in ts
            and _balanced(ts)
            and all(_TOK_OK.fullmatch(t) for t in ts)
        ):
            return _toks_to_py(ts)
        return self.todo_expr(c)

    def _binop(self, c) -> str:
        kids = list(c.get_children())
        if len(kids) == 2:
            n = len(list(kids[0].get_tokens()))
            tk = toks(c)
            if n < len(tk):
                return {"&&": "and", "||": "or"}.get(tk[n], tk[n])
        return "?"

    def _call(self, c) -> str:
        argl = list(c.get_children())
        ref = c.referenced
        opname = (ref.spelling if ref else "") or c.spelling or ""
        if opname.startswith("operator"):
            op = opname[len("operator") :]
            recv = argl[0] if argl else None  # receiver is the object
            rest = argl[1:]
            if op in ("->", "*") and rest:
                return self.e(rest[-1])
            if op == "[]" and recv is not None and rest:
                base = self.e(recv)
                if base:
                    return f"{base}[{self.e(rest[0])}]"
                return self._tok_fallback(c)
            if op == "()" and recv is not None:
                # ObjexxFCL `arr(i)` element access. When the array base emits
                # empty (name-degraded recv), recovering only the index yields
                # garbage like `(k+1)` -> `k+1` as an LHS. Recover the whole call
                # from tokens (its extent still covers the array name).
                base = self.e(recv)
                if base:
                    return f"{base}({', '.join(self.e(a) for a in rest)})"
                return self._tok_fallback(c)
            if (
                op in ("+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=")
                and len(rest) == 2
            ):
                lhs = _paren_operand(self.e(rest[0]), op)
                rhs = _paren_operand(self.e(rest[1]), op, right=True)
                return f"{lhs} {op} {rhs}"
            return self.e(rest[-1]) if rest else "None"
        if argl and argl[0].kind == K.MEMBER_REF_EXPR:
            recv = argl[0]
            meth = snake(recv.spelling)
            rest = [self.e(a) for a in argl[1:]]
            if (
                not meth
            ):  # name-degraded method -> recover the whole receiver chain from tokens
                return f"{self._tok_fallback(recv)}({', '.join(rest)})"
            rk = list(recv.get_children())
            base = self.e(rk[0]) if rk else "self"
            if meth == "size" and not rest:
                return f"len({base})"
            return f"{base}.{meth}({', '.join(rest)})"
        # free call (or call through a name-degraded callee -> recover from tokens)
        if not opname and argl:
            return f"{self._tok_fallback(argl[0])}({', '.join(self.e(a) for a in argl[1:])})"
        name = snake(opname) if opname else "_call"
        return f"{name}({', '.join(self.e(a) for a in argl[1:])})"

    # ----------------------------- statements ------------------------------ #
    def stmt(self, c, ind) -> list[str]:
        self.nodes += 1
        k = c.kind
        p = ind * IND
        if k == K.COMPOUND_STMT:
            out = []
            for ch in c.get_children():
                out += self.stmt(ch, ind)
            return out or [p + "pass"]
        if k == K.DECL_STMT:
            out = []
            for v in c.get_children():
                if v.kind == K.VAR_DECL:
                    init = [
                        x
                        for x in v.get_children()
                        if x.kind not in (K.TYPE_REF, K.TEMPLATE_REF, K.NAMESPACE_REF)
                    ]
                    out.append(
                        f"{p}{snake(v.spelling)} = {self.e(init[-1]) if init else 'None'}"
                    )
            return out or [p + "pass"]
        if k == K.RETURN_STMT:
            kids = list(c.get_children())
            if not kids:
                return [p + "return"]
            # `return (x = y);` -> hoist the assignment, return the value.
            hoist, val = self._hoist_cond(kids[0], ind)
            return hoist + [p + "return " + val]
        if k == K.IF_STMT:
            kids = list(c.get_children())
            pre = []
            # C++17 if-with-initializer: `if (auto x = f(); cond)` -> hoist the
            # init DECL_STMT above the `if` (else it'd be read as the condition).
            while kids and kids[0].kind == K.DECL_STMT:
                pre += self.stmt(kids[0], ind)
                kids = kids[1:]
            # C-idiom `if ((x = f()) == 0)`: hoist the embedded assignment above
            # the `if` (a bare `=` inside a condition is a Python syntax error).
            hoist, cond = self._hoist_cond(kids[0], ind)
            out = pre + hoist + [f"{p}if {cond}:"] + self._body(kids[1], ind + 1)
            if len(kids) > 2:
                out += [f"{p}else:"] + self._body(kids[2], ind + 1)
            return out
        if k == K.WHILE_STMT:
            kids = list(c.get_children())
            hoist, cond = self._hoist_cond(kids[0], ind + 1)
            if hoist:  # assignment in the test -> while True/break form
                out = [f"{p}while True:"] + hoist
                out += [f"{p}{IND}if not ({cond}):", f"{p}{IND}{IND}break"]
                out += self._body(kids[-1], ind + 1)
                return out
            return [f"{p}while {cond}:"] + self._body(kids[-1], ind + 1)
        if k == K.FOR_STMT:
            return self._for(c, ind)
        if k == K.SWITCH_STMT:
            return self._switch(c, ind)
        if k == K.DO_STMT:
            # `do { body } while (cond);` -> `while True: body; if not cond: break`
            kids = list(c.get_children())
            body = next((x for x in kids if x.kind == K.COMPOUND_STMT), None)
            cond = next((x for x in kids if x.kind != K.COMPOUND_STMT), None)
            out = [f"{p}while True:"]
            out += self.stmt(body, ind + 1) if body is not None else [p + IND + "pass"]
            if cond is not None:
                out += [f"{p}{IND}if not ({self.e(cond)}):", f"{p}{IND}{IND}break"]
            return out
        if k == K.BREAK_STMT:
            return [p + "break"]
        if k == K.CONTINUE_STMT:
            return [p + "continue"]
        if k == K.NULL_STMT:
            return []
        if k in (
            K.CALL_EXPR,
            K.BINARY_OPERATOR,
            K.COMPOUND_ASSIGNMENT_OPERATOR,
            K.UNARY_OPERATOR,
            K.UNEXPOSED_EXPR,
            K.MEMBER_REF_EXPR,
            K.PAREN_EXPR,
        ):
            # C++ comma operator `a, b, c;` -> one Python statement per operand.
            parts = self._comma_split(c)
            return [p + self._exprstmt(x) for x in parts]
        self.todo += 1
        return [f"{p}pass  # TODO[lift]: {k.name} :: {' '.join(src(c).split())[:70]}"]

    def _body(self, c, ind):
        return self.stmt(c, ind) if c.kind == K.COMPOUND_STMT else self.stmt(c, ind)

    def _for(self, c, ind):
        p = ind * IND
        kids = list(c.get_children())
        body = kids[-1] if kids and kids[-1].kind == K.COMPOUND_STMT else None
        clauses = kids[:-1] if body is not None else kids
        if len(clauses) != 3:  # only the canonical for(init;cond;inc) form
            self.todo += 1
            return [
                f"{p}# TODO[lift]: {c.kind.name} :: {' '.join(src(c).split())[:70]}",
                p + "pass",
            ]
        init, cond, inc = clauses
        out = (
            self.stmt(init, ind)
            if init.kind == K.DECL_STMT
            else [p + self._exprstmt(init)]
        )
        out.append(f"{p}while {self.e(cond)}:")
        out += self.stmt(body, ind + 1) if body is not None else [p + IND + "pass"]
        out.append(p + IND + self._exprstmt(inc))
        return out

    def _comma_split(self, c) -> list:
        """Flatten a top-level C++ comma-operator expression into its operands.
        Non-comma expressions return `[c]` unchanged."""
        if c.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR):
            kids = [x for x in c.get_children() if x.kind != K.TYPE_REF]
            if len(kids) == 1:
                return self._comma_split(kids[0])
        if c.kind == K.BINARY_OPERATOR:
            kids = list(c.get_children())
            if len(kids) == 2 and self._binop(c) == ",":
                return self._comma_split(kids[0]) + self._comma_split(kids[1])
        return [c]

    def _hoist_cond(self, c, ind):
        """Split assignment-expressions out of a condition.

        C uses `if ((x = f()) == 0)`; Python has no assignment expression with an
        attribute/subscript target, so emit `x = f()` as a pre-statement and use
        `x` in the condition. Returns (pre_lines, cond_str)."""
        p = ind * IND
        pre: list[str] = []

        def rewrite(cur) -> str:
            if cur.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR):
                kids = [x for x in cur.get_children() if x.kind != K.TYPE_REF]
                return rewrite(kids[-1]) if kids else self.e(cur)
            if cur.kind == K.BINARY_OPERATOR:
                kids = list(cur.get_children())
                if len(kids) == 2:
                    op = self._binop(cur)
                    if op == "=":
                        lhs = self._lvalue(self.e(kids[0]))
                        pre.append(f"{p}{lhs} = {self.e(kids[1])}")
                        return lhs
                    if op in ("and", "or", "==", "!=", "<", "<=", ">", ">="):
                        lhs = _paren_operand(rewrite(kids[0]), op)
                        rhs = _paren_operand(rewrite(kids[1]), op, right=True)
                        return f"{lhs} {op} {rhs}"
            return self.e(cur)

        cond = rewrite(c)
        return pre, cond

    def _flatten_switch(self, node, items):
        """Linearise clang's nested switch body into ordered items:
        ('case', value_cursor) | ('default', None) | ('stmt', cursor)."""
        k = node.kind
        if k == K.CASE_STMT:
            cc = list(node.get_children())
            items.append(("case", cc[0]))
            if len(cc) > 1:
                self._flatten_switch(cc[1], items)
        elif k == K.DEFAULT_STMT:
            items.append(("default", None))
            cc = list(node.get_children())
            if cc:
                self._flatten_switch(cc[0], items)
        else:
            items.append(("stmt", node))

    def _switch(self, c, ind):
        """`switch(x){case A: ...; break; default: ...}` -> if/elif/else on x.

        Fallthrough is not modelled (Phase-1): each `case`/`default` label run
        becomes its own branch; trailing `break` is dropped. Stacked labels
        (`case A: case B:`) collapse into `x == A or x == B`."""
        p = ind * IND
        kids = list(c.get_children())
        body = next((x for x in reversed(kids) if x.kind == K.COMPOUND_STMT), None)
        # C++17 switch-with-initializer: hoist any leading init DECL_STMT(s).
        pre, rest = [], [x for x in kids if x is not body]
        while rest and rest[0].kind == K.DECL_STMT:
            pre += self.stmt(rest[0], ind)
            rest = rest[1:]
        cond = rest[0] if rest else None
        if body is None or cond is None:
            self.todo += 1
            return [
                f"{p}pass  # TODO[lift]: SWITCH_STMT :: {' '.join(src(c).split())[:60]}"
            ]
        subj = _paren_operand(self.e(cond), "==")  # subject feeds `subj == v` tests
        items = []
        for ch in body.get_children():
            self._flatten_switch(ch, items)
        # group consecutive labels + their following statements into branches
        groups = []  # each: {"default": bool, "values": [...], "stmts": [...]}
        cur = None
        for kind, payload in items:
            if kind in ("case", "default"):
                if cur is None or cur["stmts"]:
                    cur = {"default": False, "values": [], "stmts": []}
                    groups.append(cur)
                if kind == "default":
                    cur["default"] = True
                else:
                    cur["values"].append(payload)
            else:
                if cur is None:
                    cur = {"default": False, "values": [], "stmts": []}
                    groups.append(cur)
                cur["stmts"].append(payload)
        # default branch must be last for valid if/elif/else ordering
        groups.sort(key=lambda g: g["default"])
        out, emitted_if = [], False
        for g in groups:
            stmt_lines = []
            for s in g["stmts"]:
                if s.kind == K.BREAK_STMT:
                    continue
                stmt_lines += self.stmt(s, ind + 1)
            stmt_lines = stmt_lines or [p + IND + "pass"]
            if g["default"] and not g["values"]:
                if emitted_if:
                    out.append(f"{p}else:")
                    out += stmt_lines
                else:  # default-only switch: emit unguarded
                    out += [
                        ln[len(IND) :] if ln.startswith(IND) else ln
                        for ln in stmt_lines
                    ]
            else:
                conds = (
                    " or ".join(f"{subj} == {self.e(v)}" for v in g["values"]) or "True"
                )
                out.append(f"{p}{'if' if not emitted_if else 'elif'} {conds}:")
                out += stmt_lines
                emitted_if = True
        return pre + (out or [p + "pass"])

    @staticmethod
    def _lvalue(s: str) -> str:
        """Make an emitted expression usable as an assignment target.

        ObjexxFCL indexes with `()` (1-based), so the lifter renders element
        access as a call `arr(i)`. As an LHS that's invalid Python (can't assign
        to a call), so rewrite the trailing call `…name(idx)` -> subscript
        `…name[idx]`. A target ending in `.field` is left alone (valid setattr)."""
        if not s.endswith(")"):
            return s
        depth = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] == ")":
                depth += 1
            elif s[i] == "(":
                depth -= 1
                if depth == 0:
                    inner = s[i + 1 : -1]
                    return s[:i] + "[" + inner + "]" if inner else s
        return s

    @staticmethod
    def _assign_op(c):
        """Find an assignment operator token at paren-depth 0 (robust against
        `_binop`'s token-index heuristic, which mis-splits some nodes)."""
        depth = 0
        for t in toks(c):
            if t in ("(", "[", "{"):
                depth += 1
            elif t in (")", "]", "}"):
                depth -= 1
            elif depth == 0 and t in _ASSIGN_OPS:
                return t
        return None

    def _resolve_target(self, node):
        """Resolve a legal Python assignment target for an lvalue node, or None.
        Tries the structured emit, then verbatim token recovery (which keeps a
        degraded ObjexxFCL array name), validating each via `compile`."""
        lhs = self._lvalue(self.e(node))
        if _valid_target(lhs):
            return lhs
        lhs = self._lvalue(self._tok_fallback(node))
        return lhs if _valid_target(lhs) else None

    def _todo_stmt(self, c, why) -> str:
        self.todo += 1
        return f"pass  # TODO[lift]: {why} :: {' '.join(src(c).split())[:55]}"

    def _exprstmt(self, c) -> str:
        if c.kind in (K.BINARY_OPERATOR, K.COMPOUND_ASSIGNMENT_OPERATOR):
            kids = list(c.get_children())
            if len(kids) == 2:
                op = self._assign_op(c)
                if op:
                    rhs = self.e(kids[1])  # eval RHS before target
                    lhs = self._resolve_target(kids[0])
                    return (
                        f"{lhs} {op} {rhs}"
                        if lhs
                        else self._todo_stmt(c, "assign target")
                    )
        if c.kind == K.UNARY_OPERATOR:
            t = toks(c)
            kids = list(c.get_children())
            if "++" in t or "--" in t:
                lhs = self._resolve_target(kids[0])
                aug = "+=" if "++" in t else "-="
                return f"{lhs} {aug} 1" if lhs else self._todo_stmt(c, "incr target")
        return self.e(c)

    # ------------------------------ top level ------------------------------ #
    def function(self, c, ind=0, cls=False):
        p = ind * IND
        params = ["self"] if cls else []
        params += [snake(a.spelling) or "arg" for a in c.get_arguments()]
        body = next((x for x in c.get_children() if x.kind == K.COMPOUND_STMT), None)
        name = _py_method_name(c.spelling)
        if body is None:
            return [f"{p}def {name}({', '.join(params)}): ...  # decl only"]
        return [f"{p}def {name}({', '.join(params)}):"] + self.stmt(body, ind + 1)

    def record(self, c):
        out = [f"class {c.spelling or 'Anon'}:", f"{IND}def __init__(self):"]
        fields = [f for f in c.get_children() if f.kind == K.FIELD_DECL]
        if not fields:
            out.append(f"{IND}{IND}pass")
        for f in fields:
            init = [
                x
                for x in f.get_children()
                if x.kind not in (K.TYPE_REF, K.TEMPLATE_REF, K.NAMESPACE_REF)
            ]
            default = self.e(init[-1]) if init else _default_for(f.type)
            out.append(f"{IND}{IND}self.{snake(f.spelling)} = {default}")
        for m in (
            x for x in c.get_children() if x.kind == K.CXX_METHOD and x.is_definition()
        ):
            out.append("")
            out += [IND + ln for ln in self.function(m, ind=0, cls=True)]
        return out


def _default_for(t) -> str:
    s = t.spelling
    if "bool" in s:
        return "False"
    if any(x in s for x in ("double", "float", "Real64")):
        return "0.0"
    if any(x in s for x in ("string", "char")) and "*" not in s:
        return '""'
    if any(x in s for x in ("vector", "Array1D", "Array2D", "[")):
        return "[]"
    if "map" in s:
        return "{}"
    if (
        any(x in s for x in ("int", "long", "short", "size_t", "Int"))
        and "*" not in s
        and "<" not in s
    ):
        return "0"
    return "None"


# ----------------------------- public API ---------------------------------- #
def _emit(tu, path) -> tuple[str, dict]:
    lf = _Lifter()
    lines = [
        f'"""Lifted 1:1 from {os.path.basename(path)} (Phase-1 C++->Python lift)."""',
        "",
        "def __todo__(_snippet):  # placeholder for not-yet-lifted constructs",
        "    raise NotImplementedError(_snippet)",
        "",
    ]

    def from_file(c):
        f = c.location.file
        if not f:
            return False
        try:
            return os.path.samefile(f.name, path)
        except OSError:  # in-memory unsaved file (lift_source) — name compare
            return f.name == path or os.path.basename(f.name) == os.path.basename(path)

    def walk(node):
        for c in node.get_children():
            if c.kind == K.NAMESPACE:
                walk(c)
                continue
            if not from_file(c):
                continue
            if c.kind in (K.STRUCT_DECL, K.CLASS_DECL) and c.is_definition():
                lines.extend(lf.record(c))
                lines.append("")
            elif c.kind in (K.FUNCTION_DECL, K.CXX_METHOD) and c.is_definition():
                lines.extend(lf.function(c))
                lines.append("")

    walk(tu.cursor)
    return "\n".join(lines) + "\n", {"nodes": lf.nodes, "todo": lf.todo}


def lift_source(source: str, name: str = "input.cpp", inc=None) -> tuple[str, dict]:
    args = ["-std=c++17", "-x", "c++"] + [f"-I{d}" for d in (inc or [])]
    tu = ci.Index.create().parse(name, args=args, unsaved_files=[(name, source)])
    return _emit(tu, name)


def lift_file(path: str, inc=None) -> str:
    args = ["-std=c++17", "-x", "c++"] + [f"-I{d}" for d in (inc or [])]
    tu = ci.Index.create().parse(path, args=args)
    return _emit(tu, path)[0]


def main() -> None:
    import sys

    args = sys.argv[1:]
    out_path = None
    incs = []
    if "--out" in args:
        i = args.index("--out")
        out_path = args[i + 1]
        del args[i : i + 2]
    while "--inc" in args:
        i = args.index("--inc")
        incs.append(args[i + 1])
        del args[i : i + 2]
    path = args[0]
    args2 = ["-std=c++17", "-x", "c++"] + [f"-I{d}" for d in incs]
    tu = ci.Index.create().parse(path, args=args2)
    text, st = _emit(tu, path)
    if out_path:
        open(out_path, "w").write(text)
    print(text)
    cov = 100 * (1 - st["todo"] / max(st["nodes"], 1))
    print(
        f"# lift: {st['nodes']} nodes, {st['todo']} TODO  (~{cov:.0f}% mechanical)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
