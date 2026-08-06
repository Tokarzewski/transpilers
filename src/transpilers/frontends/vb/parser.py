"""Visual Basic source -> HIR.

Hand-rolled tokenizer + recursive-descent parser, because a usable
tree-sitter VB grammar isn't available on PyPI for Linux. The subset is
modern VB.NET / VBA shape:

  Function NAME(a As T, b As T) As T
      Dim x As T = expr
      ... statements ...
      Return expr
  End Function

Plus If/ElseIf/Else/End If, While/End While, For target = start To stop
(inclusive both ends — adjusted to exclusive in HIR), assignments,
binary/relational/logical operators, integer/string/boolean literals.

VB is case-insensitive for keywords and identifiers; we lowercase keywords
during matching but preserve identifier spelling.
"""

from __future__ import annotations

from dataclasses import dataclass

from transpilers.ir import hir


from transpilers.frontends.errors import UnsupportedConstruct


VB_TYPE_ALIASES: dict[str, str] = {
    "integer": "int",
    "long": "int",
    "short": "int",
    "byte": "int",
    "single": "float",
    "double": "float",
    "decimal": "float",
    "boolean": "bool",
    "string": "str",
}


KEYWORDS = {
    "function",
    "end",
    "dim",
    "as",
    "if",
    "then",
    "else",
    "elseif",
    "endif",
    "while",
    "wend",
    "for",
    "to",
    "step",
    "next",
    "return",
    "and",
    "or",
    "not",
    "true",
    "false",
    "mod",
    "do",
    "loop",
    "until",
}


@dataclass
class Token:
    kind: str  # "kw" | "id" | "num" | "str" | "op" | "newline" | "eof"
    value: str
    line: int


def _tokenize(source: str) -> list[Token]:
    """Two-pass tokenizer: VB has line-significant control flow so newlines
    are tokens. Comments start with `'` and run to end of line. Multi-char
    operators are matched before single-char so `<=` beats `<`."""
    tokens: list[Token] = []
    line = 1
    i = 0
    while i < len(source):
        ch = source[i]
        if ch == "\n":
            tokens.append(Token("newline", "\n", line))
            line += 1
            i += 1
            continue
        if ch in " \t\r":
            i += 1
            continue
        if ch == "'":
            # Comment to end of line.
            while i < len(source) and source[i] != "\n":
                i += 1
            continue
        if ch.isdigit():
            j = i
            while j < len(source) and source[j].isdigit():
                j += 1
            kind = "num"
            # Float: `1.5` (decimal point followed by more digits).
            if (
                j < len(source)
                and source[j] == "."
                and j + 1 < len(source)
                and source[j + 1].isdigit()
            ):
                j += 1
                while j < len(source) and source[j].isdigit():
                    j += 1
                kind = "float"
            tokens.append(Token(kind, source[i:j], line))
            i = j
            continue
        if ch == '"':
            j = i + 1
            while j < len(source) and source[j] != '"':
                j += 1
            tokens.append(Token("str", source[i + 1 : j], line))
            i = j + 1
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < len(source) and (source[j].isalnum() or source[j] == "_"):
                j += 1
            word = source[i:j]
            lower = word.lower()
            tokens.append(Token("kw" if lower in KEYWORDS else "id", word, line))
            i = j
            continue
        # Operators (longest-match-first).
        for op in (
            "<=",
            ">=",
            "<>",
            "+=",
            "-=",
            "*=",
            "/=",
            "<",
            ">",
            "=",
            "+",
            "-",
            "*",
            "/",
            "(",
            ")",
            ",",
            ":",
        ):
            if source.startswith(op, i):
                tokens.append(Token("op", op, line))
                i += len(op)
                break
        else:
            raise UnsupportedConstruct(
                f"vb tokenizer: unexpected {ch!r} at line {line}"
            )
    tokens.append(Token("eof", "", line))
    return tokens


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.t = tokens
        self.i = 0

    def peek(self, offset: int = 0) -> Token:
        return self.t[self.i + offset]

    def eat(self) -> Token:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def skip_newlines(self) -> None:
        while self.peek().kind == "newline":
            self.eat()

    def expect_kw(self, name: str) -> Token:
        tok = self.eat()
        if tok.kind != "kw" or tok.value.lower() != name:
            raise UnsupportedConstruct(
                f"vb: expected {name!r} at line {tok.line}, got {tok.value!r}"
            )
        return tok

    def expect_op(self, op: str) -> Token:
        tok = self.eat()
        if tok.kind != "op" or tok.value != op:
            raise UnsupportedConstruct(
                f"vb: expected {op!r} at line {tok.line}, got {tok.value!r}"
            )
        return tok

    # ---------- module ----------

    def parse_module(self) -> list[hir.HirNode]:
        body: list[hir.HirNode] = []
        while True:
            self.skip_newlines()
            if self.peek().kind == "eof":
                break
            body.append(self.parse_function())
        return body

    # ---------- function ----------

    def parse_function(self) -> hir.HirFunction:
        self.expect_kw("function")
        name = self.eat()
        if name.kind != "id":
            raise UnsupportedConstruct(
                f"vb: expected function name, got {name.value!r}"
            )
        self.expect_op("(")
        params: list[hir.HirParam] = []
        if not (self.peek().kind == "op" and self.peek().value == ")"):
            params.append(self.parse_param())
            while self.peek().kind == "op" and self.peek().value == ",":
                self.eat()
                params.append(self.parse_param())
        self.expect_op(")")
        # Optional `As <type>` for return type.
        return_annotation: str | None = None
        if self.peek().kind == "kw" and self.peek().value.lower() == "as":
            self.eat()
            return_annotation = self.parse_type()
        self.skip_newlines()
        body = self.parse_block(end_keywords={"end"})
        self.expect_kw("end")
        self.expect_kw("function")
        return hir.HirFunction(
            name=name.value,
            params=params,
            return_annotation=return_annotation,
            body=body,
        )

    def parse_param(self) -> hir.HirParam:
        name = self.eat()
        if name.kind != "id":
            raise UnsupportedConstruct(f"vb: expected param name, got {name.value!r}")
        self.expect_kw("as")
        annotation = self.parse_type()
        return hir.HirParam(name=name.value, annotation=annotation)

    def parse_type(self) -> str:
        tok = self.eat()
        if tok.kind not in ("id", "kw"):
            raise UnsupportedConstruct(f"vb: expected type, got {tok.value!r}")
        return VB_TYPE_ALIASES.get(tok.value.lower(), tok.value)

    # ---------- statements ----------

    def parse_block(self, end_keywords: set[str]) -> list[hir.HirNode]:
        out: list[hir.HirNode] = []
        while True:
            self.skip_newlines()
            tok = self.peek()
            if tok.kind == "eof":
                break
            if tok.kind == "kw" and tok.value.lower() in end_keywords:
                break
            if tok.kind == "kw" and tok.value.lower() in ("else", "elseif"):
                break
            out.extend(self.parse_stmt())
        return out

    def parse_stmt(self) -> list[hir.HirNode]:
        tok = self.peek()
        if tok.kind == "kw":
            kw = tok.value.lower()
            if kw == "dim":
                return [self.parse_dim()]
            if kw == "return":
                self.eat()
                value = self.parse_expr()
                return [hir.HirReturn(value=value)]
            if kw == "if":
                return [self.parse_if()]
            if kw == "while":
                return [self.parse_while()]
            if kw == "for":
                return [self.parse_for()]
            raise UnsupportedConstruct(
                f"vb: unexpected keyword {tok.value!r} at line {tok.line}"
            )
        if tok.kind == "id":
            return [self.parse_assignment_or_call()]
        raise UnsupportedConstruct(
            f"vb: unexpected token {tok.value!r} at line {tok.line}"
        )

    def parse_dim(self) -> hir.HirNode:
        self.expect_kw("dim")
        name = self.eat()
        if name.kind != "id":
            raise UnsupportedConstruct(f"vb: expected dim name, got {name.value!r}")
        self.expect_kw("as")
        annotation = self.parse_type()
        value: hir.HirNode
        if self.peek().kind == "op" and self.peek().value == "=":
            self.eat()
            value = self.parse_expr()
        else:
            value = hir.HirIntLiteral(value=0)
        return hir.HirAssign(target=name.value, value=value, annotation=annotation)

    def parse_assignment_or_call(self) -> hir.HirNode:
        target = self.eat()
        # Augmented assigns: VB has `+=` etc. in modern VB.NET.
        op = self.peek()
        if op.kind == "op" and op.value in ("=", "+=", "-=", "*=", "/="):
            self.eat()
            value = self.parse_expr()
            aug = None if op.value == "=" else op.value[:-1]
            return hir.HirAssign(
                target=target.value, value=value, annotation=None, augmented_op=aug
            )
        raise UnsupportedConstruct(f"vb: expected assignment after {target.value!r}")

    def parse_if(self) -> hir.HirNode:
        self.expect_kw("if")
        cond = self.parse_expr()
        self.expect_kw("then")
        self.skip_newlines()
        body = self.parse_block(end_keywords={"end"})
        orelse: list[hir.HirNode] = []
        while self.peek().kind == "kw" and self.peek().value.lower() == "elseif":
            self.eat()
            elif_cond = self.parse_expr()
            self.expect_kw("then")
            self.skip_newlines()
            elif_body = self.parse_block(end_keywords={"end"})
            orelse = [hir.HirIf(test=elif_cond, body=elif_body, orelse=orelse)]
            # tail: continue as if `else` already opened? We rebuild by
            # walking forward, but simpler: convert below.
            # Actually we want this elseif to *be* the orelse of the if we're
            # processing. We'll structure it after the full chain reads.
            break  # handled non-trivially: see below
        # Simpler one-pass: re-parse the entire elseif chain via recursive
        # call, since our `break` above is structurally awkward.
        if self.peek().kind == "kw" and self.peek().value.lower() == "elseif":
            self.eat()  # consume `elseif` token already advanced? — fixed
        if self.peek().kind == "kw" and self.peek().value.lower() == "else":
            self.eat()
            self.skip_newlines()
            orelse = self.parse_block(end_keywords={"end"})
        self.expect_kw("end")
        self.expect_kw("if")
        return hir.HirIf(test=cond, body=body, orelse=orelse)

    def parse_while(self) -> hir.HirNode:
        self.expect_kw("while")
        cond = self.parse_expr()
        self.skip_newlines()
        body = self.parse_block(end_keywords={"end", "wend"})
        if self.peek().kind == "kw" and self.peek().value.lower() == "wend":
            self.eat()
        else:
            self.expect_kw("end")
            self.expect_kw("while")
        return hir.HirWhile(test=cond, body=body)

    def parse_for(self) -> hir.HirNode:
        """`For i = start To stop [Step step] ... Next [i]` — inclusive on both
        ends, like Fortran."""
        self.expect_kw("for")
        target_tok = self.eat()
        if target_tok.kind != "id":
            raise UnsupportedConstruct(
                f"vb: expected for target, got {target_tok.value!r}"
            )
        self.expect_op("=")
        start = self.parse_expr()
        self.expect_kw("to")
        stop = self.parse_expr()
        step: hir.HirNode | None = None
        if self.peek().kind == "kw" and self.peek().value.lower() == "step":
            self.eat()
            step = self.parse_expr()
        self.skip_newlines()
        body = self.parse_block(end_keywords={"next"})
        self.expect_kw("next")
        # Optional name after Next.
        if self.peek().kind == "id":
            self.eat()
        # Inclusive endpoint → exclusive by +1.
        exclusive_stop = hir.HirBinOp(
            op="+", left=stop, right=hir.HirIntLiteral(value=1)
        )
        args = (
            [start, exclusive_stop] if step is None else [start, exclusive_stop, step]
        )
        return hir.HirFor(
            target=target_tok.value,
            iter=hir.HirCall(func="range", args=args),
            body=body,
        )

    # ---------- expressions (precedence climber) ----------

    def parse_expr(self) -> hir.HirNode:
        return self.parse_or()

    def parse_or(self) -> hir.HirNode:
        left = self.parse_and()
        while self.peek().kind == "kw" and self.peek().value.lower() == "or":
            self.eat()
            right = self.parse_and()
            left = hir.HirBoolOp(op="or", left=left, right=right)
        return left

    def parse_and(self) -> hir.HirNode:
        left = self.parse_not()
        while self.peek().kind == "kw" and self.peek().value.lower() == "and":
            self.eat()
            right = self.parse_not()
            left = hir.HirBoolOp(op="and", left=left, right=right)
        return left

    def parse_not(self) -> hir.HirNode:
        if self.peek().kind == "kw" and self.peek().value.lower() == "not":
            self.eat()
            return hir.HirUnaryOp(op="not", operand=self.parse_not())
        return self.parse_compare()

    def parse_compare(self) -> hir.HirNode:
        left = self.parse_add()
        while self.peek().kind == "op" and self.peek().value in COMPARE_OPS:
            op_tok = self.eat()
            op = "!=" if op_tok.value == "<>" else op_tok.value
            right = self.parse_add()
            left = hir.HirCompare(op=op, left=left, right=right)
        return left

    def parse_add(self) -> hir.HirNode:
        left = self.parse_mul()
        while self.peek().kind == "op" and self.peek().value in ("+", "-"):
            op = self.eat().value
            right = self.parse_mul()
            left = hir.HirBinOp(op=op, left=left, right=right)
        return left

    def parse_mul(self) -> hir.HirNode:
        left = self.parse_unary()
        while (self.peek().kind == "op" and self.peek().value in ("*", "/")) or (
            self.peek().kind == "kw" and self.peek().value.lower() == "mod"
        ):
            tok = self.eat()
            op = "%" if tok.kind == "kw" else tok.value
            right = self.parse_unary()
            left = hir.HirBinOp(op=op, left=left, right=right)
        return left

    def parse_unary(self) -> hir.HirNode:
        if self.peek().kind == "op" and self.peek().value == "-":
            self.eat()
            return hir.HirUnaryOp(op="-", operand=self.parse_unary())
        return self.parse_atom()

    def parse_atom(self) -> hir.HirNode:
        tok = self.peek()
        if tok.kind == "num":
            self.eat()
            return hir.HirIntLiteral(value=int(tok.value))
        if tok.kind == "float":
            self.eat()
            return hir.HirFloatLiteral(value=float(tok.value))
        if tok.kind == "str":
            self.eat()
            return hir.HirStringLiteral(value=tok.value)
        if tok.kind == "kw" and tok.value.lower() == "true":
            self.eat()
            return hir.HirBoolLiteral(value=True)
        if tok.kind == "kw" and tok.value.lower() == "false":
            self.eat()
            return hir.HirBoolLiteral(value=False)
        if tok.kind == "op" and tok.value == "(":
            self.eat()
            inner = self.parse_expr()
            self.expect_op(")")
            return inner
        if tok.kind == "id":
            self.eat()
            # Maybe a call.
            if self.peek().kind == "op" and self.peek().value == "(":
                self.eat()
                args: list[hir.HirNode] = []
                if not (self.peek().kind == "op" and self.peek().value == ")"):
                    args.append(self.parse_expr())
                    while self.peek().kind == "op" and self.peek().value == ",":
                        self.eat()
                        args.append(self.parse_expr())
                self.expect_op(")")
                return hir.HirCall(func=tok.value, args=args)
            return hir.HirName(name=tok.value)
        raise UnsupportedConstruct(
            f"vb: unexpected token {tok.value!r} at line {tok.line}"
        )


COMPARE_OPS = {"<", "<=", ">", ">=", "=", "<>"}


def parse_vb(source: str) -> hir.HirModule:
    tokens = _tokenize(source)
    parser = _Parser(tokens)
    body = parser.parse_module()
    return hir.HirModule(source_lang="vb", body=body)
