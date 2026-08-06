"""Wire the LLM client into the type-inference pass as a typed-hole filler.

The pass takes a generic `LlmFill` callable so unit tests can substitute a
fake. This module supplies the production wiring: it builds a TypedHole per
unresolved slot, calls the cached LLM client, validates the JSON response,
and returns a Type drawn from the lattice.
"""

from __future__ import annotations

import json

from transpilers.ir.types import BoolT, FloatT, IntT, ListT, NoneT, StrT, Type
from transpilers.llm.client import LlmClient, TypedHole


ALLOWED: dict[str, Type] = {
    "int": IntT(),
    "float": FloatT(),
    "bool": BoolT(),
    "str": StrT(),
    "none": NoneT(),
    "list[int]": ListT(elem=IntT()),
    "list[float]": ListT(elem=FloatT()),
    "list[bool]": ListT(elem=BoolT()),
    "list[str]": ListT(elem=StrT()),
}


def make_llm_inferencer(client: LlmClient):
    def fill(name: str, context: dict) -> Type:
        hole = TypedHole(
            kind="python_type_inference",
            context={"slot": name, **context},
            validate=_parse_type,
        )
        return client.fill(hole)

    return fill


def make_llm_renamer(client: LlmClient):
    """LlmFill-compatible callable for the variable-rename pass. Returns
    the proposed name string (already validated as a Python identifier)."""

    def fill(old_name: str, context: dict) -> str:
        hole = TypedHole(
            kind="variable_rename",
            context={"old_name": old_name, **context},
            validate=_parse_rename,
        )
        return client.fill(hole)

    return fill


def _parse_type(raw: str) -> Type:
    payload = json.loads(raw.strip())
    label = payload["type"].lower().strip()
    if label not in ALLOWED:
        raise ValueError(
            f"LLM returned unknown type {label!r}; allowed: {sorted(ALLOWED)}"
        )
    return ALLOWED[label]


def _parse_rename(raw: str) -> str:
    payload = json.loads(raw.strip())
    name = payload["name"].strip()
    if not name.isidentifier():
        raise ValueError(f"LLM returned non-identifier name {name!r}")
    return name
