"""Map C++ type-text strings (as used in HIR annotations) to ``Type`` instances.

The ``_type_text`` helper in ``types.py`` returns *string* annotations
(e.g. ``"int"``, ``"list[int]"``) so the HIR layer can carry them
around cheaply. The ground-truth pass at issue #50 needs the same
information in *typed* form so ``isinstance(ty, UnknownT)`` checks
work. This module is the reverse mapping: string -> ``Type`` instance.

Kept small and total: any unknown spelling collapses to
``UnknownT()`` rather than raising, because the ground-truth
extractor is on the hot path of every parse.
"""

from __future__ import annotations


from transpilers.ir.types import (
    BoolT,
    FloatT,
    IntT,
    ListT,
    NoneT,
    SimdT,
    StrT,
    Type,
    UnknownT,
)


_PRIMITIVES: dict[str, Type] = {
    "int": IntT(),
    "float": FloatT(),
    "bool": BoolT(),
    "str": StrT(),
    "None": NoneT(),
}


def text_to_type(text: str) -> Type:
    """Convert a HIR-style type-text string to a ``Type`` instance.

    Recognised shapes:

    * ``int``, ``float``, ``bool``, ``str``, ``None`` -- the five
      primitive transpiler types.
    * ``list[T]`` -- the IR's list shape, where ``T`` is recursively
      converted (so ``list[list[int]]`` and ``list[list[list[int]]]``
      both work).
    * ``simd[<elem>, <lanes>]`` -- the IR's SIMD shape.
    * ``UnknownT`` or any string starting with ``unknown`` -- leaves
      the hole in place.
    * Anything else -- returns ``UnknownT()`` so the call site stays
      total.
    """
    if text is None:
        return UnknownT()
    text = text.strip()
    if text in _PRIMITIVES:
        return _PRIMITIVES[text]
    if text.startswith("list[") and text.endswith("]"):
        inner = text[len("list[") : -1].strip()
        return ListT(elem=text_to_type(inner))
    if text.startswith("simd[") and text.endswith("]"):
        inner = text[len("simd[") : -1].strip()
        elem_text, _, lanes_text = inner.rpartition(",")
        try:
            lanes = int(lanes_text.strip())
        except ValueError:
            return UnknownT(hint=text)
        return SimdT(elem=text_to_type(elem_text.strip()), lanes=lanes)
    if text.startswith("UnknownT") or text == "unknown":
        return UnknownT()
    return UnknownT(hint=text)


__all__ = ["text_to_type"]
