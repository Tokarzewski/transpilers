"""LIR base node (shared marker for every target dialect)."""

from __future__ import annotations


class LirNode:
    """Base class for all LIR nodes.

    Every subclass that is a ``@dataclass`` automatically gets a
    ``_hir_provenance_id: int`` instance attribute (set to 0 by default,
    set to a real HIR node id during lowering).
    """

    _hir_provenance_id: int

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        old_post = getattr(cls, "__post_init__", None)

        def _post_init(self):
            self._hir_provenance_id = 0
            if old_post:
                old_post(self)

        cls.__post_init__ = _post_init


__all__ = ["LirNode"]
