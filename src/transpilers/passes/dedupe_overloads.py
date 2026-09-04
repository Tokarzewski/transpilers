"""Rename same-name overloads that collapse to an identical signature.

C++ allows overloading by signedness alone (`int` vs `unsigned int`), but
every backend here maps both to the same target scalar type (Mojo/Rust
`Int`/`i64`, ...) -- so two such overloads emit two methods/functions with
an *identical* signature in the same struct/module, which Mojo and Rust
both reject as a duplicate definition (neither supports overloading
by signedness either). Found stress-testing against a real-world corpus
(github.com/wassimj/Topologic, issue #79/#80): `Bitwise::NOT(int)` and
`Bitwise::NOT(unsigned int)`.

This runs after type inference (MIR param/return types are resolved) and
before lowering to LIR. Call sites here never did overload resolution by
argument type to begin with -- a call to `NOT(x)` couldn't already
distinguish which overload it meant -- so renaming the losing duplicate
turns a guaranteed compile failure into working code without making any
call site less correct than it implicitly already was.
"""

from __future__ import annotations

from transpilers.ir import mir


def _signature(fn: mir.MirFunction, *, skip_self: bool) -> tuple:
    params = fn.params[1:] if skip_self else fn.params
    return (fn.name, tuple(p.ty for p in params), fn.return_type)


def _dedupe(fns: list[mir.MirFunction], *, skip_self: bool) -> None:
    seen: dict[tuple, int] = {}
    for fn in fns:
        sig = _signature(fn, skip_self=skip_self)
        count = seen.get(sig, 0)
        seen[sig] = count + 1
        if count > 0:
            fn.name = f"{fn.name}_overload{count + 1}"


def dedupe_overloads(module: mir.MirModule) -> mir.MirModule:
    """Rename in place; returns *module* for chaining."""
    _dedupe(module.functions, skip_self=False)
    for struct in module.structs:
        _dedupe(struct.methods, skip_self=True)
    return module


__all__ = ["dedupe_overloads"]
