# Class examples

Cross-language demos of the class/struct subset: fields, methods,
construction, field assignment, method calls. The canonical source is
[`point.cpp`](point.cpp) — the C++ frontend has the broadest coverage of
this subset because libclang's AST exposes the explicit `class` shape.

## Try it

```sh
for tgt in rust c mojo python fortran; do
    just transpile examples/classes/point.cpp $tgt
done
```

Each target emits idiomatic code for the same source. The IR shape is
shared; the per-target LIRs handle the syntactic differences:

| Target | Struct emission | Field access | Method receiver | Construction |
|--------|----------------|--------------|-----------------|--------------|
| Rust | `struct` + `impl` | `.` | `&self` | `Point { x: 0, y: 0 }` |
| C | `typedef struct {...} T` | `->` (self) / `.` | `T *self` | `(Point){.x = 0, .y = 0}` |
| Mojo | `@fieldwise_init struct T(Copyable, Movable)` | `.` | `self` | `Point(0, 0)` |
| Python | `@dataclass class T` | `.` | `self` | `Point(0, 0)` |
| Fortran | `type :: T ... end type T` in a module | `%` | `type(T), intent(in)` | `Point(0, 0)` |

## What's not yet supported

These trigger refusals at the C++ frontend or earlier passes:

- Inheritance (`class D : public Base`)
- Templates (`template<typename T> ...`)
- Private/protected members
- Constructors with explicit bodies
- Static methods / static fields
- Operator overloading
- Field assignment to nested types beyond one level
