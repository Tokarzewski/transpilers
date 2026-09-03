// A class with public fields and methods, instantiated in another
// function. Exercises the full struct + method + field-access pipeline:
// struct-definition, method emission, struct construction, field
// assignment, and method call.
//
//   $ transpile examples/classes/point.cpp --target rust   --verify
//   $ transpile examples/classes/point.cpp --target c      --verify
//   $ transpile examples/classes/point.cpp --target go     --verify
//   $ transpile examples/classes/point.cpp --target mojo   --verify
//   $ transpile examples/classes/point.cpp --target python --verify
//   $ transpile examples/classes/point.cpp --target fortran --verify

class Point {
public:
    int x;
    int y;

    int sum() {
        return this->x + this->y;
    }

    int scale(int factor) {
        return this->x * factor + this->y * factor;
    }
};

int compute() {
    Point p;
    p.x = 3;
    p.y = 4;
    return p.sum();
}

int scaled(int factor) {
    Point p;
    p.x = 1;
    p.y = 2;
    return p.scale(factor);
}
