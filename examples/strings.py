# Strings: target divergence in action.
# Rust:  format!() for concat, String::from() for literals

def shout() -> str:
    return "loud"


def greet(name: str) -> str:
    return "hello, " + name


def banner(title: str) -> str:
    return "=== " + title + " ==="
