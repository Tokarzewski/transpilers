"""Basic smoke tests for the FastAPI REST endpoint.

Requires: uv sync --group api --group dev
Run with: uv run pytest tests/test_api.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed; run: uv sync --group api")
pytest.importorskip("httpx", reason="httpx not installed; run: uv sync --group dev")

from fastapi.testclient import TestClient

from transpilers.api.server import app

_client = TestClient(app)

_FIB_PY = """\
def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
"""


def test_health():
    r = _client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "llm_available" in data


def test_languages():
    r = _client.get("/languages")
    assert r.status_code == 200
    data = r.json()
    assert "python" in data["source"]
    assert "rust" in data["target"]
    assert "mojo" in data["target"]


def test_transpile_python_to_rust():
    r = _client.post(
        "/transpile",
        json={"source": _FIB_PY, "source_lang": "python", "target": "rust"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "output" in data
    assert data["source_lang"] == "python"
    assert data["target"] == "rust"
    assert len(data["output"]) > 0


def test_transpile_python_to_mojo():
    r = _client.post(
        "/transpile",
        json={"source": _FIB_PY, "source_lang": "python", "target": "mojo"},
    )
    assert r.status_code == 200
    assert len(r.json()["output"]) > 0


def test_transpile_unknown_source_lang():
    r = _client.post(
        "/transpile", json={"source": "x", "source_lang": "brainfuck", "target": "rust"}
    )
    assert r.status_code == 422  # pydantic rejects unknown Literal


def test_transpile_unknown_target():
    r = _client.post(
        "/transpile", json={"source": "x", "source_lang": "python", "target": "cobol"}
    )
    assert r.status_code == 422


def test_verify_python_to_rust():
    r = _client.post(
        "/transpile/verify",
        json={"source": _FIB_PY, "source_lang": "python", "target": "rust"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "compile" in data
    assert isinstance(data["compile"]["ok"], bool)


def test_verify_maps_compiler_timeout_to_504(monkeypatch):
    """None of the native verifiers used to set a `subprocess` timeout, so a
    pathological input could hang the compiler indefinitely -- a real DoS
    vector on this network-facing, auth-optional endpoint. Now that they do,
    the resulting `subprocess.TimeoutExpired` must surface as a clean 504,
    not an unhandled 500."""
    import subprocess

    from transpilers.api import server

    def _timeout(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd=["rustc"], timeout=30)

    monkeypatch.setattr(
        server,
        "TARGETS",
        {
            **server.TARGETS,
            "rust": (server.TARGETS["rust"][0], server.TARGETS["rust"][1], _timeout),
        },
    )
    r = _client.post(
        "/transpile/verify",
        json={"source": _FIB_PY, "source_lang": "python", "target": "rust"},
    )
    assert r.status_code == 504


def test_repair_without_llm_returns_501():
    r = _client.post(
        "/transpile/repair",
        json={"source": _FIB_PY, "source_lang": "python", "target": "rust"},
    )
    # 501 when no LLM key is configured in test environment
    assert r.status_code in (200, 501)


def test_auth_rejected_with_wrong_key(monkeypatch):
    monkeypatch.setenv("TRANSPILER_API_KEY", "secret")
    # Reload the cached client after env change
    from transpilers.api import server
    from fastapi.testclient import TestClient

    c = TestClient(server.app)
    r = c.post(
        "/transpile",
        json={"source": _FIB_PY, "source_lang": "python", "target": "rust"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401


def test_auth_accepted_with_correct_key(monkeypatch):
    monkeypatch.setenv("TRANSPILER_API_KEY", "secret")
    from transpilers.api import server
    from fastapi.testclient import TestClient

    c = TestClient(server.app)
    r = c.post(
        "/transpile",
        json={"source": _FIB_PY, "source_lang": "python", "target": "rust"},
        headers={"Authorization": "Bearer secret"},
    )
    assert r.status_code == 200
