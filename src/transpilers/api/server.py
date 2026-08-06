"""FastAPI REST service wrapping the transpilers pipeline.

Endpoints:
    GET  /health                    — liveness + llm_available flag
    GET  /languages                 — supported source/target language lists
    POST /transpile                 — basic transpilation (no verification)
    POST /transpile/verify          — transpile + compile gate + structural fidelity
    POST /transpile/repair          — transpile + iterative LLM repair loop

Authentication (optional):
    Set TRANSPILER_API_KEY env var. If set, every request must carry
    `Authorization: Bearer <key>`. Omit to run unauthenticated (dev/localhost).

LLM configuration:
    ANTHROPIC_API_KEY or OPENAI_API_KEY + OPENAI_BASE_URL — enables LLM features.
    TRANSPILER_LLM_MODEL — override model (default: claude-opus-4-7).
    Without any LLM key, /transpile/repair returns 501 and use_llm is ignored.
"""

from __future__ import annotations

import hmac
import os
import subprocess
from functools import lru_cache
from importlib.metadata import version

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from transpilers.api.models import (
    HealthResponse,
    LanguagesResponse,
    RepairPassResult,
    RepairRequest,
    RepairResponse,
    TranspileRequest,
    TranspileResponse,
    VerifyRequest,
    VerifyResponse,
)
from transpilers.pipeline.stages import FRONTENDS, TARGETS, run_stages

try:
    _PKG_VERSION = version("transpilers")
except Exception:
    _PKG_VERSION = "0.0.0"

app = FastAPI(
    title="Transpilers API",
    description="Hybrid algorithmic + LLM source-to-source transpiler",
    version=_PKG_VERSION,
)

_bearer = HTTPBearer(auto_error=False)


def _check_auth(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    api_key = os.getenv("TRANSPILER_API_KEY")
    if not api_key:
        return
    if credentials is None or not hmac.compare_digest(credentials.credentials, api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


@lru_cache(maxsize=1)
def _llm_client():
    """Lazy singleton — returns None if no LLM credentials are configured."""
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"))
    if not (has_anthropic or has_openai):
        return None
    from transpilers.llm import LlmClient

    return LlmClient()


def _make_inferencer(client):
    if client is None:
        return None
    from transpilers.llm import make_llm_inferencer

    return make_llm_inferencer(client)


def _make_renamer(client):
    if client is None:
        return None
    from transpilers.llm import make_llm_renamer

    return make_llm_renamer(client)


def _require_llm():
    client = _llm_client()
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM features require ANTHROPIC_API_KEY or OPENAI_API_KEY to be set",
        )
    return client


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(version=_PKG_VERSION, llm_available=_llm_client() is not None)


@app.get("/languages", response_model=LanguagesResponse, tags=["meta"])
def languages():
    return LanguagesResponse(source=sorted(FRONTENDS), target=sorted(TARGETS))


@app.post("/transpile", response_model=TranspileResponse, tags=["transpile"])
def transpile_basic(req: TranspileRequest, _auth=Depends(_check_auth)):
    if req.source_lang not in FRONTENDS:
        raise HTTPException(400, f"Unknown source language {req.source_lang!r}")
    if req.target not in TARGETS:
        raise HTTPException(400, f"Unknown target language {req.target!r}")

    client = _llm_client() if (req.use_llm or req.llm_rename) else None
    if (req.use_llm or req.llm_rename) and client is None:
        raise HTTPException(
            501, "LLM features require ANTHROPIC_API_KEY or OPENAI_API_KEY"
        )

    llm_fill = _make_inferencer(client) if req.use_llm else None
    llm_rename = _make_renamer(client) if req.llm_rename else None
    ir_hints = _extract_ir_hints(req)

    if req.path == "python_pivot":
        if req.source_lang != "cpp":
            raise HTTPException(400, "python_pivot path requires source_lang=cpp")
        from transpilers.cli.main import transpile_cpp_via_python

        _, output = transpile_cpp_via_python(
            req.source, req.target, llm_fill=llm_fill, ir_hints=ir_hints
        )
    else:
        try:
            output = run_stages(
                req.source,
                source_lang=req.source_lang,
                target=req.target,
                llm_fill=llm_fill,
                llm_rename_fill=llm_rename,
                ir_hints=ir_hints,
            ).output
        except Exception as exc:
            raise HTTPException(422, str(exc)) from exc

    return TranspileResponse(
        output=output, source_lang=req.source_lang, target=req.target
    )


@app.post("/transpile/verify", response_model=VerifyResponse, tags=["transpile"])
def transpile_verify(req: VerifyRequest, _auth=Depends(_check_auth)):
    client = _llm_client() if (req.use_llm or req.llm_rename) else None
    if (req.use_llm or req.llm_rename) and client is None:
        raise HTTPException(
            501, "LLM features require ANTHROPIC_API_KEY or OPENAI_API_KEY"
        )

    llm_fill = _make_inferencer(client) if req.use_llm else None
    llm_rename = _make_renamer(client) if req.llm_rename else None
    ir_hints = _extract_ir_hints(req)

    try:
        trace = run_stages(
            req.source,
            source_lang=req.source_lang,
            target=req.target,
            llm_fill=llm_fill,
            llm_rename_fill=llm_rename,
            ir_hints=ir_hints,
        )
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc

    output = trace.output
    _, _, verify_fn = TARGETS[req.target]
    try:
        compile_result = verify_fn(output)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, f"{req.target} compiler timed out") from exc

    structural = None
    if compile_result.ok and req.fidelity == "structural":
        from transpilers.verify.structural import check_structural_fidelity

        report = check_structural_fidelity(trace.hir, trace.lir)
        from transpilers.api.models import StructuralResult

        structural = StructuralResult(
            ok=report.ok, summary=report.summary() if not report.ok else ""
        )

    from transpilers.api.models import CompileGateResult

    return VerifyResponse(
        output=output,
        source_lang=req.source_lang,
        target=req.target,
        compile=CompileGateResult(ok=compile_result.ok, stderr=compile_result.stderr),
        structural=structural,
    )


@app.post("/transpile/repair", response_model=RepairResponse, tags=["transpile"])
def transpile_repair(req: RepairRequest, _auth=Depends(_check_auth)):
    client = _require_llm()
    from transpilers.repair.repair import repair

    try:
        result = repair(
            req.source,
            source_lang=req.source_lang,
            target=req.target,
            llm_client=client,
            max_passes=req.max_passes,
        )
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc

    return RepairResponse(
        output=result.code,
        source_lang=req.source_lang,
        target=req.target,
        passed=result.passed,
        passes=result.passes,
        history=[
            RepairPassResult(
                attempt=p.attempt,
                compile_ok=p.compile_ok,
                error=p.error,
                fix_applied=p.fix_applied,
            )
            for p in result.history
        ],
    )


def _extract_ir_hints(req: TranspileRequest):
    if not req.ir_augment or req.source_lang not in ("c", "cpp"):
        return None
    try:
        import tempfile
        from pathlib import Path
        from transpilers.passes.ir_preload import extract_ir_types

        with tempfile.NamedTemporaryFile(
            suffix=".cpp" if req.source_lang == "cpp" else ".c",
            delete=False,
            mode="w",
        ) as f:
            f.write(req.source)
            tmp_path = Path(f.name)
        return extract_ir_types(tmp_path)
    except Exception:
        return None


def main():
    import uvicorn

    uvicorn.run(
        "transpilers.api.server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "").lower() in ("1", "true"),
    )


if __name__ == "__main__":
    main()
