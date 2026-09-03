from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceLang = Literal[
    "python",
    "c",
    "cpp",
    "java",
    "csharp",
    "typescript",
    "javascript",
    "fortran",
    "vb",
    "asm",
]
TargetLang = Literal["rust", "c", "mojo", "python", "fortran"]
Path = Literal["direct", "python_pivot"]
Fidelity = Literal["structural", "idiomatic"]


class TranspileRequest(BaseModel):
    source: str = Field(..., description="Source code to transpile")
    source_lang: SourceLang = Field("python", description="Source language")
    target: TargetLang = Field("rust", description="Target language")
    use_llm: bool = Field(
        False, description="Use LLM to fill type holes (requires API key)"
    )
    llm_rename: bool = Field(False, description="Use LLM to rename opaque locals")
    ir_augment: bool = Field(
        False, description="Pre-seed types from LLVM IR (C/C++ only, requires clang)"
    )
    path: Path = Field("direct", description="Translation path")


class VerifyRequest(TranspileRequest):
    fidelity: Fidelity = Field("structural", description="Fidelity check level")


class RepairRequest(BaseModel):
    source: str = Field(..., description="Source code to transpile and repair")
    source_lang: SourceLang = Field("python")
    target: TargetLang = Field("rust")
    max_passes: int = Field(3, ge=1, le=10, description="Max repair iterations")


class CompileGateResult(BaseModel):
    ok: bool
    stderr: str = ""


class StructuralResult(BaseModel):
    ok: bool
    summary: str = ""


class RepairPassResult(BaseModel):
    attempt: int
    compile_ok: bool
    error: str = ""
    fix_applied: str = ""


class TranspileResponse(BaseModel):
    output: str
    source_lang: str
    target: str


class VerifyResponse(TranspileResponse):
    compile: CompileGateResult
    structural: StructuralResult | None = None


class RepairResponse(BaseModel):
    output: str
    source_lang: str
    target: str
    passed: bool
    passes: int
    history: list[RepairPassResult]


class LanguagesResponse(BaseModel):
    source: list[str]
    target: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    llm_available: bool
