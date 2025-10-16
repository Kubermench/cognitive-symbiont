from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SymbiontBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class MemoryConfig(SymbiontBaseModel):
    layer: str = Field(default="local")

    @field_validator("layer")
    @classmethod
    def validate_layer(cls, value: str) -> str:
        if not value:
            raise ValueError("memory.layer must be a non-empty string")
        return value


class LLMFallbackConfig(SymbiontBaseModel):
    provider: Optional[str] = None
    cmd: Optional[str] = None
    timeout_seconds: Optional[int] = None


class LLMCloudConfig(SymbiontBaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout_seconds: Optional[int] = None


class LLMConfig(SymbiontBaseModel):
    provider: str = Field(default="ollama")
    model: str = Field(default="phi3:mini")
    mode: str = Field(default="local")
    timeout_seconds: int = Field(default=180, ge=30)
    cmd: Optional[str] = None
    fallback: Optional[LLMFallbackConfig] = None
    cloud: Optional[LLMCloudConfig] = None

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        allowed = {"local", "cloud", "hybrid"}
        if value not in allowed:
            raise ValueError(f"llm.mode must be one of {', '.join(sorted(allowed))}")
        return value


class InitiativeStateConfig(SymbiontBaseModel):
    path: Optional[str] = None


class InitiativeConfig(SymbiontBaseModel):
    enabled: bool = Field(default=False)
    repo_path: str = Field(default=".")
    state: InitiativeStateConfig = Field(default_factory=InitiativeStateConfig)


class ForesightConfig(SymbiontBaseModel):
    enabled: bool = Field(default=False)


class RetrievalExternalConfig(SymbiontBaseModel):
    enabled: bool = Field(default=False)


class RetrievalConfig(SymbiontBaseModel):
    external: RetrievalExternalConfig = Field(default_factory=RetrievalExternalConfig)


class EvolutionConfig(SymbiontBaseModel):
    enabled: bool = Field(default=False)


class ToolsConfig(SymbiontBaseModel):
    allow_code_runner: bool = Field(default=True)
    allow_files: bool = Field(default=True)
    network_access: bool = Field(default=False)


class GuardConfig(SymbiontBaseModel):
    auto_approve_safe: bool = Field(default=False)


class SymbiontConfig(SymbiontBaseModel):
    env: str = Field(default="dev")
    data_root: Optional[str] = Field(default="./data")
    db_path: str = Field(default="./data/symbiont.db")
    max_tokens: Optional[int] = Field(default=4000, ge=512)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    initiative: InitiativeConfig = Field(default_factory=InitiativeConfig)
    foresight: ForesightConfig = Field(default_factory=ForesightConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)

    @field_validator("env")
    @classmethod
    def validate_env(cls, value: str) -> str:
        if not value:
            raise ValueError("env must be a non-empty string")
        return value

    @field_validator("db_path")
    @classmethod
    def normalize_db_path(cls, value: str) -> str:
        if not value:
            raise ValueError("db_path must be provided")
        return str(Path(value))

    def dump(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)
