from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class NodeModel(BaseModel):
    agent: str
    next: Optional[str] = None
    on_success: Optional[str] = None
    on_failure: Optional[str] = None
    on_block: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class GraphSectionModel(BaseModel):
    start: str
    nodes: Dict[str, NodeModel]
    parallel: List[List[str]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @field_validator("parallel", mode="before")
    @classmethod
    def ensure_parallel(cls, value: Any) -> List[List[str]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("parallel must be a list of lists")
        for group in value:
            if not isinstance(group, list):
                raise ValueError("parallel entries must be lists of node names")
        return value


class GraphFileModel(BaseModel):
    graph: GraphSectionModel
    crew_config: Optional[str] = None
    crews: Optional[str] = None
    simulation: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")

    def require_crew_path(self) -> str:
        if self.crew_config:
            return self.crew_config
        if self.crews:
            return self.crews
        raise ValueError("Graph spec missing 'crew_config'")


class AgentModel(BaseModel):
    role: str
    llm: Dict[str, Any] = Field(default_factory=dict)
    cache: Optional[str] = None
    tools: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class CrewModel(BaseModel):
    sequence: List[str] = Field(default_factory=list)
    roles: Optional[List[str]] = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("sequence", mode="before")
    @classmethod
    def ensure_sequence(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raise ValueError("Sequence must be a list of role names")
        return list(value)

    @field_validator("roles", mode="before")
    @classmethod
    def ensure_roles(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            raise ValueError("Roles must be a list of role names")
        return list(value)

    def resolved_sequence(self) -> List[str]:
        if self.sequence:
            return self.sequence
        if self.roles:
            return self.roles
        return []


class CrewFileModel(BaseModel):
    agents: Dict[str, AgentModel] = Field(default_factory=dict)
    crew: Dict[str, CrewModel] = Field(default_factory=dict)
    crews: Dict[str, CrewModel] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")

    def resolved_crews(self) -> Dict[str, CrewModel]:
        return self.crew or self.crews

