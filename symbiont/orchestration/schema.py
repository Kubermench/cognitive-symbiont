from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class NodeModel(BaseModel):
    agent: str
    next: Optional[str] = None
    on_success: Optional[str] = None
    on_failure: Optional[str] = None
    on_block: Optional[str] = None
    timeout: Optional[int] = Field(None, ge=1, le=3600)  # 1 second to 1 hour
    retry_attempts: Optional[int] = Field(None, ge=0, le=10)
    retry_delay: Optional[float] = Field(None, ge=0.1, le=60.0)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("agent")
    @classmethod
    def validate_agent_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Agent name cannot be empty")
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Agent name must contain only alphanumeric characters, underscores, and hyphens")
        return v.strip()

    @field_validator("next", "on_success", "on_failure", "on_block")
    @classmethod
    def validate_node_references(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Node references must contain only alphanumeric characters, underscores, and hyphens")
        return v


class GraphSectionModel(BaseModel):
    start: str
    nodes: Dict[str, NodeModel]
    parallel: List[List[str]] = Field(default_factory=list)
    max_concurrent: Optional[int] = Field(None, ge=1, le=100)
    timeout: Optional[int] = Field(None, ge=1, le=86400)  # 1 second to 24 hours
    retry_policy: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")

    @field_validator("start")
    @classmethod
    def validate_start_node(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Start node cannot be empty")
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Start node must contain only alphanumeric characters, underscores, and hyphens")
        return v.strip()

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

    @model_validator(mode="after")
    def validate_graph_structure(self) -> "GraphSectionModel":
        # Validate that start node exists
        if self.start not in self.nodes:
            raise ValueError(f"Start node '{self.start}' not found in nodes")
        
        # Validate that all node references exist
        for node_name, node in self.nodes.items():
            for ref_field in ["next", "on_success", "on_failure", "on_block"]:
                ref_value = getattr(node, ref_field)
                if ref_value and ref_value not in self.nodes:
                    raise ValueError(f"Node '{node_name}' references non-existent node '{ref_value}' in {ref_field}")
        
        # Validate parallel groups
        for group in self.parallel:
            for node_name in group:
                if node_name not in self.nodes:
                    raise ValueError(f"Parallel group contains non-existent node '{node_name}'")
        
        return self


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


class LLMConfigModel(BaseModel):
    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=100000)
    timeout: Optional[int] = Field(None, ge=1, le=300)
    retry_attempts: Optional[int] = Field(None, ge=0, le=10)
    retry_delay: Optional[float] = Field(None, ge=0.1, le=60.0)
    fallback: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


class AgentModel(BaseModel):
    role: str
    llm: Union[Dict[str, Any], LLMConfigModel] = Field(default_factory=dict)
    cache: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    timeout: Optional[int] = Field(None, ge=1, le=3600)
    retry_attempts: Optional[int] = Field(None, ge=0, le=10)
    retry_delay: Optional[float] = Field(None, ge=0.1, le=60.0)
    memory: Optional[Dict[str, Any]] = Field(default_factory=dict)
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Role cannot be empty")
        return v.strip()

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: List[str]) -> List[str]:
        for tool in v:
            if not tool or not tool.strip():
                raise ValueError("Tool names cannot be empty")
            if not re.match(r"^[a-zA-Z0-9_-]+$", tool):
                raise ValueError("Tool names must contain only alphanumeric characters, underscores, and hyphens")
        return v

    @field_validator("llm", mode="before")
    @classmethod
    def validate_llm_config(cls, v: Any) -> Any:
        if isinstance(v, dict) and v:
            # Convert dict to LLMConfigModel for validation
            try:
                return LLMConfigModel(**v)
            except ValidationError:
                # If validation fails, return as-is for backward compatibility
                return v
        return v


class CrewModel(BaseModel):
    sequence: List[str] = Field(default_factory=list)
    roles: Optional[List[str]] = None
    max_concurrent: Optional[int] = Field(None, ge=1, le=100)
    timeout: Optional[int] = Field(None, ge=1, le=86400)
    retry_policy: Optional[Dict[str, Any]] = Field(default_factory=dict)
    memory: Optional[Dict[str, Any]] = Field(default_factory=dict)
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

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

    @field_validator("sequence", "roles")
    @classmethod
    def validate_role_names(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        for role in v:
            if not role or not role.strip():
                raise ValueError("Role names cannot be empty")
            if not re.match(r"^[a-zA-Z0-9_-]+$", role):
                raise ValueError("Role names must contain only alphanumeric characters, underscores, and hyphens")
        return v

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
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def validate_crew_structure(self) -> "CrewFileModel":
        # Validate that all roles referenced in crews exist in agents
        all_crews = self.resolved_crews()
        for crew_name, crew in all_crews.items():
            sequence = crew.resolved_sequence()
            for role in sequence:
                if role not in self.agents:
                    raise ValueError(f"Crew '{crew_name}' references non-existent role '{role}' in sequence")
        
        return self

    def resolved_crews(self) -> Dict[str, CrewModel]:
        return self.crew or self.crews


# Validation functions
def validate_crew_config(data: Dict[str, Any]) -> CrewFileModel:
    """Validate crew configuration data."""
    try:
        return CrewFileModel(**data)
    except ValidationError as e:
        raise ValueError(f"Crew configuration validation failed: {e}") from e


def validate_graph_config(data: Dict[str, Any]) -> GraphFileModel:
    """Validate graph configuration data."""
    try:
        return GraphFileModel(**data)
    except ValidationError as e:
        raise ValueError(f"Graph configuration validation failed: {e}") from e


def validate_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate a YAML file and return parsed data."""
    import yaml
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {file_path}: {e}") from e
    
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a dictionary at root level: {file_path}")
    
    return data


def validate_crew_yaml_file(file_path: Union[str, Path]) -> CrewFileModel:
    """Validate a crew YAML file."""
    data = validate_yaml_file(file_path)
    return validate_crew_config(data)


def validate_graph_yaml_file(file_path: Union[str, Path]) -> GraphFileModel:
    """Validate a graph YAML file."""
    data = validate_yaml_file(file_path)
    return validate_graph_config(data)


def get_validation_errors(data: Dict[str, Any], model_type: str = "crew") -> List[str]:
    """Get detailed validation errors for data."""
    errors = []
    
    try:
        if model_type == "crew":
            CrewFileModel(**data)
        elif model_type == "graph":
            GraphFileModel(**data)
        else:
            errors.append(f"Unknown model type: {model_type}")
    except ValidationError as e:
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            errors.append(f"{field_path}: {error['msg']}")
    
    return errors


def lint_yaml_file(file_path: Union[str, Path], model_type: str = "crew") -> Dict[str, Any]:
    """Lint a YAML file and return detailed results."""
    file_path = Path(file_path)
    
    result = {
        "file_path": str(file_path),
        "valid": False,
        "errors": [],
        "warnings": [],
        "suggestions": []
    }
    
    try:
        # Parse YAML
        data = validate_yaml_file(file_path)
        
        # Validate structure
        if model_type == "crew":
            validate_crew_config(data)
        elif model_type == "graph":
            validate_graph_config(data)
        else:
            result["errors"].append(f"Unknown model type: {model_type}")
            return result
        
        result["valid"] = True
        
        # Add suggestions for improvement
        if model_type == "crew":
            if not data.get("agents"):
                result["suggestions"].append("Consider adding agents to the configuration")
            if not data.get("crew") and not data.get("crews"):
                result["suggestions"].append("Consider adding crew definitions")
        
        elif model_type == "graph":
            if not data.get("graph", {}).get("nodes"):
                result["suggestions"].append("Consider adding nodes to the graph")
            if not data.get("graph", {}).get("start"):
                result["suggestions"].append("Consider specifying a start node")
    
    except Exception as e:
        result["errors"].append(str(e))
    
    return result

