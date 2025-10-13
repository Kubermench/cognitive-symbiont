"""Enhanced Pydantic schemas for comprehensive validation."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel, ConfigDict, Field, ValidationError, field_validator, 
    model_validator, HttpUrl, EmailStr, constr
)


class NodeType(str, Enum):
    """Types of graph nodes."""
    AGENT = "agent"
    DECISION = "decision"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    LOOP = "loop"
    TERMINAL = "terminal"


class AgentRole(str, Enum):
    """Standard agent roles."""
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    TESTER = "tester"
    ANALYST = "analyst"
    COORDINATOR = "coordinator"
    MONITOR = "monitor"
    RESEARCHER = "researcher"
    WRITER = "writer"
    CUSTOM = "custom"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    CMD = "cmd"
    CUSTOM = "custom"


class CacheType(str, Enum):
    """Cache types."""
    MEMORY = "memory"
    REDIS = "redis"
    SQLITE = "sqlite"
    FILE = "file"


class ValidationLevel(str, Enum):
    """Validation levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


# Base validation functions
def validate_node_name(name: str) -> str:
    """Validate node name format."""
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
        raise ValueError("Node name must start with letter and contain only letters, numbers, underscores, and hyphens")
    return name


def validate_agent_name(name: str) -> str:
    """Validate agent name format."""
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
        raise ValueError("Agent name must start with letter and contain only letters, numbers, underscores, and hyphens")
    return name


def validate_crew_name(name: str) -> str:
    """Validate crew name format."""
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
        raise ValueError("Crew name must start with letter and contain only letters, numbers, underscores, and hyphens")
    return name


# LLM Configuration Models
class LLMConfig(BaseModel):
    """LLM configuration with validation."""
    provider: LLMProvider = Field(..., description="LLM provider")
    model: str = Field(..., min_length=1, description="Model name")
    api_key_env: Optional[str] = Field(None, description="Environment variable for API key")
    timeout_seconds: int = Field(30, ge=1, le=300, description="Request timeout in seconds")
    max_tokens: Optional[int] = Field(None, ge=1, le=100000, description="Maximum tokens per request")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    retry_attempts: int = Field(3, ge=1, le=10, description="Number of retry attempts")
    retry_initial_delay: float = Field(0.5, ge=0.1, le=10.0, description="Initial retry delay in seconds")
    retry_max_delay: float = Field(20.0, ge=1.0, le=300.0, description="Maximum retry delay in seconds")
    retry_multiplier: float = Field(2.0, ge=1.0, le=5.0, description="Retry delay multiplier")
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator("model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class CacheConfig(BaseModel):
    """Cache configuration."""
    type: CacheType = Field(..., description="Cache type")
    path: Optional[str] = Field(None, description="Cache path (for file/sqlite)")
    host: Optional[str] = Field(None, description="Cache host (for redis)")
    port: Optional[int] = Field(None, ge=1, le=65535, description="Cache port (for redis)")
    db: Optional[int] = Field(None, ge=0, le=15, description="Cache database (for redis)")
    ttl_seconds: int = Field(3600, ge=60, le=86400, description="Default TTL in seconds")
    max_size: Optional[int] = Field(None, ge=1, description="Maximum cache size")
    
    model_config = ConfigDict(extra="forbid")
    
    @model_validator(mode="after")
    def validate_cache_config(self) -> "CacheConfig":
        if self.type in [CacheType.REDIS] and not self.host:
            raise ValueError("Redis cache requires host configuration")
        if self.type in [CacheType.FILE, CacheType.SQLITE] and not self.path:
            raise ValueError("File/SQLite cache requires path configuration")
        return self


# Tool Configuration Models
class ToolConfig(BaseModel):
    """Tool configuration."""
    name: str = Field(..., min_length=1, description="Tool name")
    enabled: bool = Field(True, description="Whether tool is enabled")
    config: Dict[str, Any] = Field(default_factory=dict, description="Tool-specific configuration")
    timeout_seconds: Optional[int] = Field(None, ge=1, le=300, description="Tool timeout")
    retry_attempts: int = Field(1, ge=0, le=5, description="Tool retry attempts")
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator("name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', v):
            raise ValueError("Tool name must start with letter and contain only letters, numbers, underscores, and hyphens")
        return v


# Agent Models
class AgentConfig(BaseModel):
    """Enhanced agent configuration."""
    role: Union[AgentRole, str] = Field(..., description="Agent role")
    name: Optional[str] = Field(None, description="Agent name (auto-generated if not provided)")
    description: Optional[str] = Field(None, description="Agent description")
    llm: LLMConfig = Field(..., description="LLM configuration")
    cache: Optional[CacheConfig] = Field(None, description="Cache configuration")
    tools: List[ToolConfig] = Field(default_factory=list, description="Available tools")
    max_iterations: int = Field(10, ge=1, le=100, description="Maximum iterations per task")
    timeout_seconds: int = Field(300, ge=30, le=3600, description="Agent timeout in seconds")
    memory_limit: Optional[int] = Field(None, ge=1000, description="Memory limit in tokens")
    temperature_override: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature override")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator("role")
    @classmethod
    def validate_role(cls, v: Union[AgentRole, str]) -> str:
        if isinstance(v, AgentRole):
            return v.value
        if not v.strip():
            raise ValueError("Role cannot be empty")
        return v.strip()
    
    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return validate_agent_name(v)
        return v


# Crew Models
class CrewConfig(BaseModel):
    """Enhanced crew configuration."""
    name: str = Field(..., description="Crew name")
    description: Optional[str] = Field(None, description="Crew description")
    sequence: List[str] = Field(default_factory=list, description="Execution sequence")
    roles: Optional[List[str]] = Field(None, description="Required roles")
    parallel: bool = Field(False, description="Whether to run agents in parallel")
    max_parallel: Optional[int] = Field(None, ge=1, le=10, description="Maximum parallel agents")
    timeout_seconds: int = Field(1800, ge=60, le=7200, description="Crew timeout in seconds")
    retry_attempts: int = Field(2, ge=0, le=5, description="Crew retry attempts")
    error_handling: Literal["stop", "continue", "retry"] = Field("stop", description="Error handling strategy")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return validate_crew_name(v)
    
    @field_validator("sequence", mode="before")
    @classmethod
    def validate_sequence(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            raise ValueError("Sequence must be a list of agent names")
        if not isinstance(v, list):
            raise ValueError("Sequence must be a list")
        return [str(item) for item in v]
    
    @field_validator("roles", mode="before")
    @classmethod
    def validate_roles(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return None
        if isinstance(v, str):
            raise ValueError("Roles must be a list of role names")
        if not isinstance(v, list):
            raise ValueError("Roles must be a list")
        return [str(item) for item in v]
    
    @model_validator(mode="after")
    def validate_crew_config(self) -> "CrewConfig":
        if not self.sequence and not self.roles:
            raise ValueError("Crew must have either sequence or roles defined")
        if self.parallel and self.max_parallel and len(self.sequence) > self.max_parallel:
            raise ValueError("Sequence length exceeds max_parallel limit")
        return self


# Graph Models
class NodeConfig(BaseModel):
    """Enhanced node configuration."""
    name: str = Field(..., description="Node name")
    type: NodeType = Field(NodeType.AGENT, description="Node type")
    agent: Optional[str] = Field(None, description="Agent name (for agent nodes)")
    condition: Optional[str] = Field(None, description="Condition expression (for condition nodes)")
    next: Optional[str] = Field(None, description="Next node on success")
    on_success: Optional[str] = Field(None, description="Next node on success")
    on_failure: Optional[str] = Field(None, description="Next node on failure")
    on_block: Optional[str] = Field(None, description="Next node on block")
    timeout_seconds: Optional[int] = Field(None, ge=1, le=3600, description="Node timeout")
    retry_attempts: int = Field(1, ge=0, le=5, description="Node retry attempts")
    parallel: bool = Field(False, description="Whether to run in parallel")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return validate_node_name(v)
    
    @model_validator(mode="after")
    def validate_node_config(self) -> "NodeConfig":
        if self.type == NodeType.AGENT and not self.agent:
            raise ValueError("Agent nodes must specify an agent")
        if self.type == NodeType.CONDITION and not self.condition:
            raise ValueError("Condition nodes must specify a condition")
        return self


class GraphConfig(BaseModel):
    """Enhanced graph configuration."""
    name: str = Field(..., description="Graph name")
    description: Optional[str] = Field(None, description="Graph description")
    start: str = Field(..., description="Start node name")
    nodes: Dict[str, NodeConfig] = Field(..., description="Graph nodes")
    parallel: List[List[str]] = Field(default_factory=list, description="Parallel execution groups")
    timeout_seconds: int = Field(3600, ge=60, le=14400, description="Graph timeout in seconds")
    retry_attempts: int = Field(1, ge=0, le=3, description="Graph retry attempts")
    error_handling: Literal["stop", "continue", "retry"] = Field("stop", description="Error handling strategy")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return validate_crew_name(v)  # Reuse crew name validation
    
    @field_validator("parallel", mode="before")
    @classmethod
    def validate_parallel(cls, v: Any) -> List[List[str]]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("Parallel must be a list of lists")
        for group in v:
            if not isinstance(group, list):
                raise ValueError("Parallel entries must be lists of node names")
            if not all(isinstance(name, str) for name in group):
                raise ValueError("Parallel group entries must be strings")
        return v
    
    @model_validator(mode="after")
    def validate_graph_config(self) -> "GraphConfig":
        # Validate start node exists
        if self.start not in self.nodes:
            raise ValueError(f"Start node '{self.start}' not found in nodes")
        
        # Validate all referenced nodes exist
        all_referenced = set()
        for node in self.nodes.values():
            if node.next:
                all_referenced.add(node.next)
            if node.on_success:
                all_referenced.add(node.on_success)
            if node.on_failure:
                all_referenced.add(node.on_failure)
            if node.on_block:
                all_referenced.add(node.on_block)
        
        for group in self.parallel:
            all_referenced.update(group)
        
        missing_nodes = all_referenced - set(self.nodes.keys())
        if missing_nodes:
            raise ValueError(f"Referenced nodes not found: {missing_nodes}")
        
        return self


# File Models
class CrewFileConfig(BaseModel):
    """Enhanced crew file configuration."""
    version: str = Field("1.0", description="Configuration version")
    name: Optional[str] = Field(None, description="File name")
    description: Optional[str] = Field(None, description="File description")
    agents: Dict[str, AgentConfig] = Field(default_factory=dict, description="Agent configurations")
    crews: Dict[str, CrewConfig] = Field(default_factory=dict, description="Crew configurations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(extra="forbid")
    
    @model_validator(mode="after")
    def validate_crew_file(self) -> "CrewFileConfig":
        # Validate that all crew agents exist
        for crew_name, crew in self.crews.items():
            for agent_name in crew.sequence:
                if agent_name not in self.agents:
                    raise ValueError(f"Crew '{crew_name}' references undefined agent '{agent_name}'")
        
        return self


class GraphFileConfig(BaseModel):
    """Enhanced graph file configuration."""
    version: str = Field("1.0", description="Configuration version")
    name: Optional[str] = Field(None, description="File name")
    description: Optional[str] = Field(None, description="File description")
    graph: GraphConfig = Field(..., description="Graph configuration")
    crew_config: Optional[str] = Field(None, description="Crew configuration file path")
    crews: Optional[str] = Field(None, description="Crews configuration file path")
    simulation: Optional[Dict[str, Any]] = Field(None, description="Simulation configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(extra="forbid")
    
    def require_crew_path(self) -> str:
        """Get the required crew configuration path."""
        if self.crew_config:
            return self.crew_config
        if self.crews:
            return self.crews
        raise ValueError("Graph spec missing 'crew_config' or 'crews'")


# Validation Functions
def validate_crew_file(data: Dict[str, Any], validation_level: ValidationLevel = ValidationLevel.MODERATE) -> CrewFileConfig:
    """Validate crew file data."""
    try:
        return CrewFileConfig.model_validate(data)
    except ValidationError as e:
        if validation_level == ValidationLevel.STRICT:
            raise
        elif validation_level == ValidationLevel.MODERATE:
            # Log warnings but continue
            logger.warning("Crew file validation warnings: %s", e)
            return CrewFileConfig.model_validate(data, strict=False)
        else:  # PERMISSIVE
            # Try to fix common issues
            fixed_data = _fix_common_issues(data)
            return CrewFileConfig.model_validate(fixed_data, strict=False)


def validate_graph_file(data: Dict[str, Any], validation_level: ValidationLevel = ValidationLevel.MODERATE) -> GraphFileConfig:
    """Validate graph file data."""
    try:
        return GraphFileConfig.model_validate(data)
    except ValidationError as e:
        if validation_level == ValidationLevel.STRICT:
            raise
        elif validation_level == ValidationLevel.MODERATE:
            logger.warning("Graph file validation warnings: %s", e)
            return GraphFileConfig.model_validate(data, strict=False)
        else:  # PERMISSIVE
            fixed_data = _fix_common_issues(data)
            return GraphFileConfig.model_validate(fixed_data, strict=False)


def _fix_common_issues(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fix common configuration issues."""
    fixed = data.copy()
    
    # Fix missing version
    if "version" not in fixed:
        fixed["version"] = "1.0"
    
    # Fix string sequences
    if "crews" in fixed:
        for crew_name, crew_data in fixed["crews"].items():
            if isinstance(crew_data, dict):
                if "sequence" in crew_data and isinstance(crew_data["sequence"], str):
                    crew_data["sequence"] = [crew_data["sequence"]]
                if "roles" in crew_data and isinstance(crew_data["roles"], str):
                    crew_data["roles"] = [crew_data["roles"]]
    
    return fixed


# Template Models
class TemplateMetadata(BaseModel):
    """Template metadata."""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    version: str = Field("1.0", description="Template version")
    author: Optional[str] = Field(None, description="Template author")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    category: str = Field("general", description="Template category")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    
    model_config = ConfigDict(extra="forbid")


class CrewTemplate(BaseModel):
    """Crew template."""
    metadata: TemplateMetadata = Field(..., description="Template metadata")
    config: CrewFileConfig = Field(..., description="Crew configuration")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables")
    
    model_config = ConfigDict(extra="forbid")


class GraphTemplate(BaseModel):
    """Graph template."""
    metadata: TemplateMetadata = Field(..., description="Template metadata")
    config: GraphFileConfig = Field(..., description="Graph configuration")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables")
    
    model_config = ConfigDict(extra="forbid")