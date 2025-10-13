"""Template manager for Symbiont configurations.

This module provides template management, validation, and instantiation
capabilities for YAML configurations.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

from ..orchestration.schema import validate_crew_config, validate_graph_config


class TemplateVariable(BaseModel):
    """Represents a template variable."""
    name: str
    description: str
    type: str = "string"  # string, int, float, bool, list, dict
    default: Optional[Any] = None
    required: bool = True
    validation_pattern: Optional[str] = None
    choices: Optional[List[Any]] = None

    @property
    def is_optional(self) -> bool:
        return not self.required or self.default is not None


class TemplateMetadata(BaseModel):
    """Metadata for a template."""
    name: str
    description: str
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    category: str = "general"
    variables: List[TemplateVariable] = Field(default_factory=list)
    examples: List[Dict[str, Any]] = Field(default_factory=list)


class Template(BaseModel):
    """Represents a configuration template."""
    metadata: TemplateMetadata
    content: Dict[str, Any]
    template_type: str  # "crew" or "graph"
    
    def validate_content(self) -> bool:
        """Validate the template content."""
        try:
            if self.template_type == "crew":
                validate_crew_config(self.content)
            elif self.template_type == "graph":
                validate_graph_config(self.content)
            else:
                return False
            return True
        except Exception:
            return False
    
    def instantiate(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Instantiate the template with given variables."""
        # Validate required variables
        missing_vars = []
        for var in self.metadata.variables:
            if var.required and var.name not in variables:
                missing_vars.append(var.name)
        
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Apply variable substitutions
        content_str = json.dumps(self.content)
        
        for var in self.metadata.variables:
            var_name = var.name
            var_value = variables.get(var_name, var.default)
            
            if var_value is None:
                continue
            
            # Convert value to string for substitution
            if isinstance(var_value, (dict, list)):
                var_value_str = json.dumps(var_value)
            else:
                var_value_str = str(var_value)
            
            # Replace variable placeholders
            placeholder = f"${{{var_name}}}"
            content_str = content_str.replace(placeholder, var_value_str)
        
        # Parse back to dict
        return json.loads(content_str)


class TemplateManager:
    """Manages configuration templates."""
    
    def __init__(self, template_dirs: Optional[List[Union[str, Path]]] = None):
        self.template_dirs = template_dirs or [Path("./configs/templates")]
        self.templates: Dict[str, Template] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load templates from template directories."""
        for template_dir in self.template_dirs:
            template_dir = Path(template_dir)
            if not template_dir.exists():
                continue
            
            for template_file in template_dir.glob("*.yaml"):
                try:
                    template = self._load_template_file(template_file)
                    if template:
                        self.templates[template.metadata.name] = template
                except Exception as e:
                    print(f"Warning: Failed to load template {template_file}: {e}")
    
    def _load_template_file(self, file_path: Path) -> Optional[Template]:
        """Load a template from a YAML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not isinstance(data, dict):
                return None
            
            # Extract metadata
            metadata_data = data.get("metadata", {})
            metadata = TemplateMetadata(**metadata_data)
            
            # Determine template type
            template_type = "crew"
            if "graph" in data:
                template_type = "graph"
            elif "agents" in data or "crew" in data or "crews" in data:
                template_type = "crew"
            
            # Extract content (everything except metadata)
            content = {k: v for k, v in data.items() if k != "metadata"}
            
            template = Template(
                metadata=metadata,
                content=content,
                template_type=template_type
            )
            
            # Validate template
            if not template.validate_content():
                print(f"Warning: Template {metadata.name} has invalid content")
                return None
            
            return template
            
        except Exception as e:
            print(f"Error loading template {file_path}: {e}")
            return None
    
    def list_templates(self, category: Optional[str] = None) -> List[TemplateMetadata]:
        """List available templates."""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.metadata.category == category]
        
        return [t.metadata for t in templates]
    
    def get_template(self, name: str) -> Optional[Template]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def create_from_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Create a configuration from a template."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Instantiate template
        config = template.instantiate(variables)
        
        # Save to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        
        return config
    
    def validate_template(self, template_name: str) -> Dict[str, Any]:
        """Validate a template."""
        template = self.get_template(template_name)
        if not template:
            return {"valid": False, "error": "Template not found"}
        
        result = {
            "valid": True,
            "template_name": template_name,
            "template_type": template.template_type,
            "variables": len(template.metadata.variables),
            "required_variables": [v.name for v in template.metadata.variables if v.required],
            "optional_variables": [v.name for v in template.metadata.variables if not v.required]
        }
        
        # Validate content
        if not template.validate_content():
            result["valid"] = False
            result["error"] = "Template content is invalid"
        
        return result
    
    def get_template_variables(self, template_name: str) -> List[TemplateVariable]:
        """Get variables for a template."""
        template = self.get_template(template_name)
        if not template:
            return []
        
        return template.metadata.variables
    
    def search_templates(self, query: str) -> List[TemplateMetadata]:
        """Search templates by name, description, or tags."""
        query_lower = query.lower()
        results = []
        
        for template in self.templates.values():
            metadata = template.metadata
            
            # Search in name, description, and tags
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                results.append(metadata)
        
        return results
    
    def add_template(self, template: Template) -> None:
        """Add a template to the manager."""
        self.templates[template.metadata.name] = template
    
    def remove_template(self, name: str) -> bool:
        """Remove a template from the manager."""
        if name in self.templates:
            del self.templates[name]
            return True
        return False
    
    def export_template(self, name: str, output_path: Union[str, Path]) -> bool:
        """Export a template to a YAML file."""
        template = self.get_template(name)
        if not template:
            return False
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine metadata and content
        export_data = {
            "metadata": template.metadata.model_dump(),
            **template.content
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(export_data, f, default_flow_style=False, indent=2)
        
        return True
    
    def import_template(self, file_path: Union[str, Path]) -> bool:
        """Import a template from a YAML file."""
        try:
            template = self._load_template_file(Path(file_path))
            if template:
                self.templates[template.metadata.name] = template
                return True
        except Exception:
            pass
        
        return False


def create_template_from_config(
    config: Dict[str, Any],
    template_name: str,
    description: str,
    template_type: str = "crew",
    variables: Optional[List[TemplateVariable]] = None
) -> Template:
    """Create a template from an existing configuration."""
    metadata = TemplateMetadata(
        name=template_name,
        description=description,
        template_type=template_type,
        variables=variables or []
    )
    
    return Template(
        metadata=metadata,
        content=config,
        template_type=template_type
    )