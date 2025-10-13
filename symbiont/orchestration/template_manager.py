"""Template management system for crew and graph configurations."""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from jinja2 import Environment, FileSystemLoader, Template, TemplateError

from .enhanced_schema import (
    CrewTemplate, GraphTemplate, TemplateMetadata, 
    CrewFileConfig, GraphFileConfig, ValidationLevel
)

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manages crew and graph templates."""
    
    def __init__(self, templates_dir: Path):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.crew_templates_dir = self.templates_dir / "crews"
        self.graph_templates_dir = self.templates_dir / "graphs"
        self.crew_templates_dir.mkdir(exist_ok=True)
        self.graph_templates_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.jinja_env.filters['to_yaml'] = self._to_yaml_filter
        self.jinja_env.filters['to_json'] = self._to_json_filter
    
    def _to_yaml_filter(self, value: Any) -> str:
        """Convert value to YAML string."""
        return yaml.dump(value, default_flow_style=False, allow_unicode=True)
    
    def _to_json_filter(self, value: Any) -> str:
        """Convert value to JSON string."""
        return json.dumps(value, indent=2, ensure_ascii=False)
    
    def create_crew_template(self, 
                           name: str,
                           description: str,
                           config: CrewFileConfig,
                           variables: Optional[Dict[str, Any]] = None,
                           author: Optional[str] = None,
                           tags: Optional[List[str]] = None,
                           category: str = "general") -> str:
        """Create a new crew template."""
        
        template_id = self._sanitize_name(name)
        template_dir = self.crew_templates_dir / template_id
        template_dir.mkdir(exist_ok=True)
        
        # Create metadata
        metadata = TemplateMetadata(
            name=name,
            description=description,
            version="1.0",
            author=author,
            tags=tags or [],
            category=category,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        # Create template
        template = CrewTemplate(
            metadata=metadata,
            config=config,
            variables=variables or {}
        )
        
        # Save template files
        self._save_crew_template(template, template_dir)
        
        logger.info("Created crew template: %s", template_id)
        return template_id
    
    def create_graph_template(self,
                            name: str,
                            description: str,
                            config: GraphFileConfig,
                            variables: Optional[Dict[str, Any]] = None,
                            author: Optional[str] = None,
                            tags: Optional[List[str]] = None,
                            category: str = "general") -> str:
        """Create a new graph template."""
        
        template_id = self._sanitize_name(name)
        template_dir = self.graph_templates_dir / template_id
        template_dir.mkdir(exist_ok=True)
        
        # Create metadata
        metadata = TemplateMetadata(
            name=name,
            description=description,
            version="1.0",
            author=author,
            tags=tags or [],
            category=category,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        # Create template
        template = GraphTemplate(
            metadata=metadata,
            config=config,
            variables=variables or {}
        )
        
        # Save template files
        self._save_graph_template(template, template_dir)
        
        logger.info("Created graph template: %s", template_id)
        return template_id
    
    def _save_crew_template(self, template: CrewTemplate, template_dir: Path):
        """Save crew template to directory."""
        # Save metadata
        metadata_file = template_dir / "metadata.json"
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(template.metadata.model_dump(), f, indent=2)
        
        # Save template YAML
        template_file = template_dir / "template.yaml"
        with template_file.open("w", encoding="utf-8") as f:
            yaml.dump(template.config.model_dump(), f, default_flow_style=False, allow_unicode=True)
        
        # Save variables
        variables_file = template_dir / "variables.json"
        with variables_file.open("w", encoding="utf-8") as f:
            json.dump(template.variables, f, indent=2)
        
        # Create Jinja2 template
        jinja_template = self._create_crew_jinja_template(template)
        jinja_file = template_dir / "template.j2"
        with jinja_file.open("w", encoding="utf-8") as f:
            jinja_file.write(jinja_template)
    
    def _save_graph_template(self, template: GraphTemplate, template_dir: Path):
        """Save graph template to directory."""
        # Save metadata
        metadata_file = template_dir / "metadata.json"
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(template.metadata.model_dump(), f, indent=2)
        
        # Save template YAML
        template_file = template_dir / "template.yaml"
        with template_file.open("w", encoding="utf-8") as f:
            yaml.dump(template.config.model_dump(), f, default_flow_style=False, allow_unicode=True)
        
        # Save variables
        variables_file = template_dir / "variables.json"
        with variables_file.open("w", encoding="utf-8") as f:
            json.dump(template.variables, f, indent=2)
        
        # Create Jinja2 template
        jinja_template = self._create_graph_jinja_template(template)
        jinja_file = template_dir / "template.j2"
        with jinja_file.open("w", encoding="utf-8") as f:
            jinja_file.write(jinja_template)
    
    def _create_crew_jinja_template(self, template: CrewTemplate) -> str:
        """Create Jinja2 template for crew configuration."""
        jinja_content = f"""# {template.metadata.name}
# {template.metadata.description}
# Version: {template.metadata.version}
# Author: {template.metadata.author or 'Unknown'}
# Category: {template.metadata.category}

version: "{{ version | default('1.0') }}"
name: "{{ name | default('{template.metadata.name}') }}"
description: "{{ description | default('{template.metadata.description}') }}"

agents:
{% for agent_name, agent in agents.items() %}
  {{{{ agent_name }}}}:
    role: "{{{{ agent.role }}}}"
    name: "{{{{ agent.name | default(agent_name) }}}}"
    description: "{{{{ agent.description | default('') }}}}"
    llm:
      provider: "{{{{ agent.llm.provider }}}}"
      model: "{{{{ agent.llm.model }}}}"
      api_key_env: "{{{{ agent.llm.api_key_env | default('') }}}}"
      timeout_seconds: {{{{ agent.llm.timeout_seconds | default(30) }}}}
      temperature: {{{{ agent.llm.temperature | default(0.7) }}}}
      retry_attempts: {{{{ agent.llm.retry_attempts | default(3) }}}}
    tools:
{% for tool in agent.tools %}
      - name: "{{{{ tool.name }}}}"
        enabled: {{{{ tool.enabled | default(true) }}}}
        config: {{{{ tool.config | to_yaml }}}}
{% endfor %}
    max_iterations: {{{{ agent.max_iterations | default(10) }}}}
    timeout_seconds: {{{{ agent.timeout_seconds | default(300) }}}}
    metadata: {{{{ agent.metadata | to_yaml }}}}
{% endfor %}

crews:
{% for crew_name, crew in crews.items() %}
  {{{{ crew_name }}}}:
    name: "{{{{ crew.name }}}}"
    description: "{{{{ crew.description | default('') }}}}"
    sequence: {{{{ crew.sequence | to_yaml }}}}
    parallel: {{{{ crew.parallel | default(false) }}}}
    timeout_seconds: {{{{ crew.timeout_seconds | default(1800) }}}}
    retry_attempts: {{{{ crew.retry_attempts | default(2) }}}}
    error_handling: "{{{{ crew.error_handling | default('stop') }}}}"
    metadata: {{{{ crew.metadata | to_yaml }}}}
{% endfor %}

metadata: {{{{ metadata | default({{}}) | to_yaml }}}}
"""
        return jinja_content
    
    def _create_graph_jinja_template(self, template: GraphTemplate) -> str:
        """Create Jinja2 template for graph configuration."""
        jinja_content = f"""# {template.metadata.name}
# {template.metadata.description}
# Version: {template.metadata.version}
# Author: {template.metadata.author or 'Unknown'}
# Category: {template.metadata.category}

version: "{{ version | default('1.0') }}"
name: "{{ name | default('{template.metadata.name}') }}"
description: "{{ description | default('{template.metadata.description}') }}"

crew_config: "{{{{ crew_config | default('') }}}}"

graph:
  name: "{{{{ graph.name }}}}"
  description: "{{{{ graph.description | default('') }}}}"
  start: "{{{{ graph.start }}}}"
  timeout_seconds: {{{{ graph.timeout_seconds | default(3600) }}}}
  retry_attempts: {{{{ graph.retry_attempts | default(1) }}}}
  error_handling: "{{{{ graph.error_handling | default('stop') }}}}"
  
  nodes:
{% for node_name, node in graph.nodes.items() %}
    {{{{ node_name }}}}:
      name: "{{{{ node.name }}}}"
      type: "{{{{ node.type }}}}"
{% if node.agent %}
      agent: "{{{{ node.agent }}}}"
{% endif %}
{% if node.condition %}
      condition: "{{{{ node.condition }}}}"
{% endif %}
{% if node.next %}
      next: "{{{{ node.next }}}}"
{% endif %}
{% if node.on_success %}
      on_success: "{{{{ node.on_success }}}}"
{% endif %}
{% if node.on_failure %}
      on_failure: "{{{{ node.on_failure }}}}"
{% endif %}
{% if node.on_block %}
      on_block: "{{{{ node.on_block }}}}"
{% endif %}
      timeout_seconds: {{{{ node.timeout_seconds | default('') }}}}
      retry_attempts: {{{{ node.retry_attempts | default(1) }}}}
      parallel: {{{{ node.parallel | default(false) }}}}
      metadata: {{{{ node.metadata | to_yaml }}}}
{% endfor %}
  
  parallel: {{{{ graph.parallel | to_yaml }}}}
  metadata: {{{{ graph.metadata | to_yaml }}}}

simulation: {{{{ simulation | default({{}}) | to_yaml }}}}
metadata: {{{{ metadata | default({{}}) | to_yaml }}}}
"""
        return jinja_content
    
    def list_templates(self, template_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available templates."""
        templates = []
        
        if template_type is None or template_type == "crew":
            templates.extend(self._list_crew_templates())
        
        if template_type is None or template_type == "graph":
            templates.extend(self._list_graph_templates())
        
        return sorted(templates, key=lambda t: t["name"])
    
    def _list_crew_templates(self) -> List[Dict[str, Any]]:
        """List crew templates."""
        templates = []
        
        for template_dir in self.crew_templates_dir.iterdir():
            if template_dir.is_dir():
                metadata_file = template_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with metadata_file.open("r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        metadata["type"] = "crew"
                        metadata["id"] = template_dir.name
                        templates.append(metadata)
                    except Exception as e:
                        logger.warning("Failed to load crew template %s: %s", template_dir.name, e)
        
        return templates
    
    def _list_graph_templates(self) -> List[Dict[str, Any]]:
        """List graph templates."""
        templates = []
        
        for template_dir in self.graph_templates_dir.iterdir():
            if template_dir.is_dir():
                metadata_file = template_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with metadata_file.open("r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        metadata["type"] = "graph"
                        metadata["id"] = template_dir.name
                        templates.append(metadata)
                    except Exception as e:
                        logger.warning("Failed to load graph template %s: %s", template_dir.name, e)
        
        return templates
    
    def get_template(self, template_id: str, template_type: str) -> Optional[Union[CrewTemplate, GraphTemplate]]:
        """Get a template by ID and type."""
        if template_type == "crew":
            return self._get_crew_template(template_id)
        elif template_type == "graph":
            return self._get_graph_template(template_id)
        else:
            return None
    
    def _get_crew_template(self, template_id: str) -> Optional[CrewTemplate]:
        """Get crew template by ID."""
        template_dir = self.crew_templates_dir / template_id
        
        metadata_file = template_dir / "metadata.json"
        template_file = template_dir / "template.yaml"
        variables_file = template_dir / "variables.json"
        
        if not all(f.exists() for f in [metadata_file, template_file, variables_file]):
            return None
        
        try:
            # Load metadata
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata_data = json.load(f)
            metadata = TemplateMetadata.model_validate(metadata_data)
            
            # Load config
            with template_file.open("r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            config = CrewFileConfig.model_validate(config_data)
            
            # Load variables
            with variables_file.open("r", encoding="utf-8") as f:
                variables = json.load(f)
            
            return CrewTemplate(
                metadata=metadata,
                config=config,
                variables=variables
            )
            
        except Exception as e:
            logger.error("Failed to load crew template %s: %s", template_id, e)
            return None
    
    def _get_graph_template(self, template_id: str) -> Optional[GraphTemplate]:
        """Get graph template by ID."""
        template_dir = self.graph_templates_dir / template_id
        
        metadata_file = template_dir / "metadata.json"
        template_file = template_dir / "template.yaml"
        variables_file = template_dir / "variables.json"
        
        if not all(f.exists() for f in [metadata_file, template_file, variables_file]):
            return None
        
        try:
            # Load metadata
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata_data = json.load(f)
            metadata = TemplateMetadata.model_validate(metadata_data)
            
            # Load config
            with template_file.open("r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            config = GraphFileConfig.model_validate(config_data)
            
            # Load variables
            with variables_file.open("r", encoding="utf-8") as f:
                variables = json.load(f)
            
            return GraphTemplate(
                metadata=metadata,
                config=config,
                variables=variables
            )
            
        except Exception as e:
            logger.error("Failed to load graph template %s: %s", template_id, e)
            return None
    
    def generate_from_template(self, 
                             template_id: str, 
                             template_type: str,
                             variables: Optional[Dict[str, Any]] = None,
                             output_path: Optional[Path] = None) -> Optional[Path]:
        """Generate configuration from template."""
        
        template = self.get_template(template_id, template_type)
        if not template:
            logger.error("Template not found: %s (%s)", template_id, template_type)
            return None
        
        # Merge template variables with provided variables
        merged_variables = {**template.variables, **(variables or {})}
        
        try:
            # Load Jinja2 template
            jinja_file = f"{template_type}s/{template_id}/template.j2"
            jinja_template = self.jinja_env.get_template(jinja_file)
            
            # Render template
            rendered_content = jinja_template.render(**merged_variables)
            
            # Determine output path
            if output_path is None:
                output_path = self.templates_dir / f"{template_id}_generated.yaml"
            
            # Write rendered content
            with output_path.open("w", encoding="utf-8") as f:
                f.write(rendered_content)
            
            logger.info("Generated configuration from template %s: %s", template_id, output_path)
            return output_path
            
        except TemplateError as e:
            logger.error("Template rendering error for %s: %s", template_id, e)
            return None
        except Exception as e:
            logger.error("Failed to generate from template %s: %s", template_id, e)
            return None
    
    def delete_template(self, template_id: str, template_type: str) -> bool:
        """Delete a template."""
        if template_type == "crew":
            template_dir = self.crew_templates_dir / template_id
        elif template_type == "graph":
            template_dir = self.graph_templates_dir / template_id
        else:
            return False
        
        if template_dir.exists():
            try:
                shutil.rmtree(template_dir)
                logger.info("Deleted template: %s (%s)", template_id, template_type)
                return True
            except Exception as e:
                logger.error("Failed to delete template %s: %s", template_id, e)
                return False
        
        return False
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use as directory name."""
        import re
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^\w\-_]', '_', name.lower())
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized or "unnamed_template"


def create_template_manager(templates_dir: Path) -> TemplateManager:
    """Create a template manager instance."""
    return TemplateManager(templates_dir)


def create_default_templates(template_manager: TemplateManager) -> None:
    """Create default templates."""
    
    # Basic crew template
    from .enhanced_schema import AgentConfig, LLMConfig, CrewConfig, CrewFileConfig
    
    basic_crew_config = CrewFileConfig(
        version="1.0",
        agents={
            "architect": AgentConfig(
                role="architect",
                description="System architect",
                llm=LLMConfig(provider="ollama", model="phi3:mini"),
                tools=[
                    {"name": "code_analysis", "enabled": True},
                    {"name": "design_review", "enabled": True}
                ]
            ),
            "developer": AgentConfig(
                role="developer", 
                description="Software developer",
                llm=LLMConfig(provider="ollama", model="phi3:mini"),
                tools=[
                    {"name": "code_generation", "enabled": True},
                    {"name": "testing", "enabled": True}
                ]
            )
        },
        crews={
            "development": CrewConfig(
                name="development",
                description="Basic development crew",
                sequence=["architect", "developer"],
                parallel=False
            )
        }
    )
    
    template_manager.create_crew_template(
        name="Basic Development Crew",
        description="A simple crew with architect and developer agents",
        config=basic_crew_config,
        variables={
            "version": "1.0",
            "name": "{{ name | default('development_crew') }}",
            "description": "{{ description | default('Development crew') }}"
        },
        author="Symbiont",
        tags=["basic", "development", "default"],
        category="development"
    )
    
    logger.info("Created default templates")