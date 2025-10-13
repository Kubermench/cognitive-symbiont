"""Built-in templates for common Symbiont configurations.

This module provides pre-defined templates for typical use cases.
"""

from __future__ import annotations

from typing import Dict, Any, List
from .template_manager import Template, TemplateMetadata, TemplateVariable


# Built-in crew templates
CREW_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "basic_crew": {
        "metadata": {
            "name": "basic_crew",
            "description": "Basic crew with a single agent",
            "version": "1.0.0",
            "category": "basic",
            "tags": ["simple", "single-agent"],
            "variables": [
                TemplateVariable(
                    name="agent_name",
                    description="Name of the agent",
                    type="string",
                    required=True,
                    validation_pattern=r"^[a-zA-Z0-9_-]+$"
                ),
                TemplateVariable(
                    name="agent_role",
                    description="Role description for the agent",
                    type="string",
                    required=True
                ),
                TemplateVariable(
                    name="llm_provider",
                    description="LLM provider",
                    type="string",
                    required=True,
                    choices=["ollama", "openai", "anthropic"]
                ),
                TemplateVariable(
                    name="llm_model",
                    description="LLM model name",
                    type="string",
                    required=True
                )
            ]
        },
        "content": {
            "agents": {
                "${agent_name}": {
                    "role": "${agent_role}",
                    "llm": {
                        "provider": "${llm_provider}",
                        "model": "${llm_model}",
                        "temperature": 0.7,
                        "max_tokens": 1000
                    },
                    "tools": []
                }
            },
            "crew": {
                "main": {
                    "sequence": ["${agent_name}"]
                }
            }
        }
    },
    
    "multi_agent_crew": {
        "metadata": {
            "name": "multi_agent_crew",
            "description": "Crew with multiple agents working in sequence",
            "version": "1.0.0",
            "category": "advanced",
            "tags": ["multi-agent", "sequential"],
            "variables": [
                TemplateVariable(
                    name="crew_name",
                    description="Name of the crew",
                    type="string",
                    required=True
                ),
                TemplateVariable(
                    name="agents",
                    description="List of agent configurations",
                    type="list",
                    required=True
                )
            ]
        },
        "content": {
            "agents": {
                "researcher": {
                    "role": "Research specialist who gathers and analyzes information",
                    "llm": {
                        "provider": "${llm_provider}",
                        "model": "${llm_model}",
                        "temperature": 0.3
                    },
                    "tools": ["web_search", "file_reader"]
                },
                "analyst": {
                    "role": "Data analyst who processes and interprets research findings",
                    "llm": {
                        "provider": "${llm_provider}",
                        "model": "${llm_model}",
                        "temperature": 0.5
                    },
                    "tools": ["data_analyzer", "chart_generator"]
                },
                "writer": {
                    "role": "Technical writer who creates clear, comprehensive reports",
                    "llm": {
                        "provider": "${llm_provider}",
                        "model": "${llm_model}",
                        "temperature": 0.7
                    },
                    "tools": ["markdown_writer", "file_writer"]
                }
            },
            "crew": {
                "${crew_name}": {
                    "sequence": ["researcher", "analyst", "writer"],
                    "timeout": 1800,
                    "max_concurrent": 1
                }
            }
        }
    },
    
    "parallel_crew": {
        "metadata": {
            "name": "parallel_crew",
            "description": "Crew with agents working in parallel",
            "version": "1.0.0",
            "category": "advanced",
            "tags": ["parallel", "multi-agent"],
            "variables": [
                TemplateVariable(
                    name="crew_name",
                    description="Name of the crew",
                    type="string",
                    required=True
                ),
                TemplateVariable(
                    name="max_concurrent",
                    description="Maximum number of concurrent agents",
                    type="int",
                    required=False,
                    default=3
                )
            ]
        },
        "content": {
            "agents": {
                "task_1_agent": {
                    "role": "Handles task 1",
                    "llm": {
                        "provider": "${llm_provider}",
                        "model": "${llm_model}"
                    },
                    "tools": []
                },
                "task_2_agent": {
                    "role": "Handles task 2",
                    "llm": {
                        "provider": "${llm_provider}",
                        "model": "${llm_model}"
                    },
                    "tools": []
                },
                "task_3_agent": {
                    "role": "Handles task 3",
                    "llm": {
                        "provider": "${llm_provider}",
                        "model": "${llm_model}"
                    },
                    "tools": []
                }
            },
            "crew": {
                "${crew_name}": {
                    "sequence": ["task_1_agent", "task_2_agent", "task_3_agent"],
                    "max_concurrent": "${max_concurrent}",
                    "timeout": 900
                }
            }
        }
    }
}

# Built-in graph templates
GRAPH_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "simple_workflow": {
        "metadata": {
            "name": "simple_workflow",
            "description": "Simple linear workflow with error handling",
            "version": "1.0.0",
            "category": "basic",
            "tags": ["linear", "error-handling"],
            "variables": [
                TemplateVariable(
                    name="workflow_name",
                    description="Name of the workflow",
                    type="string",
                    required=True
                ),
                TemplateVariable(
                    name="start_agent",
                    description="Starting agent name",
                    type="string",
                    required=True
                ),
                TemplateVariable(
                    name="end_agent",
                    description="Ending agent name",
                    type="string",
                    required=True
                )
            ]
        },
        "content": {
            "graph": {
                "start": "${start_agent}",
                "nodes": {
                    "${start_agent}": {
                        "agent": "${start_agent}",
                        "next": "${end_agent}",
                        "on_failure": "error_handler",
                        "timeout": 300
                    },
                    "${end_agent}": {
                        "agent": "${end_agent}",
                        "timeout": 300
                    },
                    "error_handler": {
                        "agent": "error_handler",
                        "timeout": 60
                    }
                }
            },
            "crew_config": "./configs/crews.yaml"
        }
    },
    
    "conditional_workflow": {
        "metadata": {
            "name": "conditional_workflow",
            "description": "Workflow with conditional branching",
            "version": "1.0.0",
            "category": "advanced",
            "tags": ["conditional", "branching"],
            "variables": [
                TemplateVariable(
                    name="workflow_name",
                    description="Name of the workflow",
                    type="string",
                    required=True
                )
            ]
        },
        "content": {
            "graph": {
                "start": "decision_maker",
                "nodes": {
                    "decision_maker": {
                        "agent": "decision_maker",
                        "on_success": "success_handler",
                        "on_failure": "failure_handler",
                        "on_block": "timeout_handler",
                        "timeout": 300
                    },
                    "success_handler": {
                        "agent": "success_handler",
                        "timeout": 300
                    },
                    "failure_handler": {
                        "agent": "failure_handler",
                        "timeout": 300
                    },
                    "timeout_handler": {
                        "agent": "timeout_handler",
                        "timeout": 60
                    }
                }
            },
            "crew_config": "./configs/crews.yaml"
        }
    },
    
    "parallel_workflow": {
        "metadata": {
            "name": "parallel_workflow",
            "description": "Workflow with parallel execution",
            "version": "1.0.0",
            "category": "advanced",
            "tags": ["parallel", "concurrent"],
            "variables": [
                TemplateVariable(
                    name="workflow_name",
                    description="Name of the workflow",
                    type="string",
                    required=True
                ),
                TemplateVariable(
                    name="parallel_groups",
                    description="Number of parallel groups",
                    type="int",
                    required=False,
                    default=2
                )
            ]
        },
        "content": {
            "graph": {
                "start": "coordinator",
                "nodes": {
                    "coordinator": {
                        "agent": "coordinator",
                        "next": "aggregator",
                        "timeout": 300
                    },
                    "worker_1": {
                        "agent": "worker_1",
                        "timeout": 600
                    },
                    "worker_2": {
                        "agent": "worker_2",
                        "timeout": 600
                    },
                    "aggregator": {
                        "agent": "aggregator",
                        "timeout": 300
                    }
                },
                "parallel": [
                    ["worker_1", "worker_2"]
                ],
                "max_concurrent": 2
            },
            "crew_config": "./configs/crews.yaml"
        }
    }
}

# Combine all templates
BUILTIN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    **CREW_TEMPLATES,
    **GRAPH_TEMPLATES
}


def get_builtin_template(name: str) -> Dict[str, Any]:
    """Get a built-in template by name."""
    return BUILTIN_TEMPLATES.get(name, {})


def list_builtin_templates() -> List[str]:
    """List all built-in template names."""
    return list(BUILTIN_TEMPLATES.keys())


def get_templates_by_category(category: str) -> List[str]:
    """Get template names by category."""
    templates = []
    for name, template_data in BUILTIN_TEMPLATES.items():
        if template_data.get("metadata", {}).get("category") == category:
            templates.append(name)
    return templates