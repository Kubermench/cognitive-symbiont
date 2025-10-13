"""Property-based testing for YAML parsing and configuration validation."""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union

import pytest
import yaml
from hypothesis import HealthCheck, given, settings, strategies as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.orchestration.graph import GraphSpec
from symbiont.agents.registry import AgentRegistry
from symbiont.orchestration.schema import CrewConfig, GraphConfig


# YAML fuzzing strategies
def yaml_value_strategy():
    """Generate arbitrary YAML values."""
    return st.recursive(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(min_size=0, max_size=100),
        ),
        lambda children: st.one_of(
            st.lists(children, min_size=0, max_size=10),
            st.dictionaries(
                st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "_"))),
                children,
                min_size=0,
                max_size=10,
            ),
        ),
        max_leaves=50,
    )


def malformed_yaml_strategy():
    """Generate malformed YAML strings."""
    return st.one_of(
        # Invalid YAML syntax
        st.text(min_size=1, max_size=200).filter(lambda x: not _is_valid_yaml(x)),
        # Binary data
        st.binary(min_size=1, max_size=100).map(lambda b: b.decode("utf-8", errors="ignore")),
        # JSON instead of YAML
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.text(min_size=0, max_size=50),
            min_size=1,
            max_size=5,
        ).map(json.dumps),
        # Circular references (will fail to serialize)
        st.builds(lambda: {"a": None}).map(lambda d: {**d, "a": d}),
    )


def _is_valid_yaml(text: str) -> bool:
    """Check if text is valid YAML."""
    try:
        yaml.safe_load(text)
        return True
    except Exception:
        return False


def crew_yaml_strategy():
    """Generate valid crew YAML configurations."""
    return st.builds(
        lambda agents, crew: {
            "agents": agents,
            "crew": crew,
        },
        agents=st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "_"))),
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(
                    st.text(max_size=100),
                    st.integers(),
                    st.booleans(),
                    st.lists(st.text(max_size=50), max_size=5),
                ),
                min_size=1,
                max_size=10,
            ),
            min_size=1,
            max_size=5,
        ),
        crew=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(
                st.text(max_size=100),
                st.lists(st.text(max_size=50), max_size=5),
                st.dictionaries(
                    st.text(min_size=1, max_size=20),
                    st.text(max_size=100),
                    min_size=1,
                    max_size=5,
                ),
            ),
            min_size=1,
            max_size=5,
        ),
    )


def graph_yaml_strategy():
    """Generate valid graph YAML configurations."""
    return st.builds(
        lambda crew_config, graph: {
            "crew_config": crew_config,
            "graph": graph,
        },
        crew_config=st.one_of(
            st.just("./crew_template.yaml"),
            st.text(min_size=1, max_size=100),
        ),
        graph=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(
                st.text(max_size=100),
                st.dictionaries(
                    st.text(min_size=1, max_size=20),
                    st.one_of(
                        st.text(max_size=100),
                        st.lists(st.text(max_size=50), max_size=5),
                        st.dictionaries(
                            st.text(min_size=1, max_size=20),
                            st.text(max_size=100),
                            min_size=1,
                            max_size=5,
                        ),
                    ),
                    min_size=1,
                    max_size=10,
                ),
            ),
            min_size=1,
            max_size=10,
        ),
    )


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(crew_data=crew_yaml_strategy())
def test_crew_yaml_parsing_robustness(tmp_path, crew_data):
    """Test that crew YAML parsing handles various valid configurations."""
    crew_yaml = tmp_path / "crew.yaml"
    crew_yaml.write_text(yaml.dump(crew_data), encoding="utf-8")
    
    try:
        registry = AgentRegistry.from_yaml(crew_yaml)
        assert registry is not None
        # Basic validation that the registry was created successfully
        assert hasattr(registry, 'agents') or hasattr(registry, 'crews')
    except Exception as exc:
        # Some configurations may be invalid, but should fail gracefully
        assert isinstance(exc, (ValueError, TypeError, KeyError, yaml.YAMLError))


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(graph_data=graph_yaml_strategy())
def test_graph_yaml_parsing_robustness(tmp_path, graph_data):
    """Test that graph YAML parsing handles various valid configurations."""
    graph_yaml = tmp_path / "graph.yaml"
    graph_yaml.write_text(yaml.dump(graph_data), encoding="utf-8")
    
    try:
        graph_spec = GraphSpec.from_yaml(graph_yaml)
        assert graph_spec is not None
        # Basic validation that the graph was created successfully
        assert hasattr(graph_spec, 'graph') or hasattr(graph_spec, 'crew_config')
    except Exception as exc:
        # Some configurations may be invalid, but should fail gracefully
        assert isinstance(exc, (ValueError, TypeError, KeyError, yaml.YAMLError))


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(malformed_yaml=malformed_yaml_strategy())
def test_yaml_parsing_malformed_input(tmp_path, malformed_yaml):
    """Test that YAML parsing fails gracefully on malformed input."""
    yaml_file = tmp_path / "malformed.yaml"
    yaml_file.write_text(str(malformed_yaml), encoding="utf-8")
    
    # Should either parse successfully or raise a known exception type
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        # If it parsed, it should be a valid data structure
        assert isinstance(data, (dict, list, str, int, float, bool, type(None)))
    except Exception as exc:
        # Should be a known YAML parsing error
        assert isinstance(exc, (yaml.YAMLError, UnicodeDecodeError, ValueError))


@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(yaml_data=yaml_value_strategy())
def test_yaml_roundtrip_consistency(tmp_path, yaml_data):
    """Test that YAML data can be written and read back consistently."""
    yaml_file = tmp_path / "roundtrip.yaml"
    
    try:
        # Write data to YAML
        yaml_file.write_text(yaml.dump(yaml_data), encoding="utf-8")
        
        # Read it back
        with open(yaml_file, 'r', encoding='utf-8') as f:
            loaded_data = yaml.safe_load(f)
        
        # Should be equivalent (allowing for some type conversions)
        assert _deep_compare(yaml_data, loaded_data)
    except Exception as exc:
        # Some data may not be serializable to YAML
        assert isinstance(exc, (yaml.YAMLError, TypeError, ValueError))


def _deep_compare(obj1: Any, obj2: Any) -> bool:
    """Compare two objects deeply, handling type conversions."""
    if type(obj1) != type(obj2):
        # Allow some type conversions (e.g., int/float)
        if isinstance(obj1, (int, float)) and isinstance(obj2, (int, float)):
            return abs(obj1 - obj2) < 1e-10
        return False
    
    if isinstance(obj1, dict):
        if set(obj1.keys()) != set(obj2.keys()):
            return False
        return all(_deep_compare(obj1[k], obj2[k]) for k in obj1.keys())
    
    if isinstance(obj1, list):
        if len(obj1) != len(obj2):
            return False
        return all(_deep_compare(a, b) for a, b in zip(obj1, obj2))
    
    return obj1 == obj2


@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    crew_data=st.one_of(
        crew_yaml_strategy(),
        malformed_yaml_strategy().map(lambda x: {"agents": x, "crew": x}),
    )
)
def test_crew_config_validation_edge_cases(tmp_path, crew_data):
    """Test crew configuration validation with edge cases."""
    crew_yaml = tmp_path / "crew.yaml"
    
    try:
        yaml_content = yaml.dump(crew_data) if isinstance(crew_data, dict) else str(crew_data)
        crew_yaml.write_text(yaml_content, encoding="utf-8")
        
        registry = AgentRegistry.from_yaml(crew_yaml)
        assert registry is not None
    except Exception as exc:
        # Should fail with a known exception type
        assert isinstance(exc, (ValueError, TypeError, KeyError, yaml.YAMLError, AttributeError))


@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    graph_data=st.one_of(
        graph_yaml_strategy(),
        malformed_yaml_strategy().map(lambda x: {"crew_config": x, "graph": x}),
    )
)
def test_graph_config_validation_edge_cases(tmp_path, graph_data):
    """Test graph configuration validation with edge cases."""
    graph_yaml = tmp_path / "graph.yaml"
    
    try:
        yaml_content = yaml.dump(graph_data) if isinstance(graph_data, dict) else str(graph_data)
        graph_yaml.write_text(yaml_content, encoding="utf-8")
        
        graph_spec = GraphSpec.from_yaml(graph_yaml)
        assert graph_spec is not None
    except Exception as exc:
        # Should fail with a known exception type
        assert isinstance(exc, (ValueError, TypeError, KeyError, yaml.YAMLError, AttributeError))


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    file_size=st.integers(min_value=1, max_value=10000),
    content_type=st.sampled_from(["yaml", "json", "text", "binary"]),
)
def test_large_file_handling(tmp_path, file_size, content_type):
    """Test handling of large files with different content types."""
    test_file = tmp_path / f"large_file.{content_type}"
    
    if content_type == "yaml":
        # Generate large YAML structure
        large_data = {"items": [{"id": i, "data": f"item_{i}"} for i in range(file_size // 10)]}
        content = yaml.dump(large_data)
    elif content_type == "json":
        large_data = {"items": [{"id": i, "data": f"item_{i}"} for i in range(file_size // 10)]}
        content = json.dumps(large_data)
    elif content_type == "text":
        content = "x" * file_size
    else:  # binary
        content = b"x" * file_size
    
    test_file.write_bytes(content.encode() if isinstance(content, str) else content)
    
    try:
        if content_type == "yaml":
            with open(test_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            assert data is not None
        elif content_type == "json":
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            assert data is not None
    except Exception as exc:
        # Large files may cause memory issues, but should fail gracefully
        assert isinstance(exc, (MemoryError, yaml.YAMLError, json.JSONDecodeError, UnicodeDecodeError))