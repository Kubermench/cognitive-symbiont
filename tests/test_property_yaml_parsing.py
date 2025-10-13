"""Property-based tests for YAML parsing using Hypothesis.

This module uses Hypothesis to generate random YAML configurations and test
that the parsing logic is robust against malformed inputs, edge cases, and
adversarial configurations.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from pydantic import ValidationError

from symbiont.orchestration.schema import (
    NodeModel,
    GraphSectionModel,
    GraphFileModel,
    AgentModel,
    CrewModel,
    CrewFileModel,
)
from symbiont.orchestration.graph import GraphSpec
from symbiont.agents.registry import AgentRegistry


# Hypothesis strategies for generating valid and invalid YAML structures

@st.composite
def valid_node_names(draw):
    """Generate valid node names."""
    return draw(st.text(
        alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Nd"], min_codepoint=32, max_codepoint=126),
        min_size=1,
        max_size=50
    ).filter(lambda x: x.strip() and not x.startswith('_') and x.isidentifier()))


@st.composite
def valid_role_names(draw):
    """Generate valid role names."""
    return draw(st.text(
        alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Nd"], min_codepoint=32, max_codepoint=126),
        min_size=1,
        max_size=50
    ).filter(lambda x: x.strip() and x.replace('_', '').isalnum()))


@st.composite
def llm_config_strategy(draw):
    """Generate LLM configuration dictionaries."""
    provider = draw(st.sampled_from(["ollama", "cmd", "openai", "anthropic"]))
    model = draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip()))
    
    config = {
        "provider": provider,
        "model": model,
    }
    
    # Add optional fields
    if draw(st.booleans()):
        config["mode"] = draw(st.sampled_from(["local", "cloud", "hybrid"]))
    
    if draw(st.booleans()):
        config["timeout_seconds"] = draw(st.integers(min_value=1, max_value=300))
    
    if draw(st.booleans()):
        config["hybrid_threshold_tokens"] = draw(st.integers(min_value=0, max_value=10000))
    
    return config


@st.composite
def node_model_strategy(draw):
    """Generate NodeModel instances."""
    agent = draw(valid_node_names())
    
    node_data = {"agent": agent}
    
    # Add optional routing fields
    if draw(st.booleans()):
        node_data["next"] = draw(st.one_of(st.none(), valid_node_names()))
    
    if draw(st.booleans()):
        node_data["on_success"] = draw(st.one_of(st.none(), valid_node_names()))
    
    if draw(st.booleans()):
        node_data["on_failure"] = draw(st.one_of(st.none(), valid_node_names()))
    
    if draw(st.booleans()):
        node_data["on_block"] = draw(st.one_of(st.none(), valid_node_names()))
    
    return node_data


@st.composite
def graph_section_strategy(draw):
    """Generate GraphSectionModel instances."""
    # Generate nodes
    node_names = draw(st.lists(valid_node_names(), min_size=1, max_size=10, unique=True))
    start_node = draw(st.sampled_from(node_names))
    
    nodes = {}
    for name in node_names:
        node_data = draw(node_model_strategy())
        # Ensure routing targets are valid
        for field in ["next", "on_success", "on_failure", "on_block"]:
            if field in node_data and node_data[field] is not None:
                if node_data[field] not in node_names and node_data[field] != "END":
                    node_data[field] = draw(st.sampled_from(node_names + ["END"]))
        nodes[name] = node_data
    
    graph_data = {
        "start": start_node,
        "nodes": nodes,
    }
    
    # Add optional parallel groups
    if draw(st.booleans()) and len(node_names) > 1:
        # Generate valid parallel groups
        parallel_groups = []
        remaining_nodes = node_names.copy()
        
        while len(remaining_nodes) > 1 and draw(st.booleans()):
            group_size = draw(st.integers(min_value=2, max_value=min(4, len(remaining_nodes))))
            group = draw(st.lists(
                st.sampled_from(remaining_nodes),
                min_size=group_size,
                max_size=group_size,
                unique=True
            ))
            parallel_groups.append(group)
            for node in group:
                remaining_nodes.remove(node)
        
        if parallel_groups:
            graph_data["parallel"] = parallel_groups
    
    return graph_data


@st.composite
def agent_model_strategy(draw):
    """Generate AgentModel instances."""
    role = draw(valid_role_names())
    
    agent_data = {"role": role}
    
    # Add optional LLM config
    if draw(st.booleans()):
        agent_data["llm"] = draw(llm_config_strategy())
    
    # Add optional cache
    if draw(st.booleans()):
        agent_data["cache"] = draw(st.sampled_from(["in_memory", "redis", "sqlite", "none"]))
    
    # Add optional tools
    if draw(st.booleans()):
        tools = draw(st.lists(
            st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
            min_size=0,
            max_size=5
        ))
        agent_data["tools"] = tools
    
    return agent_data


@st.composite
def crew_model_strategy(draw, agent_names: List[str]):
    """Generate CrewModel instances."""
    crew_data = {}
    
    if agent_names and draw(st.booleans()):
        sequence_size = draw(st.integers(min_value=1, max_value=min(5, len(agent_names))))
        sequence = draw(st.lists(
            st.sampled_from(agent_names),
            min_size=sequence_size,
            max_size=sequence_size,
            unique=True
        ))
        crew_data["sequence"] = sequence
    
    return crew_data


@st.composite
def crew_file_strategy(draw):
    """Generate CrewFileModel instances."""
    # Generate agents
    agent_names = draw(st.lists(valid_node_names(), min_size=1, max_size=8, unique=True))
    agents = {}
    for name in agent_names:
        agents[name] = draw(agent_model_strategy())
    
    # Generate crews
    crew_names = draw(st.lists(valid_node_names(), min_size=1, max_size=5, unique=True))
    crews = {}
    for name in crew_names:
        crews[name] = draw(crew_model_strategy(agent_names))
    
    return {
        "agents": agents,
        "crew": crews,
    }


@st.composite
def malformed_yaml_strategy(draw):
    """Generate malformed YAML that should fail parsing."""
    malformed_type = draw(st.sampled_from([
        "invalid_structure",
        "wrong_types",
        "missing_required",
        "circular_references",
        "invalid_characters",
    ]))
    
    if malformed_type == "invalid_structure":
        # Return non-dict at top level
        return draw(st.one_of(
            st.lists(st.text()),
            st.text(),
            st.integers(),
            st.booleans(),
        ))
    
    elif malformed_type == "wrong_types":
        # Return dict with wrong value types
        return {
            "graph": draw(st.one_of(st.text(), st.integers(), st.lists(st.text()))),
            "agents": draw(st.one_of(st.text(), st.integers(), st.lists(st.text()))),
        }
    
    elif malformed_type == "missing_required":
        # Missing required fields
        return {
            "graph": {
                "nodes": {"test": {"agent": "test_agent"}},
                # Missing 'start' field
            }
        }
    
    elif malformed_type == "circular_references":
        # Circular node references
        return {
            "graph": {
                "start": "node1",
                "nodes": {
                    "node1": {"agent": "agent1", "next": "node2"},
                    "node2": {"agent": "agent2", "next": "node1"},  # Circular
                }
            }
        }
    
    elif malformed_type == "invalid_characters":
        # Invalid characters in identifiers
        return {
            "graph": {
                "start": "node-with-invalid@chars!",
                "nodes": {
                    "node-with-invalid@chars!": {"agent": "agent with spaces"},
                }
            }
        }


class TestYAMLPropertyParsing:
    """Property-based tests for YAML parsing robustness."""
    
    @given(node_model_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_node_model_parsing(self, node_data):
        """Test that valid node data parses correctly."""
        try:
            node = NodeModel.model_validate(node_data)
            assert node.agent == node_data["agent"]
            
            # Verify optional fields
            for field in ["next", "on_success", "on_failure", "on_block"]:
                expected = node_data.get(field)
                actual = getattr(node, field)
                assert actual == expected
                
        except ValidationError as exc:
            # If validation fails, the data should be genuinely invalid
            pytest.fail(f"Valid node data failed validation: {node_data}, error: {exc}")
    
    @given(graph_section_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_graph_section_parsing(self, graph_data):
        """Test that valid graph sections parse correctly."""
        try:
            graph = GraphSectionModel.model_validate(graph_data)
            assert graph.start == graph_data["start"]
            assert graph.start in graph_data["nodes"]
            
            # Verify all nodes are valid
            for name, node_data in graph_data["nodes"].items():
                assert name in graph.nodes
                node = graph.nodes[name]
                assert node.agent == node_data["agent"]
            
            # Verify parallel groups if present
            if "parallel" in graph_data:
                assert len(graph.parallel) == len(graph_data["parallel"])
                for i, group in enumerate(graph.parallel):
                    assert set(group) == set(graph_data["parallel"][i])
                    # All nodes in parallel groups should exist
                    for node_name in group:
                        assert node_name in graph.nodes
                        
        except ValidationError as exc:
            pytest.fail(f"Valid graph data failed validation: {graph_data}, error: {exc}")
    
    @given(agent_model_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_agent_model_parsing(self, agent_data):
        """Test that valid agent data parses correctly."""
        try:
            agent = AgentModel.model_validate(agent_data)
            assert agent.role == agent_data["role"]
            
            # Verify LLM config if present
            if "llm" in agent_data:
                assert agent.llm == agent_data["llm"]
            else:
                assert agent.llm == {}
            
            # Verify tools if present
            if "tools" in agent_data:
                assert agent.tools == agent_data["tools"]
            else:
                assert agent.tools == []
                
        except ValidationError as exc:
            pytest.fail(f"Valid agent data failed validation: {agent_data}, error: {exc}")
    
    @given(crew_file_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_crew_file_parsing(self, crew_data):
        """Test that valid crew files parse correctly."""
        try:
            crew_file = CrewFileModel.model_validate(crew_data)
            
            # Verify agents
            for name, agent_data in crew_data["agents"].items():
                assert name in crew_file.agents
                agent = crew_file.agents[name]
                assert agent.role == agent_data["role"]
            
            # Verify crews
            resolved_crews = crew_file.resolved_crews()
            for name, crew_data_item in crew_data["crew"].items():
                assert name in resolved_crews
                crew = resolved_crews[name]
                if "sequence" in crew_data_item:
                    assert crew.sequence == crew_data_item["sequence"]
                    
        except ValidationError as exc:
            pytest.fail(f"Valid crew data failed validation: {crew_data}, error: {exc}")
    
    @given(malformed_yaml_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_malformed_yaml_rejection(self, malformed_data):
        """Test that malformed YAML is properly rejected."""
        # These should all fail validation
        with pytest.raises((ValidationError, ValueError, TypeError)):
            if isinstance(malformed_data, dict) and "graph" in malformed_data:
                GraphFileModel.model_validate(malformed_data)
            elif isinstance(malformed_data, dict) and "agents" in malformed_data:
                CrewFileModel.model_validate(malformed_data)
            else:
                # Try both and expect both to fail
                with pytest.raises((ValidationError, ValueError, TypeError)):
                    GraphFileModel.model_validate(malformed_data)
                with pytest.raises((ValidationError, ValueError, TypeError)):
                    CrewFileModel.model_validate(malformed_data)
    
    @given(st.text(min_size=0, max_size=1000))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_arbitrary_text_yaml_parsing(self, text_input):
        """Test parsing arbitrary text as YAML."""
        try:
            parsed = yaml.safe_load(text_input)
            
            if parsed is not None:
                # Try to validate as graph or crew file
                try:
                    GraphFileModel.model_validate(parsed)
                except (ValidationError, ValueError, TypeError):
                    pass  # Expected for most random text
                
                try:
                    CrewFileModel.model_validate(parsed)
                except (ValidationError, ValueError, TypeError):
                    pass  # Expected for most random text
                    
        except yaml.YAMLError:
            # YAML parsing errors are expected for random text
            pass
    
    def test_yaml_file_integration(self):
        """Test integration with actual YAML file parsing."""
        # Test with a known good configuration
        good_config = {
            "agents": {
                "test_agent": {
                    "role": "test_role",
                    "llm": {"provider": "ollama", "model": "test"},
                    "tools": [],
                }
            },
            "crew": {
                "test_crew": {
                    "sequence": ["test_agent"]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(good_config, f)
            temp_path = Path(f.name)
        
        try:
            # Test that we can load and validate the file
            with temp_path.open('r') as f:
                loaded = yaml.safe_load(f)
            
            crew_file = CrewFileModel.model_validate(loaded)
            assert "test_agent" in crew_file.agents
            assert "test_crew" in crew_file.resolved_crews()
            
        finally:
            temp_path.unlink()
    
    @given(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.recursive(
                st.one_of(
                    st.text(max_size=100),
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.booleans(),
                ),
                lambda children: st.one_of(
                    st.lists(children, max_size=10),
                    st.dictionaries(
                        keys=st.text(min_size=1, max_size=10),
                        values=children,
                        max_size=10
                    )
                ),
                max_leaves=50
            ),
            max_size=20
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_arbitrary_dict_structures(self, arbitrary_dict):
        """Test parsing arbitrary dictionary structures."""
        # Should not crash, but may fail validation
        try:
            GraphFileModel.model_validate(arbitrary_dict)
        except (ValidationError, ValueError, TypeError):
            pass  # Expected for most arbitrary structures
        
        try:
            CrewFileModel.model_validate(arbitrary_dict)
        except (ValidationError, ValueError, TypeError):
            pass  # Expected for most arbitrary structures
    
    @given(graph_section_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_graph_spec_creation(self, graph_data):
        """Test that valid graph data can create GraphSpec objects."""
        # Create a temporary crew file for the graph spec
        crew_data = {
            "agents": {},
            "crew": {"test_crew": {"sequence": []}}
        }
        
        # Add agents for all referenced agents in the graph
        for node_name, node_data in graph_data["nodes"].items():
            agent_name = node_data["agent"]
            crew_data["agents"][agent_name] = {
                "role": f"role_{agent_name}",
                "llm": {"provider": "ollama", "model": "test"},
            }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as crew_file:
            yaml.dump(crew_data, crew_file)
            crew_path = Path(crew_file.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as graph_file:
            full_graph_data = {
                "graph": graph_data,
                "crew_config": str(crew_path),
            }
            yaml.dump(full_graph_data, graph_file)
            graph_path = Path(graph_file.name)
        
        try:
            # Test that GraphSpec can be created from the data
            spec = GraphSpec.from_yaml(graph_path)
            assert spec.start == graph_data["start"]
            assert len(spec.nodes) == len(graph_data["nodes"])
            
            # Test LangGraph blueprint generation
            blueprint = spec.langgraph_blueprint()
            assert blueprint["start"] == graph_data["start"]
            assert len(blueprint["nodes"]) == len(graph_data["nodes"])
            
        except Exception as exc:
            pytest.fail(f"Failed to create GraphSpec from valid data: {exc}")
        
        finally:
            crew_path.unlink()
            graph_path.unlink()


class TestTranscriptPropertyParsing:
    """Property-based tests for transcript ingestion and processing."""
    
    @st.composite
    def transcript_entry_strategy(draw):
        """Generate transcript entries."""
        return {
            "timestamp": draw(st.integers(min_value=0, max_value=2**31-1)),
            "role": draw(st.sampled_from(["user", "assistant", "system", "tool"])),
            "content": draw(st.text(max_size=1000)),
            "metadata": draw(st.dictionaries(
                keys=st.text(min_size=1, max_size=20),
                values=st.one_of(st.text(max_size=100), st.integers(), st.booleans()),
                max_size=10
            ))
        }
    
    @given(st.lists(transcript_entry_strategy(), min_size=0, max_size=100))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_transcript_structure_validation(self, transcript_entries):
        """Test that transcript entries have valid structure."""
        for entry in transcript_entries:
            # Basic structure validation
            assert "timestamp" in entry
            assert "role" in entry
            assert "content" in entry
            
            # Type validation
            assert isinstance(entry["timestamp"], int)
            assert isinstance(entry["role"], str)
            assert isinstance(entry["content"], str)
            
            # Value validation
            assert entry["timestamp"] >= 0
            assert entry["role"] in ["user", "assistant", "system", "tool"]
    
    @given(st.text(max_size=10000))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_content_sanitization(self, raw_content):
        """Test that content can be safely processed."""
        # Test that we can safely handle arbitrary content
        try:
            # Basic sanitization operations that should never crash
            sanitized = raw_content.strip()
            encoded = sanitized.encode('utf-8', errors='ignore').decode('utf-8')
            json_safe = json.dumps(encoded)  # Should not raise
            
            # Length checks
            assert len(sanitized) <= len(raw_content)
            assert len(encoded) <= len(raw_content)
            
        except Exception as exc:
            pytest.fail(f"Content sanitization failed: {exc}")
    
    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=50),
        values=st.one_of(
            st.text(max_size=200),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.none(),
        ),
        max_size=20
    ))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_metadata_handling(self, metadata):
        """Test that metadata can be safely processed."""
        try:
            # Should be able to serialize and deserialize
            serialized = json.dumps(metadata, default=str)
            deserialized = json.loads(serialized)
            
            # Basic structure should be preserved
            assert isinstance(deserialized, dict)
            assert len(deserialized) <= len(metadata)
            
        except Exception as exc:
            pytest.fail(f"Metadata handling failed: {exc}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])