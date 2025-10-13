"""Property-based testing for YAML parsing using Hypothesis.

This module tests the robustness of YAML parsing for crew and graph configurations
by generating malformed inputs and ensuring graceful failure handling.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union

import pytest
import yaml
from hypothesis import given, strategies as st

from symbiont.agents.registry import AgentRegistry
from symbiont.orchestration.graph import GraphRunner
from symbiont.orchestration.schema import validate_crew_config, validate_graph_config


class YAMLPropertyTests:
    """Property-based tests for YAML parsing robustness."""

    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=50),
        values=st.one_of(
            st.text(max_size=1000),
            st.integers(),
            st.floats(),
            st.booleans(),
            st.lists(st.text(max_size=100)),
            st.dictionaries(st.text(max_size=20), st.text(max_size=50))
        ),
        min_size=0,
        max_size=20
    ))
    def test_crew_yaml_parsing_robustness(self, data: Dict[str, Any]) -> None:
        """Test that crew YAML parsing handles arbitrary data structures gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                yaml.dump(data, f)
                f.flush()
                
                # Test that parsing doesn't crash
                with open(f.name, 'r') as yaml_file:
                    parsed = yaml.safe_load(yaml_file)
                
                # Test validation
                try:
                    validate_crew_config(parsed)
                except Exception as e:
                    # Validation should fail gracefully with meaningful errors
                    assert isinstance(e, (ValueError, TypeError, KeyError))
                
            finally:
                Path(f.name).unlink(missing_ok=True)

    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=50),
        values=st.one_of(
            st.text(max_size=1000),
            st.integers(),
            st.floats(),
            st.booleans(),
            st.lists(st.text(max_size=100)),
            st.dictionaries(st.text(max_size=20), st.text(max_size=50))
        ),
        min_size=0,
        max_size=20
    ))
    def test_graph_yaml_parsing_robustness(self, data: Dict[str, Any]) -> None:
        """Test that graph YAML parsing handles arbitrary data structures gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                yaml.dump(data, f)
                f.flush()
                
                # Test that parsing doesn't crash
                with open(f.name, 'r') as yaml_file:
                    parsed = yaml.safe_load(yaml_file)
                
                # Test validation
                try:
                    validate_graph_config(parsed)
                except Exception as e:
                    # Validation should fail gracefully with meaningful errors
                    assert isinstance(e, (ValueError, TypeError, KeyError))
                
            finally:
                Path(f.name).unlink(missing_ok=True)

    @given(st.text(min_size=1, max_size=10000))
    def test_malformed_yaml_handling(self, malformed_yaml: str) -> None:
        """Test handling of malformed YAML content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                f.write(malformed_yaml)
                f.flush()
                
                # Should either parse successfully or raise a YAML parsing error
                with open(f.name, 'r') as yaml_file:
                    try:
                        parsed = yaml.safe_load(yaml_file)
                        # If parsing succeeds, validation should handle it
                        validate_crew_config(parsed)
                    except yaml.YAMLError:
                        # YAML parsing errors are expected for malformed content
                        pass
                    except Exception as e:
                        # Other errors should be validation errors, not crashes
                        assert isinstance(e, (ValueError, TypeError, KeyError))
                
            finally:
                Path(f.name).unlink(missing_ok=True)

    @given(st.lists(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.floats(),
                st.booleans()
            ),
            min_size=1,
            max_size=10
        ),
        min_size=0,
        max_size=50
    ))
    def test_crew_list_parsing(self, crew_list: List[Dict[str, Any]]) -> None:
        """Test parsing of crew lists with various structures."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                yaml.dump({"crews": crew_list}, f)
                f.flush()
                
                with open(f.name, 'r') as yaml_file:
                    parsed = yaml.safe_load(yaml_file)
                
                # Test that registry can handle the parsed data
                try:
                    registry = AgentRegistry.from_yaml(f.name)
                    # Should not crash even with malformed data
                except Exception as e:
                    # Should fail gracefully with validation errors
                    assert isinstance(e, (ValueError, TypeError, KeyError))
                
            finally:
                Path(f.name).unlink(missing_ok=True)

    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=30),
        values=st.one_of(
            st.text(max_size=500),
            st.integers(),
            st.floats(),
            st.booleans(),
            st.lists(st.text(max_size=50)),
            st.dictionaries(st.text(max_size=15), st.text(max_size=30))
        ),
        min_size=1,
        max_size=15
    ))
    def test_graph_config_parsing(self, graph_config: Dict[str, Any]) -> None:
        """Test parsing of graph configurations with various structures."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                yaml.dump(graph_config, f)
                f.flush()
                
                with open(f.name, 'r') as yaml_file:
                    parsed = yaml.safe_load(yaml_file)
                
                # Test that graph runner can handle the parsed data
                try:
                    runner = GraphRunner.from_yaml(f.name)
                    # Should not crash even with malformed data
                except Exception as e:
                    # Should fail gracefully with validation errors
                    assert isinstance(e, (ValueError, TypeError, KeyError))
                
            finally:
                Path(f.name).unlink(missing_ok=True)

    @given(st.text(min_size=1, max_size=1000))
    def test_yaml_with_special_characters(self, content: str) -> None:
        """Test YAML parsing with special characters and edge cases."""
        # Create YAML content with special characters
        yaml_content = f"""
        test_key: "{content}"
        unicode: "测试中文 🚀"
        special_chars: "!@#$%^&*()_+-=[]{{}}|;':\",./<>?"
        multiline: |
            {content}
            more content
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                f.write(yaml_content)
                f.flush()
                
                with open(f.name, 'r') as yaml_file:
                    parsed = yaml.safe_load(yaml_file)
                
                # Should parse without crashing
                assert isinstance(parsed, dict)
                
            finally:
                Path(f.name).unlink(missing_ok=True)

    @given(st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(),
        st.text(),
        st.lists(st.anything()),
        st.dictionaries(st.anything(), st.anything())
    ))
    def test_yaml_root_types(self, root_value: Any) -> None:
        """Test YAML parsing with various root value types."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                yaml.dump(root_value, f)
                f.flush()
                
                with open(f.name, 'r') as yaml_file:
                    parsed = yaml.safe_load(yaml_file)
                
                # Should parse without crashing
                assert parsed is not None or root_value is None
                
            finally:
                Path(f.name).unlink(missing_ok=True)


# Test cases for specific edge cases
class YAMLEdgeCaseTests:
    """Tests for specific YAML edge cases and malformed inputs."""

    def test_empty_yaml_file(self) -> None:
        """Test handling of empty YAML files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                f.write("")
                f.flush()
                
                with open(f.name, 'r') as yaml_file:
                    parsed = yaml.safe_load(yaml_file)
                
                # Empty YAML should return None
                assert parsed is None
                
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_yaml_with_comments_only(self) -> None:
        """Test YAML files containing only comments."""
        yaml_content = """
        # This is a comment
        # Another comment
        # No actual data
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                f.write(yaml_content)
                f.flush()
                
                with open(f.name, 'r') as yaml_file:
                    parsed = yaml.safe_load(yaml_file)
                
                # Comments-only YAML should return None
                assert parsed is None
                
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_yaml_with_duplicate_keys(self) -> None:
        """Test YAML with duplicate keys (should use last value)."""
        yaml_content = """
        key: "first"
        key: "second"
        key: "third"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                f.write(yaml_content)
                f.flush()
                
                with open(f.name, 'r') as yaml_file:
                    parsed = yaml.safe_load(yaml_file)
                
                # Should use the last value
                assert parsed == {"key": "third"}
                
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_yaml_with_circular_references(self) -> None:
        """Test YAML that would create circular references."""
        # This is tricky to test directly, but we can test the behavior
        yaml_content = """
        circular: &circular
          value: 1
          ref: *circular
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                f.write(yaml_content)
                f.flush()
                
                with open(f.name, 'r') as yaml_file:
                    parsed = yaml.safe_load(yaml_file)
                
                # Should handle circular references gracefully
                assert isinstance(parsed, dict)
                
            finally:
                Path(f.name).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])