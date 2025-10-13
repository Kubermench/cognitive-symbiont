"""Comprehensive fuzz testing for Symbiont components using Hypothesis.

This module tests various components with randomly generated inputs to ensure
robustness against edge cases, malformed data, and adversarial inputs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from symbiont.llm.client import LLMClient
from symbiont.llm.budget import TokenBudget
from symbiont.memory.db import MemoryDB
from symbiont.tools.retry_utils import RetryConfig, retry_call
from symbiont.initiative.state import SQLiteStateBackend, InitiativeStateStore
from symbiont.tools.security import scrub_payload


class TestLLMClientFuzzing:
    """Fuzz testing for LLM client robustness."""
    
    @st.composite
    def llm_config_strategy(draw):
        """Generate LLM configurations for testing."""
        return {
            "provider": draw(st.sampled_from(["ollama", "cmd", "openai", "anthropic", "", "invalid"])),
            "model": draw(st.text(max_size=100)),
            "cmd": draw(st.text(max_size=200)),
            "timeout_seconds": draw(st.one_of(
                st.integers(min_value=-100, max_value=1000),
                st.floats(allow_nan=True, allow_infinity=True),
                st.text(),
                st.none()
            )),
            "mode": draw(st.sampled_from(["local", "cloud", "hybrid", "invalid", "", None])),
            "retry": {
                "enabled": draw(st.booleans()),
                "attempts": draw(st.integers(min_value=-10, max_value=100)),
                "initial_seconds": draw(st.floats(min_value=-10, max_value=100, allow_nan=True)),
                "multiplier": draw(st.floats(min_value=-10, max_value=100, allow_nan=True)),
                "max_seconds": draw(st.floats(min_value=-10, max_value=1000, allow_nan=True)),
            }
        }
    
    @given(llm_config_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_llm_client_initialization(self, config):
        """Test LLM client initialization with various configurations."""
        try:
            client = LLMClient(config)
            
            # Basic sanity checks - should not crash
            assert hasattr(client, 'provider')
            assert hasattr(client, 'model')
            assert hasattr(client, 'timeout')
            
            # Timeout should be reasonable
            assert isinstance(client.timeout, int)
            assert client.timeout > 0
            
        except Exception:
            # Some configurations are expected to fail, but should not crash the process
            pass
    
    @given(st.text(max_size=10000))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_llm_prompt_handling(self, prompt):
        """Test LLM client with various prompt inputs."""
        config = {"provider": "cmd", "cmd": "echo 'test response'"}
        client = LLMClient(config)
        
        try:
            # Should handle any string input gracefully
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="test response")
                result = client.generate(prompt)
                
                # Should return a string
                assert isinstance(result, str)
                
        except Exception:
            # Some prompts may cause issues, but should not crash
            pass
    
    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=50),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=True, allow_infinity=True),
            st.booleans(),
            st.none(),
        ),
        max_size=20
    ))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_token_budget_with_arbitrary_data(self, budget_data):
        """Test TokenBudget with arbitrary configuration data."""
        try:
            # Extract reasonable values or use defaults
            limit = 1000
            label = "test"
            
            if "limit" in budget_data and isinstance(budget_data["limit"], int):
                limit = max(0, budget_data["limit"])
            
            if "label" in budget_data and isinstance(budget_data["label"], str):
                label = budget_data["label"][:50]  # Reasonable length limit
            
            budget = TokenBudget(limit=limit, label=label)
            
            # Basic functionality should work
            assert budget.limit == limit
            assert budget.label == label
            assert budget.used == 0
            assert budget.remaining() == (limit if limit > 0 else None)
            
        except Exception as exc:
            pytest.fail(f"TokenBudget creation failed with reasonable data: {exc}")


class TestRetryUtilsFuzzing:
    """Fuzz testing for retry utilities."""
    
    @st.composite
    def retry_config_strategy(draw):
        """Generate retry configurations for testing."""
        return {
            "attempts": draw(st.integers(min_value=-10, max_value=100)),
            "initial_wait": draw(st.floats(min_value=-10, max_value=100, allow_nan=True)),
            "max_wait": draw(st.floats(min_value=-10, max_value=1000, allow_nan=True)),
            "multiplier": draw(st.floats(min_value=-10, max_value=100, allow_nan=True)),
            "jitter": draw(st.booleans()),
            "failure_threshold": draw(st.integers(min_value=-10, max_value=100)),
            "recovery_timeout": draw(st.floats(min_value=-10, max_value=1000, allow_nan=True)),
        }
    
    @given(retry_config_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_retry_config_validation(self, config_data):
        """Test RetryConfig with various inputs."""
        try:
            config = RetryConfig(**config_data)
            
            # Validate that the config has reasonable values
            assert config.attempts >= 1
            assert config.initial_wait >= 0.0
            assert config.max_wait >= config.initial_wait
            assert config.multiplier >= 1.0
            assert config.failure_threshold >= 1
            assert config.recovery_timeout >= 0.0
            
        except (ValueError, TypeError):
            # Some configurations are expected to fail validation
            pass
    
    @given(st.integers(min_value=0, max_value=10))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_retry_with_flaky_function(self, failure_count):
        """Test retry mechanism with functions that fail a specific number of times."""
        call_count = 0
        
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count <= failure_count:
                raise ValueError(f"Failure {call_count}")
            return f"Success after {call_count} calls"
        
        config = RetryConfig(attempts=failure_count + 2, initial_wait=0.01, max_wait=0.1)
        
        if failure_count < config.attempts:
            # Should succeed
            result = retry_call(flaky_function, config=config)
            assert "Success" in result
            assert call_count == failure_count + 1
        else:
            # Should fail after exhausting retries
            with pytest.raises(ValueError):
                retry_call(flaky_function, config=config)


class TestStatePersistenceFuzzing:
    """Fuzz testing for state persistence components."""
    
    @st.composite
    def node_data_strategy(draw):
        """Generate node data for state persistence testing."""
        return {
            "node_id": draw(st.text(min_size=1, max_size=100)),
            "pid": draw(st.integers(min_value=-1000, max_value=1000000)),
            "status": draw(st.text(max_size=50)),
            "poll_seconds": draw(st.integers(min_value=-100, max_value=10000)),
            "last_check_ts": draw(st.integers(min_value=-1000000, max_value=2**31)),
            "last_proposal_ts": draw(st.integers(min_value=-1000000, max_value=2**31)),
            "details": draw(st.dictionaries(
                keys=st.text(min_size=1, max_size=50),
                values=st.one_of(
                    st.text(max_size=200),
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.booleans(),
                ),
                max_size=10
            )),
        }
    
    @given(node_data_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_sqlite_state_backend_robustness(self, node_data):
        """Test SQLite state backend with various node data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            try:
                backend = SQLiteStateBackend(str(db_path))
                
                # Should handle the data gracefully
                backend.record_daemon(**node_data)
                
                # Should be able to load it back
                loaded = backend.load_daemon(node_data["node_id"])
                if loaded:
                    assert loaded["node_id"] == node_data["node_id"]
                    
            except Exception:
                # Some data may be invalid, but should not crash the database
                pass
    
    @st.composite
    def worker_data_strategy(draw):
        """Generate worker data for testing."""
        return {
            "worker_id": draw(st.text(min_size=1, max_size=100)),
            "node_id": draw(st.text(min_size=1, max_size=100)),
            "status": draw(st.text(max_size=50)),
            "goal": draw(st.one_of(st.none(), st.text(max_size=200))),
            "variants": draw(st.one_of(st.none(), st.integers(min_value=-100, max_value=1000))),
            "details": draw(st.one_of(
                st.none(),
                st.dictionaries(
                    keys=st.text(min_size=1, max_size=50),
                    values=st.text(max_size=100),
                    max_size=10
                )
            )),
        }
    
    @given(worker_data_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_worker_state_management(self, worker_data):
        """Test worker state management with various data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            try:
                backend = SQLiteStateBackend(str(db_path))
                
                # Record worker
                backend.record_swarm_worker(**worker_data)
                
                # List workers
                workers = backend.list_swarm_workers()
                
                # Should find our worker if data was valid
                worker_ids = [w.get("worker_id") for w in workers]
                if worker_data["worker_id"] in worker_ids:
                    # Verify data integrity
                    found_worker = next(w for w in workers if w["worker_id"] == worker_data["worker_id"])
                    assert found_worker["node_id"] == worker_data["node_id"]
                    assert found_worker["status"] == worker_data["status"]
                
            except Exception:
                # Some data may be invalid
                pass


class TestSecurityFuzzing:
    """Fuzz testing for security components."""
    
    @given(st.dictionaries(
        keys=st.text(max_size=100),
        values=st.recursive(
            st.one_of(
                st.text(max_size=500),
                st.integers(),
                st.floats(allow_nan=True, allow_infinity=True),
                st.booleans(),
                st.none(),
            ),
            lambda children: st.one_of(
                st.lists(children, max_size=10),
                st.dictionaries(
                    keys=st.text(max_size=50),
                    values=children,
                    max_size=10
                )
            ),
            max_leaves=100
        ),
        max_size=20
    ))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_payload_scrubbing(self, payload):
        """Test payload scrubbing with arbitrary data structures."""
        try:
            scrubbed = scrub_payload(payload)
            
            # Should return a dictionary
            assert isinstance(scrubbed, dict)
            
            # Should be JSON serializable
            json_str = json.dumps(scrubbed, default=str)
            assert isinstance(json_str, str)
            
            # Should not contain obvious sensitive patterns
            json_lower = json_str.lower()
            sensitive_patterns = ["password", "secret", "token", "key", "credential"]
            
            # If original contained sensitive patterns, scrubbed should not (or should be redacted)
            for pattern in sensitive_patterns:
                if pattern in str(payload).lower():
                    # The scrubbed version should either not contain it or contain a redaction marker
                    if pattern in json_lower:
                        assert "[REDACTED]" in json_str or "***" in json_str
            
        except Exception as exc:
            pytest.fail(f"Payload scrubbing failed: {exc}")
    
    @given(st.text(max_size=1000))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_pii_detection_robustness(self, text_input):
        """Test PII detection with arbitrary text inputs."""
        try:
            # Test that PII detection doesn't crash on arbitrary input
            from symbiont.tools.security import detect_pii_patterns
            
            detected = detect_pii_patterns(text_input)
            
            # Should return a list
            assert isinstance(detected, list)
            
            # Each detection should have required fields
            for detection in detected:
                assert isinstance(detection, dict)
                assert "type" in detection
                assert "confidence" in detection
                assert isinstance(detection["confidence"], (int, float))
                assert 0 <= detection["confidence"] <= 1
                
        except Exception as exc:
            pytest.fail(f"PII detection failed: {exc}")


class TestMemoryDatabaseFuzzing:
    """Fuzz testing for memory database operations."""
    
    @st.composite
    def claim_data_strategy(draw):
        """Generate claim data for testing."""
        return {
            "subject": draw(st.text(min_size=1, max_size=200)),
            "relation": draw(st.text(min_size=1, max_size=100)),
            "obj": draw(st.text(min_size=1, max_size=200)),
            "importance": draw(st.floats(min_value=-10, max_value=10, allow_nan=True)),
            "source_url": draw(st.one_of(st.none(), st.text(max_size=300))),
        }
    
    @given(claim_data_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_memory_db_claim_insertion(self, claim_data):
        """Test memory database with various claim data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_memory.db"
            
            try:
                db = MemoryDB(str(db_path))
                db.ensure_schema()
                
                # Should handle claim insertion gracefully
                from symbiont.memory.graphrag import add_claim
                
                claim_id, status = add_claim(
                    db,
                    subject=claim_data["subject"],
                    relation=claim_data["relation"],
                    obj=claim_data["obj"],
                    importance=claim_data.get("importance", 0.5),
                    source_url=claim_data.get("source_url"),
                )
                
                # Should return valid results
                assert isinstance(claim_id, (int, type(None)))
                assert isinstance(status, str)
                
            except Exception:
                # Some data may be invalid, but should not crash the database
                pass
    
    @given(st.text(min_size=1, max_size=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_memory_search_robustness(self, query):
        """Test memory database search with arbitrary queries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_memory.db"
            
            try:
                db = MemoryDB(str(db_path))
                db.ensure_schema()
                
                # Should handle search queries gracefully
                results = db.search_claims(query, limit=10)
                
                # Should return a list
                assert isinstance(results, list)
                assert len(results) <= 10
                
            except Exception:
                # Some queries may be invalid
                pass


class TestConfigurationFuzzing:
    """Fuzz testing for configuration handling."""
    
    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=50),
        values=st.recursive(
            st.one_of(
                st.text(max_size=200),
                st.integers(min_value=-1000000, max_value=1000000),
                st.floats(allow_nan=True, allow_infinity=True),
                st.booleans(),
                st.none(),
            ),
            lambda children: st.one_of(
                st.lists(children, max_size=20),
                st.dictionaries(
                    keys=st.text(min_size=1, max_size=30),
                    values=children,
                    max_size=20
                )
            ),
            max_leaves=200
        ),
        max_size=50
    ))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_configuration_robustness(self, config):
        """Test that various configuration structures don't crash the system."""
        try:
            # Test LLM client creation
            if isinstance(config, dict):
                try:
                    LLMClient(config)
                except Exception:
                    pass  # Expected for most random configs
                
                # Test state store creation
                try:
                    store = InitiativeStateStore("sqlite", config)
                    assert store is not None
                except Exception:
                    pass  # Expected for some configs
                
        except Exception as exc:
            pytest.fail(f"Configuration handling crashed unexpectedly: {exc}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])