import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.orchestration.graph import GraphSpec
from symbiont.agents.registry import AgentRegistry
from symbiont.memory import db as memory_db


def test_graph_validation_requires_agent(tmp_path):
    graph_yaml = tmp_path / "graph.yaml"
    graph_yaml.write_text(
        """
crew_config: ./crew_template.yaml
graph:
  start: node_a
  nodes:
    node_a: {}
""",
        encoding="utf-8",
    )
    with pytest.raises((ValueError, ValidationError)):
        GraphSpec.from_yaml(graph_yaml)


def test_crew_validation_requires_list(tmp_path):
    crew_yaml = tmp_path / "crew.yaml"
    crew_yaml.write_text(
        """
agents:
  scout:
    role: scout
crew:
  default:
    sequence: architect
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        AgentRegistry.from_yaml(crew_yaml)


def test_memory_db_records_schema_version(tmp_path):
    path = tmp_path / "sym.db"
    database = memory_db.MemoryDB(str(path))
    database.ensure_schema()
    assert database.schema_version() == memory_db.SCHEMA_VERSION


def test_memory_db_rejects_newer_schema(tmp_path):
    path = tmp_path / "sym.db"
    database = memory_db.MemoryDB(str(path))
    database.ensure_schema()
    with database._conn() as conn:  # noqa: SLF001 - internal use for test
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(memory_db.SCHEMA_VERSION + 10),),
        )
    with pytest.raises(RuntimeError):
        database.ensure_schema()
