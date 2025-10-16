from symbiont.orchestrator import Orchestrator


def _create_plugin_manifest(tmp_path, monkeypatch):
    pkg_dir = tmp_path / "swarm_plugins"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "demo.py").write_text(
        "class DemoPlugin:\n"
        "    def __init__(self, message='hi'):\n"
        "        self.message = message\n"
        "        self.registered = False\n"
        "\n"
        "    def register(self, orchestrator):\n"
        "        orchestrator.config.setdefault('plugin_messages', []).append(self.message)\n"
        "        self.registered = True\n"
        "\n"
        "def build_plugin(greeting='hi'):\n"
        "    return DemoPlugin(message=greeting)\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    manifest = (
        "plugins:\n"
        "  - name: demo\n"
        "    module: swarm_plugins.demo\n"
        "    attribute: build_plugin\n"
        "    enabled: true\n"
        "    config:\n"
        "      greeting: overridden\n"
    )
    manifest_path = tmp_path / "plugins.yml"
    manifest_path.write_text(manifest, encoding="utf-8")
    return manifest_path


def test_orchestrator_loads_plugins(tmp_path, monkeypatch):
    manifest_path = _create_plugin_manifest(tmp_path, monkeypatch)
    db_path = tmp_path / "data" / "sym.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "db_path": str(db_path),
        "llm": {},
        "evolution": {"swarm_enabled": False},
        "plugins": {"manifest": str(manifest_path)},
    }

    orchestrator = Orchestrator(config)

    assert "demo" in orchestrator.plugins
    plugin = orchestrator.plugins["demo"]
    assert getattr(plugin, "registered", False) is True
    assert getattr(plugin, "message", "") == "overridden"
    assert "overridden" in orchestrator.config.get("plugin_messages", [])
