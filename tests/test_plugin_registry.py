from typer.testing import CliRunner
import yaml

from symbiont.cli import app

from symbiont.plugins import PluginRegistry


def test_registry_handles_missing_manifest(tmp_path):
    missing = tmp_path / "plugins.yml"
    registry = PluginRegistry(manifest_path=str(missing))
    assert registry.entries == {}
    assert list(registry.enabled()) == []


def _prepare_manifest(tmp_path, monkeypatch):
    pkg_dir = tmp_path / "beta_plugins"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "demo.py").write_text(
        "def build_plugin(greeting='hi', **kwargs):\n"
        "    payload = {'greeting': greeting}\n"
        "    meta = kwargs.pop('meta', None)\n"
        "    if meta is not None:\n"
        "        payload['meta'] = meta\n"
        "    if kwargs:\n"
        "        payload['extras'] = kwargs\n"
        "    return payload\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    manifest = {
        "plugins": [
            {
                "name": "demo",
                "module": "beta_plugins.demo",
                "attribute": "build_plugin",
                "description": "Demo plugin for Swarm beta.",
                "enabled": True,
                "tags": ["beta"],
                "config": {"greeting": "hello", "meta": {"build": "swarm"}},
            },
            {
                "name": "disabled",
                "module": "beta_plugins.demo",
                "attribute": "build_plugin",
                "enabled": False,
            },
        ]
    }

    manifest_path = tmp_path / "plugins.yml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    return manifest_path


def test_registry_instantiates_enabled_plugins(tmp_path, monkeypatch):
    manifest_path = _prepare_manifest(tmp_path, monkeypatch)
    registry = PluginRegistry(manifest_path=str(manifest_path))

    entry = registry.get("demo")
    assert entry is not None
    assert entry.description == "Demo plugin for Swarm beta."

    instance = entry.instantiate()
    assert instance == {"greeting": "hello", "meta": {"build": "swarm"}}

    instances = registry.instantiate_enabled()
    assert "demo" in instances
    assert "disabled" not in instances


def test_cli_plugins_list(tmp_path, monkeypatch):
    manifest_path = _prepare_manifest(tmp_path, monkeypatch)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "plugins-list",
            "--manifest",
            str(manifest_path),
            "--enabled-only",
        ],
    )
    assert result.exit_code == 0
    assert "demo" in result.stdout
    assert "disabled" not in result.stdout
