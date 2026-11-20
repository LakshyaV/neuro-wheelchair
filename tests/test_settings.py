from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from nwc import settings
from nwc.config import from_dict

ROOT = Path(__file__).resolve().parent.parent


def test_default_settings_file_matches_the_code():
    """configs/default.yaml is generated; if this fails, run `make config`."""
    spec = importlib.util.spec_from_file_location("write_default_config", ROOT / "tools" / "write_default_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = ROOT / "configs" / "default.yaml"
    assert path.read_text(encoding="utf-8") == module.render()
    loaded = from_dict(settings.Settings, yaml.safe_load(path.read_text(encoding="utf-8")))
    assert loaded == settings.Settings()  # and reading it back gives exactly the defaults
