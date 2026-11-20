"""Every tunable of the system in one typed tree.

``configs/default.yaml`` is this tree written out; a file given with ``-c`` or
an override given with ``-s section.key=value`` replaces individual leaves.
Unknown keys are rejected, so a typing mistake in a threshold cannot pass for a
setting.

A trained bundle carries the front-end, quality, clench and arbiter settings it
was built with. When driving with a bundle those win, because the model is only
valid for them; the sections here apply to recording, simulation and untrained
driving.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

from .calibration.protocol import ProtocolConfig
from .config import apply_overrides, deep_merge, from_dict, load_yaml, to_dict
from .dsp.frontend import FrontendConfig
from .dsp.quality import QualityConfig
from .runtime.engine import EngineConfig
from .sim.eeg import SimConfig


@dataclass(frozen=True)
class Settings:
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    sim: SimConfig = field(default_factory=SimConfig)


def defaults() -> dict[str, Any]:
    return to_dict(Settings())


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        default=[],
        metavar="FILE",
        help="YAML settings file (repeatable; later files win)",
    )
    parser.add_argument(
        "-s",
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one setting, e.g. -s engine.baseline_s=30",
    )


def load(args: argparse.Namespace | None = None, **replace: dict[str, Any]) -> Settings:
    """Defaults, then ``--config`` files, then ``--set`` overrides, then ``replace`` (section -> values)."""
    merged = defaults()
    for path in getattr(args, "config", []) or []:
        merged = deep_merge(merged, load_yaml(path))
    merged = apply_overrides(merged, getattr(args, "overrides", []) or [])
    for section, values in replace.items():
        merged = deep_merge(merged, {section: {k: v for k, v in values.items() if v is not None}})
    return from_dict(Settings, merged)
