from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nwc.config import ConfigError, apply_overrides, deep_merge, fingerprint, from_dict, load_json, save_json, to_dict


@dataclass(frozen=True)
class Inner:
    gain: float = 1.0
    bands: tuple[tuple[float, float], ...] = ((1.0, 4.0),)


@dataclass(frozen=True)
class Outer:
    name: str = "x"
    count: int = 3
    inner: Inner = field(default_factory=Inner)
    maybe: Inner | None = None
    table: dict[str, tuple[float, float]] = field(default_factory=dict)


def test_typed_settings_round_trip_and_reject_mistakes(tmp_path):
    value = Outer("a", 2, Inner(0.5, ((8.0, 13.0), (13.0, 30.0))), Inner(), {"left": (0.1, -0.1)})
    assert from_dict(Outer, to_dict(value)) == value
    assert from_dict(Outer, {"count": 4.0, "inner": {"gain": 2}}) == Outer(
        count=4, inner=Inner(gain=2.0)
    )  # whole floats are ints, ints are floats
    for wrong in (
        {"cuont": 1},
        {"count": "three"},
        {"count": True},
        {"inner": 5},
        {"inner": {"bands": [[1.0]]}},
        {"table": {"x": [1]}},
    ):
        with pytest.raises(ConfigError):
            from_dict(Outer, wrong)
    assert fingerprint(value) == fingerprint(
        Outer("a", 2, Inner(0.5, ((8.0, 13.0), (13.0, 30.0))), Inner(), {"left": (0.1, -0.1)})
    )
    assert fingerprint(value) != fingerprint(Outer("a", 2)) and len(fingerprint(value)) == 16
    merged = deep_merge({"a": {"b": 1, "c": 2}, "d": 3}, {"a": {"b": 9}, "e": 4})
    assert merged == {"a": {"b": 9, "c": 2}, "d": 3, "e": 4}
    assert apply_overrides({"a": {"b": 1}}, ["a.b=2.5", "x.y=[1, 2]", "z=null"]) == {
        "a": {"b": 2.5},
        "x": {"y": [1, 2]},
        "z": None,
    }
    for wrong in (["nokey"], ["=1"], ["a.b.c=1"]):
        with pytest.raises(ConfigError):
            apply_overrides({"a": {"b": 1}}, wrong)
    save_json(tmp_path / "deep" / "x.json", value)
    assert load_json(tmp_path / "deep" / "x.json")["inner"]["bands"] == [[8.0, 13.0], [13.0, 30.0]]
