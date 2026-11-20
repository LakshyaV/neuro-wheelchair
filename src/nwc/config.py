"""Typed configuration helpers shared by every layer.

Each layer owns a small frozen dataclass describing its parameters. This module
turns those dataclasses into plain dictionaries (for YAML and for bundles) and
back, and derives a stable fingerprint from them.

The fingerprint is the contract between training and inference: a bundle stores
the fingerprint of the front-end that produced its training features, and the
runtime refuses to feed a model with features computed by any other front-end.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
import typing
from pathlib import Path
from typing import Any, TypeVar, Union

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a configuration mapping does not match its dataclass."""


def to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses, tuples and paths into JSON/YAML-safe values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):  # numpy scalars and arrays
        return obj.tolist()
    return obj


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if tp is Any or value is None:
        return value
    if dataclasses.is_dataclass(tp):
        if isinstance(value, tp):
            return value
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping for {tp.__name__}, got {value!r}")
        return from_dict(tp, value, _where=where)
    if origin in (Union, types.UnionType):
        last_error: Exception | None = None
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _coerce(candidate, value, where)
            except (ConfigError, TypeError, ValueError) as exc:
                last_error = exc
        raise ConfigError(f"{where}: {value!r} matches none of {args}") from last_error
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a sequence, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
        if args and len(args) != len(value):
            raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
        if args:
            return tuple(_coerce(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
        return tuple(value)
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a sequence, got {value!r}")
        inner = args[0] if args else Any
        return [_coerce(inner, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        inner = args[1] if len(args) == 2 else Any
        return {k: _coerce(inner, v, f"{where}.{k}") for k, v in value.items()}
    if tp is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if tp is int and isinstance(value, bool):
        raise ConfigError(f"{where}: expected int, got bool")
    if tp is int and isinstance(value, float) and value.is_integer():
        return int(value)
    if tp is Path:
        return Path(value)
    if isinstance(tp, type) and not isinstance(value, tp):
        raise ConfigError(f"{where}: expected {tp.__name__}, got {type(value).__name__} {value!r}")
    return value


def from_dict(cls: type[T], data: dict[str, Any] | None, *, _where: str = "") -> T:
    """Build dataclass ``cls`` from a mapping, rejecting unknown keys.

    Unknown keys are an error rather than a warning: a silently ignored typo in a
    threshold is exactly the kind of train/inference mismatch this project is
    designed to rule out.
    """
    data = dict(data or {})
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    prefix = f"{_where}." if _where else ""
    known = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{prefix or cls.__name__ + ': '}unknown option(s) {unknown}; valid options are {sorted(known)}"
        )
    for name in known & set(data):
        kwargs[name] = _coerce(hints[name], data[name], f"{prefix}{name}")
    return cls(**kwargs)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return loaded


def save_yaml(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_dict(data), handle, sort_keys=False, allow_unicode=True)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_dict(data), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def fingerprint(obj: Any, length: int = 16) -> str:
    """Stable short hash of a configuration object (order- and platform-independent)."""
    canonical = json.dumps(to_dict(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()[:length]


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides (values parsed as YAML) to a nested mapping."""
    result = json.loads(json.dumps(to_dict(data)))
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like section.key=value")
        dotted, raw = item.split("=", 1)
        keys = [k for k in dotted.strip().split(".") if k]
        if not keys:
            raise ConfigError(f"override {item!r} has an empty key")
        cursor = result
        for key in keys[:-1]:
            nxt = cursor.setdefault(key, {})
            if not isinstance(nxt, dict):
                raise ConfigError(f"override {item!r}: {key!r} is not a section")
            cursor = nxt
        cursor[keys[-1]] = yaml.safe_load(raw)
    return result


def deep_merge(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``base`` updated recursively with ``extra`` (neither input is modified)."""
    merged = dict(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
