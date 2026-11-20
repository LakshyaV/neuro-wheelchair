"""A trained model as a directory that explains and checks itself.

::

    bundle.json   what the model is: front-end, features, classes, thresholds,
                  how it was evaluated and what that found
    model.npz     the numbers: standardisation, weights, temperature, self-test

Three things are verified every time a bundle is loaded, before it can drive
anything:

* the front-end configuration it was trained with still hashes to the
  fingerprint stored beside it, and the runtime builds its front-end from that
  configuration, so features cannot differ from training;
* the arrays fit the declared classes and features;
* a handful of real frames stored at training time are run through the model
  again and must reproduce the stored probabilities. A change in a library, a
  platform or this code that altered the arithmetic would be caught here and
  not on the road.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..config import from_dict, load_json, save_json, to_dict
from ..control.arbiter import ArbiterConfig
from ..dsp.frontend import FrontendConfig
from ..dsp.quality import QualityConfig
from ..dsp.reflex import ClenchConfig
from .features import FeatureSpec
from .model import IntentModel

FORMAT = 1
MANIFEST = "bundle.json"
ARRAYS = "model.npz"


class BundleError(RuntimeError):
    """A bundle is missing, damaged or does not reproduce its own self-test."""


@dataclass
class Bundle:
    model: IntentModel
    frontend: FrontendConfig
    quality: QualityConfig = field(default_factory=QualityConfig)
    clench: ClenchConfig = field(default_factory=ClenchConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    recommended_scheme: str = "switch"
    reliable: tuple[str, ...] = ()
    report: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def classes(self) -> tuple[str, ...]:
        return self.model.classes

    def describe(self) -> str:
        evaluation = self.report.get("evaluation", {})
        accuracy = evaluation.get("balanced_accuracy")
        parts = [
            f"classes {', '.join(self.classes)}",
            f"features {self.model.spec.name}",
            f"front-end {self.frontend.fingerprint}",
        ]
        if accuracy is not None:
            p = evaluation.get("p_value")
            parts.append(
                f"held-out accuracy {accuracy:.2f} (chance {evaluation.get('chance', 0):.2f}"
                + (f", p = {p:.3f})" if p is not None else ")")
            )
        parts.append(f"recommended scheme {self.recommended_scheme}")
        return " | ".join(parts)


def save_bundle(
    directory: str | Path, bundle: Bundle, selftest_cosp: np.ndarray, selftest_reference: np.ndarray
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = bundle.model
    selftest_cosp = np.asarray(selftest_cosp, dtype=np.float32)
    expected = model.log_proba(selftest_cosp, selftest_reference)
    np.savez(
        directory / ARRAYS,
        **model.arrays(),
        selftest_cosp=selftest_cosp,
        selftest_reference=np.asarray(selftest_reference, dtype=np.float64),
        selftest_log_proba=expected,
    )
    manifest = {
        "format": FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "software": f"nwc {__version__}",
        "classes": list(model.classes),
        "features": to_dict(model.spec),
        "frontend": to_dict(bundle.frontend),
        "frontend_fingerprint": bundle.frontend.fingerprint,
        "quality": to_dict(bundle.quality),
        "clench": to_dict(bundle.clench),
        "arbiter": to_dict(bundle.arbiter),
        "recommended_scheme": bundle.recommended_scheme,
        "reliable_commands": list(bundle.reliable),
        "report": bundle.report,
    }
    save_json(directory / MANIFEST, manifest)
    return directory


def load_bundle(directory: str | Path) -> Bundle:
    directory = Path(directory)
    if not (directory / MANIFEST).exists() or not (directory / ARRAYS).exists():
        raise BundleError(f"{directory} is not a model bundle ({MANIFEST} and {ARRAYS} are needed)")
    manifest = load_json(directory / MANIFEST)
    if manifest.get("format") != FORMAT:
        raise BundleError(f"{directory}: unsupported bundle format {manifest.get('format')!r}")
    frontend = from_dict(FrontendConfig, manifest["frontend"])
    if frontend.fingerprint != manifest.get("frontend_fingerprint"):
        raise BundleError(
            f"{directory}: the stored front-end configuration hashes to {frontend.fingerprint}, the bundle says "
            f"{manifest.get('frontend_fingerprint')}. The front-end has changed since training: train again."
        )
    spec = from_dict(FeatureSpec, manifest["features"])
    with np.load(directory / ARRAYS) as arrays:
        try:
            model = IntentModel.from_arrays(arrays, manifest["classes"], spec, frontend.n_bins, frontend.n_channels)
        except (KeyError, ValueError) as error:
            raise BundleError(f"{directory}: {error}") from error
        produced = model.log_proba(arrays["selftest_cosp"], arrays["selftest_reference"])
        expected = arrays["selftest_log_proba"]
    worst = float(np.abs(produced - expected).max()) if expected.size else 0.0
    if not np.isfinite(worst) or worst > 1e-6:
        raise BundleError(f"{directory}: the model does not reproduce its self-test (largest difference {worst:.3g})")
    return Bundle(
        model,
        frontend,
        from_dict(QualityConfig, manifest.get("quality")),
        from_dict(ClenchConfig, manifest.get("clench")),
        from_dict(ArbiterConfig, manifest.get("arbiter")),
        manifest.get("recommended_scheme", "switch"),
        tuple(manifest.get("reliable_commands", ())),
        manifest.get("report", {}),
        directory,
    )
