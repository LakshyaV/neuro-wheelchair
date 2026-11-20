"""Thresholds tuned by replaying held-out runs through the arbiter."""

from __future__ import annotations

import numpy as np

from nwc.control.arbiter import ArbiterConfig
from nwc.control.tune import ReplayScore, score_replay, tune
from nwc.dsp.frontend import FrontendConfig


def test_tuning_report_and_recommendation(trained):
    tuning, classes = trained.tuning, trained.bundle.classes
    assert tuning.within_target and tuning.score.false_per_min <= 0.5 and tuning.score.wrong_rate <= 0.15
    assert tuning.score.trials == sum(c["trials"] for c in tuning.score.per_command.values()) > 30
    assert tuning.score.hit_rate > 0.5 and 1.0 < tuning.score.latency_s < 5.0
    assert set(tuning.reliable) <= set(classes[1:]) and tuning.recommended_scheme in (
        "direct",
        "switch",
        *[f"switch:{c}" for c in classes[1:]],
    )
    assert (len(tuning.reliable) >= 2) == (tuning.recommended_scheme == "direct")
    assert len(tuning.table) == 90 and tuning.summary()["arbiter"]["on"] == tuning.config.on
    # a more demanding setting makes commands rarer and slower
    from nwc.decode.train import fit_final

    held_out = fit_final(trained.corpus).held_out
    strict = score_replay(
        trained.corpus.recordings, held_out, classes, ArbiterConfig(on=0.99, off=0.6, dwell_s=1.0), FrontendConfig()
    )
    lax = score_replay(
        trained.corpus.recordings, held_out, classes, ArbiterConfig(on=0.5, off=0.3, dwell_s=0.25), FrontendConfig()
    )
    assert strict.hits < lax.hits and strict.latency_s > lax.latency_s  # fewer commands, and later
    again = tune(
        trained.corpus.recordings,
        held_out,
        classes,
        FrontendConfig(),
        target_false_per_min=0.0,
        max_wrong_rate=0.0,
        min_hit_rate=1.01,
    )
    assert again.reliable == () and again.recommended_scheme == "switch"
    empty = ReplayScore()
    assert (
        empty.hit_rate == 0.0
        and empty.false_per_min == 0.0
        and np.isnan(empty.latency_s)
        and empty.summary()["median_latency_s"] is None
    )
