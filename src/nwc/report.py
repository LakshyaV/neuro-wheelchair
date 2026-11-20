"""Plain-text reports: what training found, in the order someone deciding whether to drive needs it."""

from __future__ import annotations

from .decode.bundle import Bundle
from .decode.dataset import Recording


def _percent(value: float) -> str:
    return f"{100 * value:.0f} %"


def recording_line(rec: Recording, frame_rate: float) -> str:
    stats = rec.stats
    minutes = len(rec) / frame_rate / 60.0
    usable = float(rec.quality.usable.mean()) if len(rec) else 0.0
    line = f"{rec.key}: {minutes:.1f} min, {stats.delivered_hz:.1f} samples/s"
    if stats.loss > 0.005:
        line += f", {_percent(stats.loss)} lost"
    if stats.breaks:
        line += f", {stats.breaks} breaks"
    return line + f", {_percent(usable)} of frames usable"


def confusion_table(classes: list[str], matrix: list[list[int]]) -> list[str]:
    width = max(8, *(len(c) for c in classes)) + 1
    head = " " * (width + 2) + "".join(f"{c:>{width}}" for c in classes)
    rows = [head]
    for name, row in zip(classes, matrix):
        rows.append(f"  {name:<{width}}" + "".join(f"{v:>{width}d}" for v in row))
    return rows


def training_report(bundle: Bundle, recordings: list[Recording] | None = None) -> str:
    report = bundle.report
    ev, tuning = report.get("evaluation", {}), report.get("tuning", {})
    replay = tuning.get("replay", {})
    lines: list[str] = []
    if recordings:
        lines.append("recordings")
        lines += [f"  {recording_line(rec, bundle.frontend.frame_rate)}" for rec in recordings]
    trials = report.get("trials", {})
    lines.append(f"trials      {', '.join(f'{k} {v}' for k, v in trials.items())} in {ev.get('runs', '?')} runs")

    accuracy, chance, p = ev.get("balanced_accuracy", 0.0), ev.get("chance", 0.0), ev.get("p_value")
    lines.append("decoding")
    line = f"  {_percent(accuracy)} of trials on held-out runs, where guessing gets {_percent(chance)}"
    if p is not None:
        line += f" (p = {p:.3f}, {ev.get('permutations', 0)} shuffles)"
    lines.append(line)
    lines.append("  per command: " + ", ".join(f"{k} {_percent(v)}" for k, v in ev.get("recall", {}).items()))
    if ev.get("confusion"):
        lines.append("  asked (rows) against decoded (columns):")
        lines += ["  " + row for row in confusion_table(ev["classes"], ev["confusion"])]
    if ev.get("selected"):
        lines.append(
            "  configuration chosen per fold: " + ", ".join(f"{k} x{v}" for k, v in ev.get("chosen", {}).items())
        )

    warnings = []
    if ev.get("confounded"):
        warnings.append(
            "every run holds a single command, so command and recording cannot be told apart: this score may reflect the recordings, not the wearer. Record with `nwc record`, which interleaves commands."
        )
    elif ev.get("weak_grouping"):
        warnings.append(
            "only one run: blocks of it were held out instead, which is weaker evidence. Record at least two runs."
        )
    if p is not None and p > 0.05:
        warnings.append(
            "shuffled labels do about as well: there is no evidence that these commands can be told apart from this data."
        )
    if trials and min(trials.values()) < 12:
        warnings.append("fewer than 12 trials for a command: estimates are rough.")

    lines.append("driving     held-out runs replayed through the controller")
    if replay.get("trials"):
        latency = replay.get("median_latency_s")
        lines.append(
            f"  commands switched on in {_percent(replay['hit_rate'])} of trials"
            + (f", {latency:.1f} s after the cue (median)" if latency is not None else "")
            + f"; a wrong command in {_percent(replay['wrong_rate'])}; {replay['false_starts_per_min']:.2f} false starts per minute of rest"
        )
        lines.append(
            "  per command: "
            + ", ".join(f"{k} {_percent(v['hit_rate'])}" for k, v in replay.get("per_command", {}).items())
        )
        if not tuning.get("within_target", False):
            warnings.append(
                f"no controller setting kept false starts below {tuning.get('target_false_starts_per_min')} per minute with few wrong commands."
            )
    else:
        lines.append("  not available for this data (it needs runs with pauses)")
    reliable = report.get("tuning", {}).get("reliable_commands", [])
    lines.append(f"  reliable commands: {', '.join(reliable) if reliable else 'none'}")
    lines.append(
        f"scheme      {bundle.recommended_scheme}"
        + {"direct": ": the reliable commands drive while held", "switch": ": the jaw clench selects and stops"}.get(
            bundle.recommended_scheme, ": that command selects, the jaw clench is the emergency stop"
        )
    )
    clench = report.get("clench", {})
    if clench:
        lines.append(
            f"stop reflex threshold {clench['on_db']:.1f} dB above rest (rest stayed below {clench['rest_p99_9_db']:.1f} dB, calibration clenches above {clench['clench_p20_db']:.1f} dB)"
        )
        if clench.get("margin_db", 0.0) < 6.0:
            warnings.append(
                "the calibration clenches were hardly stronger than rest: clench harder, or check the fit of the headband behind the ears."
            )
    else:
        lines.append(
            f"stop reflex default threshold {bundle.clench.on_db:.0f} dB above rest (no calibration clenches in the data)"
        )
    for warning in warnings:
        lines.append(f"warning     {warning}")
    return "\n".join(lines)
