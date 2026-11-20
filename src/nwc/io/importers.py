"""Bring recordings made elsewhere into the session format.

Two layouts are understood:

*Marker recordings* (muselsl ``record``, EEG-ExPy): ``timestamps,<channels>,Marker0``
with a non-zero code on the sample where a stimulus or cue began. Each file
becomes one run of the session, each marker a cue of fixed length.

*Labelled recordings*: one label per sample in a column of its own, as produced
by appending a class column to a recording. Each stretch of equal labels becomes
a cue. This is how the 2024 prototype stored its training data, usually with one
file per command; such data can be imported and trained on, but evaluation will
say that command and recording cannot be told apart in it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .session import CUE, RUN, Session, SessionWriter
from .sources import MUSE_CHANNELS
from .stream import Chunk, StreamInfo


def _read_table(path: Path) -> tuple[list[str] | None, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().strip()
    cells = [c.strip() for c in first.split(",")]
    try:
        [float(c) for c in cells]
        header = None
    except ValueError:
        header = cells
    data = np.genfromtxt(path, delimiter=",", skip_header=0 if header is None else 1)
    if data.ndim == 1:
        data = data[None, :]
    return header, data


def _after(stamps: np.ndarray, last_t: float | None) -> np.ndarray:
    """Shift a file's timestamps, if necessary, so that it starts clearly after the previous file ended.

    The gap makes the assembler see a break between files, and the filters restart there.
    """
    if last_t is not None and stamps[0] < last_t + 1.0:
        return stamps + (last_t + 1.0 - stamps[0])
    return stamps


def _columns(
    header: list[str] | None, data: np.ndarray, channels: Sequence[str], channel_columns: Sequence[int] | None
) -> list[int]:
    if channel_columns is not None:
        return list(channel_columns)
    if header is not None and all(name in header for name in channels):
        return [header.index(name) for name in channels]
    if data.shape[1] >= len(channels) + 1:
        return list(range(1, len(channels) + 1))  # timestamps first, then the channels in Muse order
    raise ValueError(f"cannot find {len(channels)} channel columns; pass channel_columns")


def import_marker_recordings(
    paths: Sequence[str | Path],
    directory: str | Path,
    *,
    markers: dict[int, str],
    cue_s: float,
    subject: str,
    session: str,
    channels: Sequence[str] = MUSE_CHANNELS,
    fs: float = 256.0,
    marker_column: str = "Marker0",
) -> Session:
    """Files with a marker column become the runs of one session."""
    info = StreamInfo(tuple(channels), float(fs), source="import", device="muselsl recording")
    meta = {"imported": [str(p) for p in paths], "markers": {str(k): v for k, v in markers.items()}}
    trial = 0
    last_t: float | None = None
    with SessionWriter(directory, info, subject=subject, session=session, meta=meta) as writer:
        for run, path in enumerate(paths, start=1):
            header, data = _read_table(Path(path))
            if header is None or marker_column not in header:
                raise ValueError(f"{path}: no column named {marker_column!r}")
            columns = _columns(header, data, channels, None)
            marker = np.nan_to_num(data[:, header.index(marker_column)]).astype(np.int64)
            first = writer.samples
            stamps = _after(data[:, 0], last_t)
            last_t = float(stamps[-1])
            writer.write(Chunk(stamps, data[:, columns]))
            length = round(cue_s * fs)
            for onset in np.flatnonzero(marker != 0):
                code = int(marker[onset])
                if code not in markers or onset + length > len(data):
                    continue
                writer.event(CUE, first + onset, first + onset + length, markers[code], run, trial)
                trial += 1
            writer.event(RUN, first, writer.samples, "", run)
    return Session(directory)


def import_labelled_recordings(
    paths: Sequence[str | Path],
    directory: str | Path,
    *,
    subject: str,
    session: str,
    names: dict[str, str] | None = None,
    label_column: int = -1,
    time_column: int | None = 0,
    channel_columns: Sequence[int] | None = None,
    channels: Sequence[str] = MUSE_CHANNELS,
    fs: float = 256.0,
    min_cue_s: float = 3.0,
    chunk_s: float = 5.0,
) -> Session:
    """Files with one label per sample become the runs of one session.

    ``names`` maps the values found in the label column (as text, e.g. ``"1"``) to
    command names; values without an entry keep their text. Stretches longer than
    ``chunk_s`` are cut into cues of that length; pieces shorter than ``min_cue_s``
    are too short to hold an analysis window and are skipped.
    """
    info = StreamInfo(tuple(channels), float(fs), source="import", device="labelled recording")
    names = names or {}
    trial = 0
    last_t: float | None = None
    with SessionWriter(
        directory, info, subject=subject, session=session, meta={"imported": [str(p) for p in paths]}
    ) as writer:
        for run, path in enumerate(paths, start=1):
            header, data = _read_table(Path(path))
            width = data.shape[1]
            label_at = label_column % width
            if channel_columns is None:
                start = 0 if time_column is None else 1
                candidates = [c for c in range(start, width) if c != label_at]
                if header is not None and all(name in header for name in channels):
                    columns = [header.index(name) for name in channels]
                elif len(candidates) >= len(channels):
                    columns = candidates[-len(channels) :]  # the columns just before the label
                else:
                    raise ValueError(f"{path}: cannot find {len(channels)} channel columns; pass channel_columns")
            else:
                columns = list(channel_columns)
            stamps = _after(data[:, time_column] if time_column is not None else np.arange(len(data)) / fs, last_t)
            last_t = float(stamps[-1])
            labels = data[:, label_at]
            first = writer.samples
            writer.write(Chunk(stamps, data[:, columns]))
            edges = np.flatnonzero(np.diff(labels) != 0) + 1
            piece = round(chunk_s * fs)
            for a, b in zip(np.concatenate(([0], edges)), np.concatenate((edges, [len(labels)]))):
                if not np.isfinite(labels[a]):
                    continue
                text = f"{labels[a]:g}"
                # A long stretch of one label is cut into cue-sized pieces so that there are trials to count.
                for start in range(int(a), int(b), piece):
                    stop = min(start + piece, int(b))
                    if stop - start < min_cue_s * fs:
                        continue
                    writer.event(CUE, first + start, first + stop, names.get(text, text), run, trial)
                    trial += 1
            writer.event(RUN, first, writer.samples, "", run)
    return Session(directory)
