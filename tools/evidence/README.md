# Evidence scripts

The measurements reported in [docs/evidence.md](../../docs/evidence.md), on
public data. Each script downloads what it needs on request and runs the code
of this package.

```bash
pip install -e ".[all]" -r tools/evidence/requirements.txt

python tools/evidence/muse_recordings.py --download          # real Muse recordings: decoders compared
python tools/evidence/muse_recordings.py --eegnet            # ...including a convolutional network (needs torch)
python tools/evidence/electrode_positions.py                 # what Muse-like positions can pick up (downloads ~800 MB)
python tools/evidence/one_recording_per_state.py --download  # one file per state, and what a classifier makes of it
python tools/evidence/resting_statistics.py                  # timing, amplitudes, stop-reflex false triggers
```

Data goes to `evidence-data/`, which git ignores. None of it is redistributed
with this repository; please respect the terms of each source.

| Script | Source |
| --- | --- |
| `muse_recordings.py` | EEG-ExPy example recordings (NeuroTechX), https://github.com/NeuroTechX/EEG-ExPy |
| `electrode_positions.py` | EEG Motor Movement/Imagery Database, https://physionet.org/content/eegmmidb/ |
| `one_recording_per_state.py` | https://github.com/jordan-bird/eeg-feature-generation |
| `resting_statistics.py` | the recordings fetched by the first and third script |
