# Mind-Controlled Wheelchair

Drive a wheelchair from a Muse EEG headband: the headset's signal is decoded on
a laptop, and an Arduino drives the motors.

<img src="https://github.com/user-attachments/assets/281f2171-3195-4668-bf5a-cb52502bc661" alt="Mind-Controlled Wheelchair" width="330">

This repository is being rebuilt as an installable package with a tested signal
path, a safety supervisor and a motor controller that stops by itself. The plan
and its reasons are in [#1](https://github.com/LakshyaV/neuro-wheelchair/issues/1).

The 2024 prototype (band-power features, a linear SVM, single-digit serial
commands) is kept under the tag
[`v0.1.0`](https://github.com/LakshyaV/neuro-wheelchair/tree/v0.1.0).

## Development

```bash
pip install -e ".[all]"
make check
```

## License

MIT, see [LICENSE](LICENSE).
