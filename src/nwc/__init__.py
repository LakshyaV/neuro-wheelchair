"""neuro-wheelchair: drive a wheelchair from a Muse EEG headband.

The package is organised along the signal path::

    io           sources (OSC, LSL, replay, simulator), stream assembly, sessions on disk
    dsp          causal front-end: cross-spectral frames, signal quality, the clench detector
    calibration  the cued protocol and the guided recorder
    decode       features, shrinkage LDA, evaluation by held-out runs, the model bundle
    control      evidence arbiter, driving schemes, safety supervisor, threshold tuning
    link         checksummed serial protocol and the link to the motor controller
    runtime      the engine that connects all of the above, and the driving loop
    sim          synthetic EEG, a virtual controller and chair for hardware-free runs

numpy, scipy and pyyaml are all it needs. The stream sources and the serial link
bring their own small libraries as optional extras.
"""

__version__ = "0.2.0"
