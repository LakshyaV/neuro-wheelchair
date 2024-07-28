# Mind-Controlled Wheelchair

This project leverages EEG data from a Muse headset to control a wheelchair using machine learning classification. The EEG data is processed and classified in real-time, and the classification results are sent to an Arduino which controls the wheelchair movements.

## Table of Contents

- [Introduction](#introduction)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Mind-Controlled Wheelchair project is an innovative solution that uses EEG signals from a Muse headset to control a wheelchair. The system captures EEG data, processes it, and classifies it using a machine learning model. The classification results are sent to an Arduino, which interprets the signals to control the wheelchair's movements.

## Hardware Requirements

- Muse Headset
- Arduino (e.g., Arduino Uno)
- L298N Motor Driver
- DC Motors (connected to the wheelchair)
- Jumper Wires
- Breadboard
- USB Cable for Arduino

## Software Requirements

- Python 3.x
- Arduino IDE
- Muse SDK
- Python libraries: `pandas`, `numpy`, `scipy`, `scikit-learn`, `python-osc`, `joblib`

## Setup Instructions

### Arduino Setup

1. Connect the motors to the L298N motor driver.
2. Connect the L298N motor driver to the Arduino as follows:
   - IN1 to pin 2
   - IN2 to pin 3
   - IN3 to pin 4
   - IN4 to pin 5
3. Connect the L298N motor driver to the power supply and motors.
4. Upload the `wheelchair_control.ino` file to the Arduino.

### Python Environment Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mind-controlled-wheelchair.git
    cd mind-controlled-wheelchair
    ```

2. Install the required Python libraries:
    ```bash
    pip install pandas numpy scipy scikit-learn python-osc joblib
    ```

3. Train your machine learning model using the provided `train_model.py` script (ensure your EEG data is in the correct format) and save the model as `model.joblib`.

4. Run the `realtime_classifier.py` script to start classifying real-time EEG data and sending the commands to the Arduino.

### Directory Structure

