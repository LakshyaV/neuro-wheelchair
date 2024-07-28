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
mind-controlled-wheelchair/
├── arduino/
│ └── wheelchair_control.ino
├── data/
│ └── eeg_data.csv
├── model/
│ └── train_model.py
│ └── model.joblib
├── src/
│ └── realtime_classifier.py
├── README.md


## Usage

1. Start the Muse headset and ensure it is connected to your computer.
2. Open a terminal and navigate to the project directory.
3. Run the real-time classifier:
    ```bash
    python src/realTime.py
    ```
4. The classifier will process the incoming EEG data and send commands to the Arduino to control the wheelchair.

## Technical Details

### Data Acquisition

The Muse headset captures EEG signals and sends them via OSC (Open Sound Control) to the Python script. The EEG data consists of multiple channels, each representing electrical activity in different regions of the brain.

### Data Preprocessing

The raw EEG data is preprocessed using the following steps:
- **Notch Filtering**: A notch filter is applied to remove powerline noise (50-60 Hz).
- **Standardization**: The data is standardized using a `StandardScaler` to have zero mean and unit variance.

### Epoching and Feature Extraction

The preprocessed data is segmented into overlapping epochs. For each epoch, features are extracted using the following steps:
- **Power Spectral Density (PSD) Calculation**: The PSD is computed using FFT (Fast Fourier Transform) to obtain the power in different frequency bands (delta, theta, alpha, beta).
- **Log Transformation**: The power values are log-transformed to stabilize variance.

### Machine Learning Model

The extracted features are used to train a machine learning model. The model is an SVM (Support Vector Machine) classifier trained to recognize different mental states corresponding to different wheelchair commands (e.g., move forward, turn left, turn right, stop).

### Real-Time Classification

The real-time classifier script processes incoming EEG data, extracts features, and uses the pre-trained SVM model to classify the data. The classification results are sent to the Arduino via serial communication.

### Arduino Control

The Arduino script (`wheelchair_control.ino`) receives the classification results and controls the wheelchair motors accordingly. The commands are mapped to specific motor actions to move the wheelchair in the desired direction.

## Contributing

We welcome contributions to improve the Mind-Controlled Wheelchair project. If you have any ideas, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
