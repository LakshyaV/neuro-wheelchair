# Mind-Controlled Wheelchair

This project leverages EEG data from a Muse headset to control a wheelchair using machine learning classification. The EEG data is processed and classified in real-time, and the classification results are sent to an Arduino which controls the wheelchair movements.

## Table of Contents

- [Introduction](#introduction)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Technical Details](#technical-details)
  - [Data Acquisition](#data-acquisition)
  - [Data Preprocessing](#data-preprocessing)
  - [Epoching and Feature Extraction](#epoching-and-feature-extraction)
  - [Machine Learning Model](#machine-learning-model)
  - [Real-Time Classification](#real-time-classification)
  - [Arduino Control](#arduino-control)
- [Machine Learning Mathematical Details](#machine-learning-mathematical-details)
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
4. Upload the `Arduino.ino` file to the Arduino.

### Python Environment Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/LakshyaV/neuro-wheelchair.git
    cd mind-controlled-wheelchair
    ```

2. Install the required Python libraries:
    ```bash
    pip install pandas numpy scipy scikit-learn python-osc joblib
    ```

3. Train your machine learning model using the provided `classification.py` script (ensure your EEG data is in the correct format) and save the model as `model.joblib`.

4. Run the `realTime.py` script to start classifying real-time EEG data and sending the commands to the Arduino.


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

## Machine Learning Mathematical Details

### Notch Filter

A notch filter is designed to remove specific frequency components from the signal. In this case, a 4th order Butterworth filter is used to remove powerline noise:
\[ \text{NOTCH\_B}, \text{NOTCH\_A} = \text{butter}(4, \left[\frac{55}{128}, \frac{65}{128}\right], \text{btype}='bandstop') \]

### Fast Fourier Transform (FFT)

The power spectral density (PSD) is calculated using FFT. The FFT converts the time-domain signal into the frequency domain:
\[ Y = \text{FFT}(x, N) \]
where \( N \) is the number of points.

### Power Spectral Density (PSD)

The PSD is used to find the power of different frequency bands:
\[ \text{PSD} = 2 \times \left| Y \right|^2 \]
The frequency bands are then averaged to extract features:
- Delta (0.5-4 Hz)
- Theta (4-8 Hz)
- Alpha (8-12 Hz)
- Beta (12-30 Hz)

### Log Transformation

To stabilize the variance, the power values are log-transformed:
\[ \text{Feature Vector} = \log_{10}(\text{Power Values}) \]

### Support Vector Machine (SVM)

The SVM model is trained using these features. The objective is to find a hyperplane that separates the data into classes:
\[ \text{minimize} \quad \frac{1}{2} \left\| w \right\|^2 \]
subject to:
\[ y_i (w \cdot x_i + b) \geq 1 - \xi_i \]
where \( w \) is the weight vector, \( x_i \) are the features, \( y_i \) are the labels, \( b \) is the bias, and \( \xi_i \) are slack variables.

### Real-Time Classification

For real-time classification, the preprocessed data is segmented into epochs, features are extracted, and the SVM model predicts the class. The results are sent to the Arduino to control the wheelchair.

## Contributing

We welcome contributions to improve the Mind-Controlled Wheelchair project. If you have any ideas, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
