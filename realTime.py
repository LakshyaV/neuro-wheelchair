import argparse
import joblib
import numpy as np
import pandas as pd
import serial  # Import pySerial for serial communication
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from scipy.signal import butter, lfilter, lfilter_zi
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threading import Thread

# Load the pre-trained model
model = joblib.load('model.joblib')

# Define the notch filter
NOTCH_B, NOTCH_A = butter(4, np.array([55, 65]) / (256 / 2), btype='bandstop')

# Global data buffer to store incoming data
data_buffer = []

# Initialize serial communication
ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with your actual serial port

def preprocess_data(data, notch_b, notch_a, scaler):
    """Preprocess the data: apply notch filter and scaling."""
    filter_state = np.tile(lfilter_zi(notch_b, notch_a), (data.shape[1], 1)).T
    filtered_data, _ = lfilter(notch_b, notch_a, data, axis=0, zi=filter_state)
    return scaler.transform(filtered_data)

def epoch(data, samples_epoch, samples_overlap=0):
    """Extract epochs from a time series."""
    if isinstance(data, list):
        data = np.array(data)
    n_samples, n_channels = data.shape
    samples_shift = samples_epoch - samples_overlap
    n_epochs = int(np.floor((n_samples - samples_epoch) / float(samples_shift)) + 1)
    markers = np.asarray(range(0, n_epochs + 1)) * samples_shift
    markers = markers.astype(int)
    epochs = np.zeros((samples_epoch, n_channels, n_epochs))
    for i in range(0, n_epochs):
        epochs[:, :, i] = data[markers[i]:markers[i] + samples_epoch, :]
    return epochs

def compute_band_powers(eegdata, fs):
    """Extract the features (band powers) from the EEG."""
    winSampleLength, nbCh = eegdata.shape
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
    dataWinCenteredHam = (dataWinCentered.T * w).T
    NFFT = nextpow2(winSampleLength)
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
    PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])
    f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))
    ind_delta, = np.where(f < 4)
    meanDelta = np.mean(PSD[ind_delta, :], axis=0)
    ind_theta, = np.where((f >= 4) & (f <= 8))
    meanTheta = np.mean(PSD[ind_theta, :], axis=0)
    ind_alpha, = np.where((f >= 8) & (f <= 12))
    meanAlpha = np.mean(PSD[ind_alpha, :], axis=0)
    ind_beta, = np.where((f >= 12) & (f < 30))
    meanBeta = np.mean(PSD[ind_beta, :], axis=0)
    feature_vector = np.concatenate((meanDelta, meanTheta, meanAlpha, meanBeta), axis=0)
    feature_vector = np.log10(feature_vector)
    return feature_vector

def nextpow2(i):
    """Find the next power of 2 for number i."""
    n = 1
    while n < i:
        n *= 2
    return n

def compute_feature_matrix(epochs, fs):
    """Call compute_band_powers for each EEG epoch."""
    n_epochs = epochs.shape[2]
    for i_epoch in range(n_epochs):
        if i_epoch == 0:
            feat = compute_band_powers(epochs[:, :, i_epoch], fs).T
            feature_matrix = np.zeros((n_epochs, feat.shape[0]))
        feature_matrix[i_epoch, :] = compute_band_powers(epochs[:, :, i_epoch], fs).T
    return feature_matrix

def classify_data():
    global data_buffer
    while True:
        if data_buffer:
            data = np.array(data_buffer)
            # Preprocess data
            data = preprocess_data(data, NOTCH_B, NOTCH_A, scaler)
            # Epoch data
            epochs = epoch(data, samples_epoch, samples_overlap)
            features = compute_feature_matrix(epochs, fs)
            # Flatten features
            features = features.reshape((features.shape[2], features.shape[0] * features.shape[1]))
            # Use the model to predict
            predictions = model.predict(features)
            # Handle predictions (e.g., print, log, or take action)
            print(f"Predictions: {predictions}")
            for prediction in predictions:
                ser.write(str(prediction).encode())  # Send prediction to Arduino
            data_buffer.clear()

# Handler function to process incoming OSC messages
def print_petal_stream_handler(unused_addr, *args):
    sample_id = args[0]
    unix_ts = args[1] + args[2]
    lsl_ts = args[3] + args[4]
    data = args[5:]
    print(
        f'sample_id: {sample_id}, unix_ts: {unix_ts}, '
        f'lsl_ts: {lsl_ts}, data: {data}'
    )
    data_buffer.append(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ip', type=str, required=False,
                        default="127.0.0.1", help="The IP to listen on")
    parser.add_argument('-p', '--udp_port', type=str, required=False, default=14739,
                        help="The UDP port to listen on")
    parser.add_argument('-t', '--topic', type=str, required=False,
                        default='/PetalStream/eeg', help="The topic to print")
    args = parser.parse_args()

    dispatcher = Dispatcher()
    dispatcher.map(args.topic, print_petal_stream_handler)

    server = ThreadingOSCUDPServer(
        (args.ip, args.udp_port),
        dispatcher
    )

    print(f"Serving on {server.server_address}")

    # Start the server in a separate thread
    server_thread = Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    # Start classification in a separate thread
    classify_thread = Thread(target=classify_data)
    classify_thread.daemon = True
    classify_thread.start()

    # Keep the main thread running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        server.shutdown()
        ser.close()  # Close the serial connection

if __name__ == "__main__":
    # Preprocess the data using the same scaler as the training data
    scaler = StandardScaler()
    fs = 256  # Example sampling frequency, change according to your data
    samples_epoch = 256  # Example epoch length, change according to your needs
    samples_overlap = 128  # Example overlap, change according to your needs

    main()
