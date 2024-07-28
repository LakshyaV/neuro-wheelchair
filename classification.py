import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.signal import butter, lfilter, lfilter_zi

# Define the notch filter
NOTCH_B, NOTCH_A = butter(4, np.array([55, 65]) / (256 / 2), btype='bandstop')

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

# Load the data from multiple CSV files
filenames = ['brain_data1.csv', 'brain_data2.csv', 'brain_data3.csv', 'brain_data4.csv']
dataframes = [pd.read_csv(filename) for filename in filenames]

# Concatenate all dataframes
data = pd.concat(dataframes, ignore_index=True)

# Assume the last column is the label
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Preprocess the data
def apply_notch_filter(data, notch_b, notch_a):
    filter_state = np.tile(lfilter_zi(notch_b, notch_a), (data.shape[1], 1)).T
    filtered_data, _ = lfilter(notch_b, notch_a, data, axis=0, zi=filter_state)
    return filtered_data

X = apply_notch_filter(X, NOTCH_B, NOTCH_A)

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Parameters for epoching
fs = 256  # Example sampling frequency, change according to your data
samples_epoch = 256  # Example epoch length, change according to your needs
samples_overlap = 128  # Example overlap, change according to your needs

# Epoch the data
epochs = epoch(X, samples_epoch, samples_overlap)
features = compute_feature_matrix(epochs, fs)

# Flatten features and corresponding labels
features = features.reshape((features.shape[2], features.shape[0] * features.shape[1]))
labels = np.repeat(y[:features.shape[0]], features.shape[1])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Train a SVM classifier
clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
