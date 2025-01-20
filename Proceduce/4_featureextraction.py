import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft

# Feature extraction function
def extract_features(segment, axis):
    """Extracts 26 features from a single axis segment of accelerometer, gyroscope, or magnetometer data."""

    # Basic Statistical Features
    min_val = segment[axis].min()
    max_val = segment[axis].max()
    mean_val = segment[axis].mean()
    skewness = skew(segment[axis].dropna())
    kurtosis_val = kurtosis(segment[axis].dropna())

    # Autocorrelation Features (lags 1-11)
    autocorr_vals = [segment[axis].autocorr(lag=i) for i in range(1, 12)]

    # Frequency Domain Features (Top 5 frequencies and their amplitudes)
    segment_data = segment[axis].fillna(0).to_numpy()
    freq_data = np.abs(fft(segment_data))
    sorted_indices = np.argsort(freq_data)[::-1]  # Sort frequencies by magnitude, descending
    top_freqs = sorted_indices[:5]
    top_amplitudes = freq_data[top_freqs]

    features = {
        f'{axis}_min': min_val,
        f'{axis}_max': max_val,
        f'{axis}_mean': mean_val,
        f'{axis}_skewness': skewness,
        f'{axis}_kurtosis': kurtosis_val,
    }

    # Add autocorrelation features
    for i, autocorr_val in enumerate(autocorr_vals, start=1):
        features[f'{axis}_autocorr_lag_{i}'] = autocorr_val

    # Add frequency and amplitude features
    for i, (freq, amp) in enumerate(zip(top_freqs, top_amplitudes), start=1):
        features[f'{axis}_freq_{i}'] = freq
        features[f'{axis}_amplitude_{i}'] = amp

    return pd.Series(features)

# Grouped feature extraction function
def extract_features_grouped(data, sensor_columns, window_size, step_size):
    """Extracts features grouped by Participant_ID and Trial_Number."""
    features = []
    labels = []

    # Group by Participant ID and Trial Number
    grouped = data.groupby(['Participant_ID', 'Trial_Number'])

    for (participant, trial), group in grouped:
        group = group.reset_index(drop=True)  # Reset index for proper windowing

        # Sliding window feature extraction
        for start in range(0, len(group) - window_size + 1, step_size):
            window = group.iloc[start:start + window_size]
            feature_row = {}

            # Extract features for each sensor axis
            for axis in sensor_columns:
                axis_features = extract_features(window, axis)
                feature_row.update(axis_features)

            # Add metadata
            feature_row['Participant_ID'] = participant
            feature_row['Trial_Number'] = trial

            # Majority label in the window
            feature_row['label'] = window['label'].mode()[0]
            features.append(feature_row)

    # Convert features to DataFrame
    return pd.DataFrame(features)

# Parameters
window_size = 60  # 1 second
step_size = 30    # 50% overlap
sensor_columns = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz']

# Load dataset
file_path = 'E:\Final year project\Fall Classificaiton\Fall_Classification\Thigh\z_score_norm.csv'
data = pd.read_csv(file_path)

# Extract features
extracted_features = extract_features_grouped(data, sensor_columns, window_size, step_size)

# Save to CSV
output_file = 'E:\Final year project\Fall Classificaiton\Fall_Classification\Thigh\Extracted_Features_thigh.csv'
extracted_features.to_csv(output_file, index=False)
print(f"Feature extraction complete. Features saved to: {output_file}")
