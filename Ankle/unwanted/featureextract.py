import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft

# Load the normalized dataset
file_path = 'B:\Fall_Classification\Ankle\\Normalized_Ankle_Data.csv'
data = pd.read_csv(file_path)

# Define sensor columns
sensor_columns = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz']

# Updated parameters for smaller window and step size
window_size = 60   # 1 second of data
step_size = 30     # 50% overlap

# Function to extract features with grouping
def extract_features_grouped(data, sensor_columns, window_size, step_size):
    features = []
    labels = []

    # Group by Participant ID and Trial Number
    grouped = data.groupby(['Participant_ID', 'Trial_Number'])

    for (participant, trial), group in grouped:
        group = group.reset_index(drop=True)  # Reset index for proper windowing

        for start in range(0, len(group) - window_size + 1, step_size):
            window = group.iloc[start:start + window_size]
            feature_row = {}

            # Extract features for each sensor column
            for col in sensor_columns:
                signal = window[col].to_numpy()  # Convert to NumPy array for FFT compatibility

                # Time-domain features
                feature_row[f'{col}_max'] = np.max(signal)
                feature_row[f'{col}_min'] = np.min(signal)
                feature_row[f'{col}_std'] = np.std(signal)
                feature_row[f'{col}_sum_abs'] = np.sum(np.abs(signal))
                feature_row[f'{col}_rms'] = np.sqrt(np.mean(signal**2))
                feature_row[f'{col}_mean'] = np.mean(signal)
                feature_row[f'{col}_range'] = np.max(signal) - np.min(signal)
                feature_row[f'{col}_max_diff_consec'] = np.max(np.abs(np.diff(signal)))

                # Frequency-domain features
                fft_values = np.abs(fft(signal))[:len(signal) // 2]  # Only positive frequencies
                feature_row[f'{col}_fft_max'] = np.max(fft_values)
                feature_row[f'{col}_fft_min'] = np.min(fft_values)
                feature_row[f'{col}_fft_std'] = np.std(fft_values)
                feature_row[f'{col}_fft_sum_abs'] = np.sum(np.abs(fft_values))
                feature_row[f'{col}_fft_rms'] = np.sqrt(np.mean(fft_values**2))
                feature_row[f'{col}_fft_kurtosis'] = kurtosis(fft_values)
                feature_row[f'{col}_fft_skew'] = skew(fft_values)
                feature_row[f'{col}_fft_mean'] = np.mean(fft_values)
                feature_row[f'{col}_fft_range'] = np.max(fft_values) - np.min(fft_values)
                feature_row[f'{col}_fft_max_diff_consec'] = np.max(np.abs(np.diff(fft_values)))

            # Add participant and trial information
            feature_row['Participant_ID'] = participant
            feature_row['Trial_Number'] = trial

            # Majority label in the window
            labels.append(window['label'].mode()[0])
            features.append(feature_row)

    # Convert to DataFrame
    features_df = pd.DataFrame(features)
    features_df['label'] = labels
    return features_df

# Extract features using the updated parameters
extracted_features = extract_features_grouped(data, sensor_columns, window_size, step_size)

# Save the extracted features to a CSV file
extracted_features_file_path = 'B:/Fall_Classification/Ankle/Extracted_Features_10k.csv'
extracted_features.to_csv(extracted_features_file_path, index=False)

print("Feature extraction complete with smaller window and step size. Features saved to:", extracted_features_file_path)
