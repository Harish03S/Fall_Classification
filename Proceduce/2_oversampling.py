from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the dataset
file_path = 'E:\Final year project\Fall Classificaiton\Fall_Classification\Thigh\Thigh_combined_data.csv'
data = pd.read_csv(file_path)

# Define sensor columns, label, and metadata
sensor_columns = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz']
label_column = 'label'
metadata_columns = ['Participant_ID', 'Trial_Number']

# Separate features, labels, and metadata
X = data[sensor_columns]
y = data[label_column]
metadata = data[metadata_columns].reset_index(drop=True)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Identify synthetic samples
num_original_samples = len(y)
num_synthetic_samples = len(y_resampled) - num_original_samples

# Assign metadata to synthetic samples
synthetic_metadata_indices = np.random.choice(metadata.index, size=num_synthetic_samples, replace=True)
synthetic_metadata = metadata.iloc[synthetic_metadata_indices].reset_index(drop=True)

# Combine original and synthetic metadata
metadata_resampled = pd.concat([metadata, synthetic_metadata], ignore_index=True).reset_index(drop=True)

# Combine metadata with oversampled data
oversampled_data = pd.DataFrame(X_resampled, columns=sensor_columns)
oversampled_data[label_column] = y_resampled
oversampled_data = pd.concat([oversampled_data, metadata_resampled], axis=1)



# Save the balanced and normalized dataset
oversampled_file_path = 'E:\Final year project\Fall Classificaiton\Fall_Classification\Thigh\Balanced_Thigh_combined_data.csv'
oversampled_data.to_csv(oversampled_file_path, index=False)

print("Oversampling and normalization complete with metadata assigned. Data saved to:", oversampled_file_path)
