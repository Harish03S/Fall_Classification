import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'B:\Fall_Classification\Ankle\Balanced_Data_With_Metadata_Final.csv'
dataset = pd.read_csv(file_path)

# Columns to normalize
sensor_columns = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz']

# Initialize the Min-Max Scaler
scaler = MinMaxScaler()

# Apply normalization to the sensor columns
dataset[sensor_columns] = scaler.fit_transform(dataset[sensor_columns])

# Save the normalized dataset to a new file (optional)
normalized_file_path = 'B:\Fall_Classification\Ankle\\Normalized_Ankle_Data.csv'
dataset.to_csv(normalized_file_path, index=False)

print("Normalization complete. Normalized dataset saved to:", normalized_file_path)
