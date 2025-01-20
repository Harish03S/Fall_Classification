import pandas as pd
from scipy.stats import zscore

# Load the dataset
file_path = 'E:/Final year project/Fall Classificaiton/Fall_Classification/Ankle/Ankle_combined_data.csv'
data = pd.read_csv(file_path)

# Define sensor columns
sensor_columns = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz']

# Apply Z-score normalization to the sensor columns
data[sensor_columns] = data[sensor_columns].apply(zscore)

# Save the normalized dataset to a new file
zscore_normalized_file_path = 'E:/Final year project/Fall Classificaiton/Fall_Classification/Ankle/Zscore_Normalized_Data.csv'
data.to_csv(zscore_normalized_file_path, index=False)

print(f"Z-score normalization complete. Normalized dataset saved to: {zscore_normalized_file_path}")
