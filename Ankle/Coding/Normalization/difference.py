import pandas as pd
import matplotlib.pyplot as plt

# Load the raw and Z-score normalized datasets
raw_file_path = 'E:/Final year project/Fall Classificaiton/Fall_Classification/Ankle/Ankle_combined_data.csv'
zscore_file_path = 'E:/Final year project/Fall Classificaiton/Fall_Classification/Ankle/Zscore_Normalized_Data.csv'

raw_data = pd.read_csv(raw_file_path)
zscore_data = pd.read_csv(zscore_file_path)

# Define sensor columns
sensor_columns = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz']

# Statistical comparison
print("Raw Data Statistics:")
print(raw_data[sensor_columns].describe())

print("nZ-Score Normalized Data Statistics:")
print(zscore_data[sensor_columns].describe())

# Visualize the distributions before and after normalization
for col in sensor_columns:
    plt.figure(figsize=(12, 6))
    plt.hist(raw_data[col], bins=50, alpha=0.5, label='Raw', color='blue')
    plt.hist(zscore_data[col], bins=50, alpha=0.5, label='Z-score Normalized', color='orange')
    plt.title(f'Distribution of {col} (Raw vs Z-Score Normalized)')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Element-wise difference
differences = raw_data[sensor_columns] - zscore_data[sensor_columns]
print("/nElement-wise differences (first 5 rows):")
print(differences.head())
