import pandas as pd
import numpy as np

# Load dataset
file_path = 'E:/Final year project/Fall Classificaiton/Fall_Classification/Ankle/Balanced_Data_With_Metadata_Final.csv'
data = pd.read_csv(file_path)

# Compute magnitude vectors
data['accel_magnitude'] = np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)
data['gyro_magnitude'] = np.sqrt(data['wx']**2 + data['wy']**2 + data['wz']**2)
data['mag_magnitude'] = np.sqrt(data['Bx']**2 + data['By']**2 + data['Bz']**2)

# Save the updated dataset
updated_file_path = 'E:/Final year project/Fall Classificaiton/Fall_Classification/Ankle/Updated_Data_with_Magnitudes.csv'
data.to_csv(updated_file_path, index=False)

print(f"Dataset updated with magnitude vectors. Saved to: {updated_file_path}")
