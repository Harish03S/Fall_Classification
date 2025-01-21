import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the input and output folder paths
input_folder = r'C:/Users/haris/Desktop/Fall_Classification/Waist/Lateral Fall Labelled'
output_folder = r'C:/Users/haris/Desktop/Fall_Classification/Waist/Lateral Fall Labelled/vis'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):  # Process only CSV files
        # Full path to the CSV file
        file_path = os.path.join(input_folder, file_name)
        
        # Load the CSV file
        data = pd.read_csv(file_path)
        
        # Create a figure and line plots for ax, ay, and az
        plt.figure(figsize=(15, 7))
        
        # Plot ax over time
        plt.plot(data['time'], data['ax'], label='ax (Acceleration X)', color='red')
        
        # Plot ay over time
        plt.plot(data['time'], data['ay'], label='ay (Acceleration Y)', color='blue')
        
        # Plot az over time
        plt.plot(data['time'], data['az'], label='az (Acceleration Z)', color='green')
        
        # Add labels, title, and legend
        plt.xlabel('Time (seconds)')
        plt.ylabel('Acceleration')
        plt.title(f'Line Plots of Accelerometer Values for {file_name}')
        plt.legend()
        
        # Add grid
        plt.grid()
        
        # Save the plot as an image in the output folder
        output_file_path = os.path.join(output_folder, f'{os.path.splitext(file_name)[0]}.png')
        plt.savefig(output_file_path)
        plt.close()  # Close the plot to save memory

print(f"Plots saved in: {output_folder}")
