import os
import pandas as pd
import matplotlib.pyplot as plt

# Input and output folder paths
input_folder = r'C:/Users/haris/Desktop/Fall_Classification/Waist/Lateral Fall Labelled'
output_folder = r'C:/Users/haris/Desktop/Fall_Classification/Waist/Lateral Fall Labelled/magvis'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all CSV files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):  # Process only CSV files
        file_path = os.path.join(input_folder, file_name)
        
        # Load the CSV file
        data = pd.read_csv(file_path)
        
        # Check if the required columns are present
        required_columns = ['time', 'Bx', 'By', 'Bz']  # Assuming 'time' column exists
        if not all(col in data.columns for col in required_columns):
            print(f"Skipping file {file_name}: Missing required columns")
            continue

        # Create a line plot for Bx, By, Bz over time
        plt.figure(figsize=(15, 7))
        
        # Plot Bx
        plt.plot(data['time'], data['Bx'], label='Bx (Magnetometer X)', color='red')
        
        # Plot By
        plt.plot(data['time'], data['By'], label='By (Magnetometer Y)', color='blue')
        
        # Plot Bz
        plt.plot(data['time'], data['Bz'], label='Bz (Magnetometer Z)', color='green')

        # Add labels, title, and legend
        plt.xlabel('Time (seconds)')
        plt.ylabel('Magnetometer Values')
        plt.title(f'Magnetometer Line Plot for {file_name}')
        plt.legend()

        # Add grid
        plt.grid()

        # Save the plot as a PNG file
        output_file_path = os.path.join(output_folder, f'{os.path.splitext(file_name)[0]}_magnetometer_line.png')
        plt.savefig(output_file_path)
        plt.close()  # Close the plot to save memory

        print(f"Line plot saved for {file_name} at {output_file_path}")

print(f"All plots saved in: {output_folder}")
