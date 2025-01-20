import os
import pandas as pd

# Define the base directory containing the location data
base_dir = 'E:/Final year project/Fall Classificaiton/Fall_Classification/Thigh'

# Directory for "Activities of daily living"
adl_dir = os.path.join(base_dir, 'Activities of daily living')

# Check if the directory exists
if not os.path.isdir(adl_dir):
    print(f"{adl_dir} does not exist or is not a directory.")
else:
    # Process each file in the "Activities of daily living" directory
    for file_name in os.listdir(adl_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(adl_dir, file_name)

            try:
                # Read the CSV file
                data = pd.read_csv(file_path)

                # Add the "adl" column
                data['label'] = 'adl'

                # Save the updated file (overwrite the existing file or save as a new file)
                data.to_csv(file_path, index=False)
                print(f"Updated file: {file_path} with 'adl' column.")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
