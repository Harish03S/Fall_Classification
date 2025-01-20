import os
import pandas as pd

# Define the base directory containing the location data
base_dir = 'E:\Final year project\Fall Classificaiton\Fall_Classification\Thigh'  # Replace with the path to the "Ankle" directory

# Define activity folders
activities = ['Activities of daily living', 'Forward fall labeled', 'Backward fall labeled', 'Lateral fall labeled']

# Initialize an empty list to store all data
all_data = []

# Process each activity folder
for activity in activities:
    activity_path = os.path.join(base_dir, activity)
    if not os.path.isdir(activity_path):
        print(f"Skipping {activity_path}, not a directory.")
        continue

    # Process each file in the activity folder
    for file_name in os.listdir(activity_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(activity_path, file_name)

            # Extract metadata from the file name
            parts = file_name.split('_')
            if activity == 'Activities of daily living' and len(parts) >= 3:
                participant_id = parts[0]  # First part is participant ID
                trial_number = parts[-1].split('.')[0]  # Last part (before .csv) is trial number
            elif len(parts) >= 3:
                participant_id = parts[0]  # First part is participant ID
                trial_number = parts[-2]  # Second to last part is trial number
            else:
                print(f"Skipping file with unexpected format: {file_name}")
                continue

            # Read the CSV file
            try:
                data = pd.read_csv(file_path)

                # Select only the desired columns: the 9 axis data (ax, ay, az, wx, wy, wz, Bx, By, Bz) and the label
                selected_columns = ['time','ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz', 'label']
                data = data[selected_columns]

                # Add metadata columns
                data['Participant_ID'] = participant_id
                data['Trial_Number'] = trial_number

                # Append to the list of all data
                all_data.append(data)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Combine all data into a single DataFrame
if all_data:
    combined_data = pd.concat(all_data, ignore_index=True)

    # Save the combined data to a new CSV file
    output_file = os.path.join(base_dir, 'Thigh_combined_data.csv')
    combined_data.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
else:
    print("No data found to combine.")
