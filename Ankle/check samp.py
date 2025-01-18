import pandas as pd

# Load your dataset
# Replace 'path_to_dataset.csv' with the actual path to your dataset file
data = pd.read_csv('B:\Fall_Classification\Ankle\Balanced_Data_With_Metadata_Final.csv')

# Count the number of samples for each label
label_counts = data['label'].value_counts()

# Print the counts
print("Number of samples for each label:")
print(label_counts)

