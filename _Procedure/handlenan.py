import pandas as pd

# Load the extracted features dataset
file_path = 'B:\Fall_Classification\Chest\Chest_featureextracted_data.csv'
data = pd.read_csv(file_path)

# Check for missing values
missing_summary = data.isnull().sum()
print("Missing Values per Column:")
print(missing_summary[missing_summary > 0])

# Option 1: Impute Missing Values (Replace NaN with Column Mean)
#
# Option 2: Drop Rows with Missing Values (Uncomment to use this method)
data_imputed = data.dropna()

# Verify no missing values remain
print("\nAfter Handling Missing Values:")
print(data_imputed.isnull().sum().sum())  # Should output 0

# Save the cleaned dataset
cleaned_file_path = 'B:\Fall_Classification\Chest\Chest_featureextracted_data_clean.csv'
data_imputed.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to: {cleaned_file_path}")
