import pandas as pd

# Load the pipe-separated data
data_file = '../data/MachineLearningRating_v3.txt'
df = pd.read_csv(data_file, sep='|')

# Convert TransactionMonth to datetime
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], format='%Y-%m', errors='coerce')

# Verify column data types
print("Column Data Types:")
print(df.dtypes)

# Initial data quality checks
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics for Numerical Columns:")
numerical_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate', 
                 'Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors']
print(df[numerical_cols].describe())

print("\nSample Data (First 5 Rows):")
print(df.head())

# Check for duplicates
print("\nNumber of Duplicate Rows:", df.duplicated().sum())

# Save cleaned data for further analysis
df.to_csv('../data/MachineLearningRating_v3_cleaned.csv', index=False)
print("Data saved to 'data/MachineLearningRating_v3_cleaned.csv'")

# Optional: Save a subset for quick inspection
df.head(100).to_csv('../data/MachineLearningRating_v3_sample.csv', index=False)
print("Sample data saved to 'data/MachineLearningRating_v3_sample.csv'")
