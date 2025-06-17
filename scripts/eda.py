import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from uuid import uuid4

# Set plot style
plt.style.use('seaborn')

# Load data
df = pd.read_csv('data/insurance_data.csv')
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], format='%Y-%m')

# Data Summarization
print("Descriptive Statistics:")
print(df[['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate']].describe())

# Data Quality Assessment
print("\nMissing Values:")
print(df.isnull().sum())

# Remove or impute missing values (example: drop rows with missing critical fields)
df = df.dropna(subset=['TotalPremium', 'TotalClaims', 'Province', 'Gender'])

# Univariate Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['TotalClaims'], bins=50, kde=True)
plt.title('Distribution of Total Claims')
plt.xlabel('Total Claims (Rand)')
plt.savefig('reports/total_claims_histogram.png')
plt.close()

# Bivariate Analysis: Loss Ratio by Province
df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
plt.figure(figsize = (12, 6))
sns.barplot(x='Province', y='LossRatio', data=df)
plt.title('Loss Ratio by Province')
plt.xticks(rotation=45)
plt.savefig('reports/loss_ratio_province.png')
plt.close()

# Claim Frequency by Vehicle Type
claim_freq = df.groupby('VehicleType')['TotalClaims'].apply(lambda x: (x > 0).mean()).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='VehicleType', y='TotalClaims', data=claim_freq)
plt.title('Claim Frequency by Vehicle Type')
plt.xticks(rotation=45)
plt.savefig('reports/claim_freq_vehicle_type.png')
plt.close()

# Temporal Trend of Total Claims
monthly_claims = df.groupby('TransactionMonth')['TotalClaims'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x='TransactionMonth', y='TotalClaims', data=monthly_claims)
plt.title('Total Claims Over Time')
plt.savefig('reports/total_claims_trend.png')
plt.close()

# Outlier Detection
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['TotalClaims'])
plt.title('Box Plot of Total Claims')
plt.savefig('reports/total_claims_boxplot.png')
plt.close()

# Correlation Matrix
corr = df[['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('reports/correlation_matrix.png')
plt.close()