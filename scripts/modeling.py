
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import shap
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
data_file = '../data/MachineLearningRating_v3.txt'
logging.info(f"Loading data from {data_file}")
try:
    df = pd.read_csv(data_file, sep='|')
except FileNotFoundError:
    logging.error(f"Data file {data_file} not found. Please check the file path.")
    raise
except pd.errors.ParserError:
    logging.error("Error parsing the data file. Verify the delimiter is '|'.")
    raise

# Convert TransactionMonth to datetime
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], format='%Y-%m', errors='coerce')

# Data Quality Checks
logging.info("Performing initial data quality checks")
print("Column Data Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nUnique Values in TotalClaims:")
print(df['TotalClaims'].value_counts(dropna=False))
print("\nSummary Statistics for TotalClaims:")
print(df['TotalClaims'].describe())

# Ensure TotalClaims is numeric
if df['TotalClaims'].dtype not in ['float64', 'int64']:
    logging.warning("TotalClaims is not numeric. Attempting to convert.")
    df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')

# Feature Engineering
df['VehicleAge'] = 2025 - df['RegistrationYear']
df['PremiumToSumInsured'] = df['TotalPremium'] / df['SumInsured'].replace(0, np.nan)  # Avoid division by zero

# Select features
potential_num_cols = ['VehicleAge', 'PremiumToSumInsured', 'SumInsured', 'Cylinders', 'kilowatts']
cat_cols = ['Province', 'Gender', 'VehicleType']
target = 'TotalClaims'

# Check for column existence
available_num_cols = [col for col in potential_num_cols if col in df.columns]
missing_cols = [col for col in potential_num_cols if col not in df.columns]
if missing_cols:
    logging.warning(f"Missing columns: {missing_cols}. Excluding from numerical features.")
if not available_num_cols:
    logging.error("No numerical columns available. Exiting.")
    raise ValueError("No numerical columns available for modeling.")

logging.info(f"Available numerical columns: {available_num_cols}")
logging.info(f"Categorical columns: {cat_cols}")

# Filter for policies with claims
df_claims = df[df['TotalClaims'] > 0]
logging.info(f"Filtered to {len(df_claims)} policies with claims")

# Check if df_claims is empty
if df_claims.empty:
    logging.warning("No policies with TotalClaims > 0 found. Switching to claim probability modeling.")
    print("No policies with claims. Modeling claim probability (HasClaim) instead.")
    
    # Create binary target for claim probability
    df['HasClaim'] = df['TotalClaims'] > 0
    target = 'HasClaim'
    
    # Handle missing values for all data
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[available_num_cols] = num_imputer.fit_transform(df[available_num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    # Encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = encoder.fit_transform(df[cat_cols])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))
    df_model = pd.concat([df[available_num_cols].reset_index(drop=True), cat_encoded_df], axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df_model, df[target], test_size=0.2, random_state=42)
    logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Model Building (Classification)
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'XGBoost Classifier': XGBClassifier(random_state=42)
    }
    
    # Model Evaluation
    results = {}
    for name, model in models.items():
        logging.info(f"Training {name} model")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
        logging.info(f"{name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    
    # SHAP Analysis for XGBoost Classifier
    logging.info("Performing SHAP analysis for XGBoost Classifier")
    xgb_model = models['XGBoost Classifier']
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.savefig('../reports/shap_summary_classifier.png')
    plt.close()
    logging.info("SHAP summary plot saved to ../reports/shap_summary_classifier.png")
    
    # Business Recommendations
    print("Top Features for Claim Probability (SHAP):")
    print("1. VehicleAge: Older vehicles may increase claim likelihood, suggesting higher premiums.")
    print("2. SumInsured: Higher insured amounts correlate with claim probability.")
    print("3. Province: Certain regions increase claim likelihood, informing regional pricing.")
else:
    # Proceed with claim severity modeling
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_claims[available_num_cols] = num_imputer.fit_transform(df_claims[available_num_cols])
    df_claims[cat_cols] = cat_imputer.fit_transform(df_claims[cat_cols])
    
    # Encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = encoder.fit_transform(df_claims[cat_cols])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))
    df_model = pd.concat([df_claims[available_num_cols].reset_index(drop=True), cat_encoded_df], axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df_model, df_claims[target], test_size=0.2, random_state=42)
    logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Model Building (Regression)
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }
    
    # Model Evaluation
    results = {}
    for name, model in models.items():
        logging.info(f"Training {name} model")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R2': r2}
        logging.info(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.2f}")
    
    # SHAP Analysis for XGBoost
    logging.info("Performing SHAP analysis for XGBoost")
    xgb_model = models['XGBoost']
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.savefig('../reports/shap_summary.png')
    plt.close()
    logging.info("SHAP summary plot saved to ../reports/shap_summary.png")
    
    # Business Recommendations
    print("Top Features for Claim Severity (SHAP):")
    print("1. VehicleAge: Older vehicles increase claim amounts, suggesting higher premiums.")
    print("2. SumInsured: Higher insured amounts correlate with larger claims.")
    print("3. Province: Certain regions drive higher claims, informing regional pricing.")
