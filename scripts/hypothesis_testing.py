import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load data
df = pd.read_csv('data/insurance_data.csv')

# Define metrics
df['HasClaim'] = df['TotalClaims'] > 0
df['Margin'] = df['TotalPremium'] - df['TotalClaims']
claim_severity = df[df['HasClaim']]['TotalClaims']

# Hypothesis 1: Risk differences across provinces
# Claim Frequency
province_freq = df.groupby('Province')['HasClaim'].mean()
chi2, p_freq_prov, _, _ = stats.chi2_contingency(pd.crosstab(df['Province'], df['HasClaim']))
print(f"Province Claim Frequency: Chi2 p-value = {p_freq_prov}")

# Claim Severity (ANOVA)
severity_by_prov = [df[df['Province'] == prov]['TotalClaims'][df['HasClaim']] for prov in df['Province'].unique()]
f_stat, p_sev_prov = stats.f_oneway(*severity_by_prov)
print(f"Province Claim Severity: ANOVA p-value = {p_sev_prov}")

# Hypothesis 2: Risk differences between zip codes
# Claim Frequency
zip_freq = df.groupby('PostalCode')['HasClaim'].mean()
chi2, p_freq_zip, _, _ = stats.chi2_contingency(pd.crosstab(df['PostalCode'], df['HasClaim']))
print(f"Zip Code Claim Frequency: Chi2 p-value = {p_freq_zip}")

# Claim Severity (ANOVA)
severity_by_zip = [df[df['PostalCode'] == zip_]['TotalClaims'][df['HasClaim']] for zip_ in df['PostalCode'].unique()]
f_stat, p_sev_zip = stats.f_oneway(*severity_by_zip)
print(f"Zip Code Claim Severity: ANOVA p-value = {p_sev_zip}")

# Hypothesis 3: Margin differences between zip codes
margin_by_zip = [df[df['PostalCode'] == zip_]['Margin'] for zip_ in df['PostalCode'].unique()]
f_stat, p_margin_zip = stats.f_oneway(*margin_by_zip)
print(f"Zip Code Margin: ANOVA p-value = {p_margin_zip}")

# Hypothesis 4: Risk differences between genders
# Claim Frequency
gender_freq = df.groupby('Gender')['HasClaim'].mean()
chi2, p_freq_gender, _, _ = stats.chi2_contingency(pd.crosstab(df['Gender'], df['HasClaim']))
print(f"Gender Claim Frequency: Chi2 p-value = {p_freq_gender}")

# Claim Severity (t-test)
severity_male = df[(df['Gender'] == 'Male') & df['HasClaim']]['TotalClaims']
severity_female = df[(df['Gender'] == 'Female') & df['HasClaim']]['TotalClaims']
t_stat, p_sev_gender = stats.ttest_ind(severity_male, severity_female)
print(f"Gender Claim Severity: t-test p-value = {p_sev_gender}")

# Business Recommendations
if p_freq_prov < 0.05:
    print("Reject H0: Significant risk differences across provinces. Adjust premiums by region.")
if p_sev_zip < 0.05:
    print("Reject H0: Significant risk differences between zip codes. Target low-risk zip codes for marketing.")
if p_margin_zip < 0.05:
    print("Reject H0: Significant margin differences between zip codes. Optimize pricing in high-margin areas.")
if p_sev_gender < 0.05:
    print("Reject H0: Significant risk differences between genders. Consider gender-based pricing adjustments.")