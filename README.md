Insurance Analytics Project
Overview
This repository contains the code and documentation for the B5W3: End-to-End Insurance Risk Analytics & Predictive Modeling project for AlphaCare Insurance Solutions (ACIS). The goal is to analyze historical car insurance data from South Africa (Feb 2014 - Aug 2015) to identify low-risk segments, optimize premiums, and develop predictive models.
Project Structure

/data/: Raw and processed datasets (tracked with DVC).
/src/: Python scripts for EDA, hypothesis testing, and modeling.
/notebooks/: Jupyter notebooks for exploratory analysis.
/reports/: Final reports and visualizations.
/tests/: Unit tests for code validation.

Setup Instructions

Clone the Repository:
git clone https://github.com/minte-atnafu/insurance-analytics.git
cd insurance-analytics


Install Dependencies:
pip install -r requirements.txt


Initialize DVC:
dvc init
dvc remote add -d localstorage /path/to/local/storage



Tasks

Task 1: Git setup, EDA, and statistical analysis.
Task 2: Data version control with DVC.
Task 3: A/B hypothesis testing for risk and margin differences.
Task 4: Predictive modeling for claim severity and premium optimization.

CI/CD

GitHub Actions workflow (.github/workflows/ci.yml) lints code and runs tests on push/pull requests.


