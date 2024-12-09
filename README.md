# Housing Price Prediction Using XGBoost 
## Overview
This project leverages XGBoost, a popular gradient boosting algorithm, to predict property prices in Belgium as a project at Becode in 2024. The data used for this project is scraped from Immoweb ealier in another project at Becode. The pipeline includes data preprocessing, feature engineering, model training, and evaluation.
## Installation
To run this project, you need to install the required dependencies. You can use the following command to set up a virtual environment and install the necessary libraries:
python -m venv venv
source venv/bin/activate  # For Windows use: venv\Scripts\activate
pip install -r requirements.txt

## Usage
1. Scraping Data (if applicable)
You can scrape the property data from Immoweb using the scraping script provided in  https://github.com/Fatemeh992/immo_eliza_scraping (not included in this repo for simplicity). Alternatively, you can use an existing CSV file if you have it.

2. Running the Preprocessing Pipeline
Once the data is scraped, run the preprocessing pipeline to clean the data, handle missing values, and encode categorical features. The preprocessing script will also save the cleaned data to a CSV file.

3. Training the Model
After preprocessing, you can train the XGBoost regression model on the prepared data.

4. Evaluating the Model
Once the model is trained, evaluate its performance using metrics such as MAE, RMSE, R², MAPE, and sMAPE.

5. Generating the Evaluation Report
After evaluation, a detailed evaluation report (evaluation_report.md) will be generated. This report includes model metrics, features used, accuracy computing procedures, efficiency (training and inference times), and dataset details.

## Features and Methodology
Feature engineering:
Based on correlation and handing missing values the final features are:

Living Area
Terrace Area
Number of Facades
Swimming Pool
Fully Equipped Kitchen (Encoded)
Furnished (Encoded)
Subtype of Property (Encoded)
State of the Building (Encoded)
Compound Listing (Encoded)
Price (Target)

Preprocessing Pipeline:

Duplicates: Identified and removed duplicate rows.
Missing Values: Imputed missing values in categorical and numerical columns.
Outlier Handling: Used Interquartile Range (IQR) method to detect and handle outliers in Price and Living Area.
Categorical Encoding: One-hot encoded categorical features.
Log Transformation: Applied logarithmic transformation to skewed numerical features to improve model performance.

XGBoost Model:
The model is built using the XGBoostRegressor with the following key hyperparameters:

Max Depth: 7
Learning Rate: 0.1
Number of Estimators: 400
Subsample: 0.8
Objective: reg:squarederror
Cross-validation:
Used 5-fold cross-validation to ensure robust evaluation and model tuning.

Metrics:
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
R² (R-squared)
MAPE (Mean Absolute Percentage Error)
sMAPE (Symmetric Mean Absolute Percentage Error)
 ## Performance Summary
 Training Set
	•	MAE: 0.0826
	•	RMSE: 0.1162
	•	R2: 91.7015
	•	MAPE: 0.6530
	•	sMAPE: 65.2797
Test Set
	•	MAE: 0.1272
	•	RMSE: 0.1811
	•	R2: 79.9272
	•	MAPE: 1.0040
	•	sMAPE: 100.3561

Cross-Validation Accuracy: 77.43% (1.42%)