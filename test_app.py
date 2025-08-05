# test_app.py
import requests
import json
import pandas as pd
import os
import time

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_full_workflow():
    """Test the full workflow of the churn prediction app."""
    print("Testing Customer Churn Prediction API")
    print("=" * 50)
    
    # Check if sample dataset exists, otherwise create a small sample
    if not os.path.exists("data/Telco-Customer-Churn.csv"):
        print("Creating a small sample dataset...")
        # Create a sample based on the provided structure
        sample_data = {
            'customerID': ['7590-VHVEG', '5575-GNVDE', '3668-QPYBK', '7795-CFOCW'],
            'gender': ['Female', 'Male', 'Male', 'Male'],
            'SeniorCitizen': [0, 0, 0, 0],
            'Partner': ['Yes', 'No', 'No', 'No'],
            'Dependents': ['No', 'No', 'No', 'No'],
            'tenure': [1, 34, 2, 45],
            'PhoneService': ['No', 'Yes', 'Yes', 'No'],
            'MultipleLines': ['No phone service', 'No', 'No', 'No phone service'],
            'InternetService': ['DSL', 'DSL', 'DSL', 'DSL'],
            'OnlineSecurity': ['No', 'Yes', 'Yes', 'Yes'],
            'OnlineBackup': ['Yes', 'No', 'Yes', 'No'],
            'DeviceProtection': ['No', 'Yes', 'No', 'Yes'],
            'TechSupport': ['No', 'No', 'No', 'Yes'],
            'StreamingTV': ['No', 'No', 'No', 'No'],
            'StreamingMovies': ['No', 'No', 'No', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'One year'],
            'PaperlessBilling': ['Yes', 'No', 'Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Mailed check', 'Bank transfer (automatic)'],
            'MonthlyCharges': [29.85, 56.95, 53.85, 42.30],
            'TotalCharges': [29.85, 1889.50, 108.15, 1840.75],
            'Churn': ['No', 'No', 'Yes', 'No']
        }
        df = pd.DataFrame(sample_data)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/Telco-Customer-Churn.csv", index=False)
    
    # 1. Upload Dataset
    print("\n1. Uploading dataset...")
    with open("data/Telco-Customer-Churn.csv", "rb") as f:
        upload_response = requests.post(
            f"{BASE_URL}/upload",
            files={"file": ("Telco-Customer-Churn.csv", f)},
            data={"target_column": "Churn"}
        )
    
    if upload_response.status_code != 200:
        print(f"Error uploading dataset: {upload_response.text}")
        return
    
    upload_data = upload_response.json()
    dataset_id = upload_data["dataset_id"]
    print(f"Dataset uploaded successfully. ID: {dataset_id}")
    print(f"Columns: {len(upload_data['metadata']['column_types'])}")
    print(f"Rows: {upload_data['shape'][0]}")
    
    # 2. Data Cleaning
    print("\n2. Performing data cleaning...")
    cleaning_params = {
        "dataset_id": dataset_id,
        "missing_strategy": "impute",
        "categorical_encoding": "label",
        "scaling": "standard"
    }
    
    cleaning_response = requests.post(
        f"{BASE_URL}/data_cleaning",
        json=cleaning_params
    )
    
    if cleaning_response.status_code != 200:
        print(f"Error cleaning data: {cleaning_response.text}")
        return
    
    cleaning_data = cleaning_response.json()
    print("Data cleaning completed successfully")
    print(f"Preview of cleaned data: {len(cleaning_data['preview'])} rows")
    
    # 3. Train Model
    print("\n3. Training models...")
    training_params = {
        "dataset_id": dataset_id,
        "model_types": ["logistic_regression", "random_forest", "xgboost"],
        "target_column": "Churn"
    }
    
    training_response = requests.post(
        f"{BASE_URL}/train_model",
        json=training_params
    )
    
    if training_response.status_code != 200:
        print(f"Error training models: {training_response.text}")
        return
    
    training_data = training_response.json()
    model_id = training_data["model_id"]
    print(f"Models trained successfully. Model ID: {model_id}")
    
    # Print metrics for each model
    for model_type, results in training_data["performance"].items():
        print(f"\n{model_type.upper()} Performance:")
        for metric, value in results["metrics"].items():
            print(f"  {metric}: {value:.4f}")
    
    # 4. Make Predictions
    print("\n4. Making predictions on test data...")
    # Create a small test dataset from the original
    df = pd.read_csv("data/Telco-Customer-Churn.csv")
    test_df = df.sample(min(5, len(df)))  # Take 5 random samples or less if dataset is small
    test_df.to_csv("data/test_samples.csv", index=False)
    
    with open("data/test_samples.csv", "rb") as f:
        predict_response = requests.post(
            f"{BASE_URL}/predict",
            files={"file": ("test_samples.csv", f)},
            data={"model_id": model_id, "model_type": "random_forest"}
        )
    
    if predict_response.status_code != 200:
        print(f"Error making predictions: {predict_response.text}")
        return
    
    prediction_data = predict_response.json()
    print("Predictions generated successfully")
    print("\nResponse from prediction endpoint:")
    print(json.dumps(prediction_data, indent=2))
    
    # Display predictions with explanations if available
    print("\nPredictions and Explanations:")
    if "preview" in prediction_data:
        for i, prediction in enumerate(prediction_data["preview"]):
            print(f"\nCustomer {i+1}:")
            churn = "Yes" if prediction.get("prediction") == 1 else "No"
            print(f"Churn Prediction: {churn}")
            print(f"Explanation: {prediction.get('explanation')}")
    else:
        print("No preview data available in the response")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_full_workflow()