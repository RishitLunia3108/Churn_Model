# app/main.py
import os
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List, Optional
import uuid
from pydantic import BaseModel, ConfigDict
import json


from app.utils.data_cleaning import perform_data_cleaning, analyze_dataset
from app.utils.model_training import train_and_evaluate_models
from app.utils.genai import generate_churn_explanation

# Create directories if they don't exist
os.makedirs("app/uploads", exist_ok=True)
os.makedirs("app/models", exist_ok=True)
os.makedirs("app/storage", exist_ok=True)  # Add this line for storage directory

def save_dataset_info(dataset_id, info):
    """Save dataset info to a JSON file for persistence across server restarts."""
    try:
        # Create a serializable copy of the info
        serializable_info = {}
        for key, value in info.items():
            # Skip non-serializable objects
            if key in ['df', 'transformer_objects']:
                continue
                
            # Handle numpy and pandas types
            if isinstance(value, (np.integer, np.floating)):
                serializable_info[key] = float(value)
            elif isinstance(value, np.bool_):
                serializable_info[key] = bool(value)
            elif isinstance(value, np.ndarray):
                serializable_info[key] = value.tolist()
            elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                serializable_info[key] = value
            else:
                # Convert other types to string
                serializable_info[key] = str(value)
        
        # Save to file
        with open(f"app/storage/{dataset_id}_info.json", "w") as f:
            json.dump(serializable_info, f, indent=2)
            
    except Exception as e:
        print(f"Error saving dataset info: {e}")

def load_dataset_info(dataset_id):
    """Load dataset info from a JSON file."""
    try:
        with open(f"app/storage/{dataset_id}_info.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading dataset info: {e}")
        return None

app = FastAPI(title="Customer Churn Prediction API",
              description="API for predicting customer churn with ML and GenAI explanations")

# In-memory storage for dataset and model info
datasets = {}
models = {}

class CleaningParams(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    dataset_id: str
    missing_strategy: str = "drop"  # 'drop' or 'impute'
    categorical_encoding: str = "label"  # 'label' or 'onehot'
    scaling: str = "standard"  # 'standard' or 'minmax'

class TrainingParams(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    dataset_id: str
    model_types: List[str]  # e.g., ["logistic_regression", "random_forest", "xgboost"]
    target_column: str


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None)
):
    """
    Upload a CSV dataset and optionally specify the target column.
    Returns dataset metadata including column types, missing values, and a sample preview.
    """
    # Generate a unique ID for this dataset
    dataset_id = str(uuid.uuid4())
    
    # Save the uploaded file
    file_path = f"app/uploads/{dataset_id}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Read and analyze the dataset
    try:
        df = pd.read_csv(file_path)
        metadata = analyze_dataset(df)
        
        # Store dataset info
        datasets[dataset_id] = {
            "path": file_path,
            "filename": file.filename,
            "target_column": target_column,
            "columns": list(df.columns),
            "shape": df.shape
        }
        
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "shape": df.shape,
            "metadata": metadata,
            "message": "Dataset uploaded and analyzed successfully"
        }
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/data_cleaning")
async def data_cleaning(params: CleaningParams):
    """
    Clean and preprocess the dataset based on specified parameters.
    Options for missing value handling, categorical encoding, and scaling.
    """
    if params.dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Load the dataset
        dataset_info = datasets[params.dataset_id]
        df = pd.read_csv(dataset_info["path"])
        
        # Perform cleaning
        cleaned_df, cleaning_info, transformers = perform_data_cleaning(
            df, 
            missing_strategy=params.missing_strategy,
            categorical_encoding=params.categorical_encoding,
            scaling=params.scaling
        )
        
        # Update dataset info
        dataset_info["cleaned"] = True
        dataset_info["cleaning_params"] = params.dict()
        
        # Save cleaned dataframe
        cleaned_path = f"app/uploads/{params.dataset_id}_cleaned.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        dataset_info["cleaned_path"] = cleaned_path
        
        # Save transformers
        transformers_path = f"app/uploads/{params.dataset_id}_transformers.joblib"
        joblib.dump(transformers, transformers_path)
        dataset_info["transformers_path"] = transformers_path
        
        # Update storage
        datasets[params.dataset_id] = dataset_info
        save_dataset_info(params.dataset_id, dataset_info)
        
        # Return a preview of the cleaned data
        return {
            "message": "Data cleaning completed successfully",
            "cleaning_info": cleaning_info,
            "preview": cleaned_df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error during data cleaning: {str(e)}")


@app.post("/train_model")
async def train_model(params: TrainingParams):
    """
    Train and evaluate selected machine learning models.
    Returns performance metrics including accuracy, precision, recall, F1-score, and confusion matrix.
    """
    if params.dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset_info = datasets[params.dataset_id]
    
    if "cleaned_path" not in dataset_info:
        raise HTTPException(status_code=400, detail="Dataset must be cleaned before training")
    
    try:
        # Load the cleaned dataset
        df = pd.read_csv(dataset_info["cleaned_path"])
        
        if params.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{params.target_column}' not found")
        
        # Train models
        model_results = train_and_evaluate_models(
            df, 
            target_column=params.target_column,
            model_types=params.model_types
        )
        
        # Generate model ID
        model_id = str(uuid.uuid4())
        
        # Save model info
        models[model_id] = {
            "dataset_id": params.dataset_id,
            "target_column": params.target_column,
            "model_types": params.model_types,
            "results": model_results,
            "feature_columns": [col for col in df.columns if col != params.target_column]
        }
        
        # Save models to disk
        joblib.dump(model_results["models"], f"app/models/{model_id}.joblib")
        
        return {
            "model_id": model_id,
            "message": "Models trained successfully",
            "performance": {
                model_type: {
                    "metrics": results["metrics"],
                    "confusion_matrix": results["confusion_matrix"].tolist() if isinstance(results.get("confusion_matrix"), np.ndarray) else results.get("confusion_matrix")
                } 
                for model_type, results in model_results["results"].items()
            }
        }
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error during model training: {str(e)}")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    model_type: Optional[str] = Form(None)
):
    """Generate predictions for new data and provide GenAI explanations."""
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Save the uploaded file
        file_path = f"app/uploads/prediction_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Load the test data
        test_df = pd.read_csv(file_path)
        
        # Load model info and models
        model_info = models[model_id]
        loaded_models = joblib.load(f"app/models/{model_id}.joblib")
        
        # If model_type is not specified, use the first model
        if model_type is None or model_type not in loaded_models:
            model_type = model_info["model_types"][0]
        
        # Get feature columns
        feature_columns = model_info["feature_columns"]
        
        # Get the dataset_id used for training this model
        dataset_id = model_info["dataset_id"]
        
        # Load dataset info
        dataset_info = datasets.get(dataset_id) or load_dataset_info(dataset_id)
        if dataset_info is None:
            raise HTTPException(status_code=404, detail="Training dataset info not found")
        
        # Load transformers
        transformers_path = dataset_info.get("transformers_path")
        if not transformers_path or not os.path.exists(transformers_path):
            raise HTTPException(status_code=404, detail="Preprocessing transformers not found")
        
        transformers = joblib.load(transformers_path)
        
        # Get cleaning parameters
        cleaning_params = dataset_info.get("cleaning_params", {})
        
        # Apply the same preprocessing to the test data
        test_cleaned_df, _, _ = perform_data_cleaning(
            test_df,
            missing_strategy=cleaning_params.get("missing_strategy", "drop"),
            categorical_encoding=cleaning_params.get("categorical_encoding", "label"),
            scaling=cleaning_params.get("scaling", "standard"),
            transformers=transformers  # Use the saved transformers
        )
        
        # Ensure all needed columns are available
        missing_features = set(feature_columns) - set(test_cleaned_df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Processed test data missing required columns: {missing_features}"
            )
        
        # Use the specified model for prediction
        model = loaded_models[model_type]
        
        # Generate predictions
        predictions = model.predict(test_cleaned_df[feature_columns])
        
        # Add predictions to original data for better interpretability
        test_df['prediction'] = predictions
        
        # Generate explanations for each prediction
        explanations = []
        for idx, row in test_df.iterrows():
            # For explanation, we'll pass both original and processed features
            # Create a dictionary combining both
            combined_data = {**row.to_dict()}
            # Add the processed features with a prefix
            for col in feature_columns:
                if col in test_cleaned_df.columns:
                    combined_data[f"processed_{col}"] = test_cleaned_df.iloc[idx][col]
            
            explanation = generate_churn_explanation(
                combined_data, 
                row['prediction'],
                feature_columns,
                model
            )
            explanations.append(explanation)
        
        test_df['explanation'] = explanations
        
        # Save results
        results_path = f"app/uploads/results_{file.filename}"
        test_df.to_csv(results_path, index=False)
        
        return {
            "message": "Predictions generated successfully",
            "model_used": model_type,
            "preview": test_df.head(5).to_dict(orient="records"),
            "results_file": results_path
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during prediction: {e}\n{error_details}")
        return HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/")
async def root():
    """API root - provides basic info about the API."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/upload", "method": "POST", "description": "Upload dataset"},
            {"path": "/data_cleaning", "method": "POST", "description": "Clean dataset"},
            {"path": "/train_model", "method": "POST", "description": "Train models"},
            {"path": "/predict", "method": "POST", "description": "Generate predictions"}
        ],
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)