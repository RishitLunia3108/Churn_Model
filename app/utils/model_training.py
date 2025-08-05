# app/utils/model_training.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate_models(df, target_column, model_types):
    """Train selected models and evaluate their performance."""
    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models based on user selection
    models = {}
    if "logistic_regression" in model_types:
        models["logistic_regression"] = LogisticRegression(max_iter=1000, random_state=42)
    
    if "random_forest" in model_types:
        models["random_forest"] = RandomForestClassifier(n_estimators=100, random_state=42)
    
    if "xgboost" in model_types:
        models["xgboost"] = xgb.XGBClassifier(random_state=42)
    
    # Train models and evaluate
    results = {}
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='binary', zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average='binary', zero_division=0))
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance (if available)
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = {col: float(imp) for col, imp in zip(X.columns, importance)}
        elif hasattr(model, 'coef_'):
            importance = model.coef_[0]
            feature_importance = {col: float(imp) for col, imp in zip(X.columns, importance)}
        
        # Store results
        results[model_name] = {
            "metrics": metrics,
            "confusion_matrix": cm,
            "feature_importance": feature_importance,
            "model": model
        }
    
    # Separate models from results for saving
    models_dict = {model_name: result_dict["model"] for model_name, result_dict in results.items()}
    
    # Remove model objects from results
    for model_name in results:
        results[model_name].pop("model")
    
    return {
        "models": models_dict,
        "results": results,
        "feature_columns": list(X.columns)
    }