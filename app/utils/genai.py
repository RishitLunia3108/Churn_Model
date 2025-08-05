# app/utils/genai.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Try to import Google's Generative AI library
try:
    import google.generativeai as genai
    # Configure the Gemini API with your API key
    GEMINI_API_KEY = "AIzaSyCnbcxYZey6wPAXmb7qZJ-Pr9m5fr5FcfU"
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False


def generate_churn_explanation(customer_data, prediction, feature_columns, model):
    """Generate a natural language explanation for the churn prediction."""
    # Try Gemini if available
    if GEMINI_AVAILABLE:
        try:
            return generate_gemini_explanation(customer_data, prediction, feature_columns, model)
        except Exception as e:
            print(f"Error with Gemini API: {e}")
            # Fall back to rule-based explanation
    
    # Use rule-based explanation as fallback
    return generate_rule_based_explanation(customer_data, prediction, feature_columns, model)


def generate_gemini_explanation(customer_data, prediction, feature_columns, model):
    """Generate explanation using Google's Gemini AI."""
    # Prepare the customer data for the prompt
    customer_info = {}
    for col in feature_columns:
        if col in customer_data:
            value = customer_data[col]
            # Convert numpy/pandas types to standard Python types for JSON serialization
            if isinstance(value, (np.integer, np.floating)):
                value = float(value)
            elif isinstance(value, np.bool_):
                value = bool(value)
            customer_info[col] = value
    
    # Get feature importance if available
    important_features = []
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        sorted_indices = np.argsort(importance)[::-1]
        important_features = [(feature_columns[i], float(importance[i])) for i in sorted_indices[:5]]
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
        sorted_indices = np.argsort(importance)[::-1]
        important_features = [(feature_columns[i], float(importance[i])) for i in sorted_indices[:5]]
    
    # Format features and values for readability
    formatted_features = []
    for feature, importance_value in important_features:
        formatted_features.append(f"- {feature}: {importance_value:.4f}")
    
    # Create the prompt for Gemini
    prompt = f"""
    As a customer churn analysis expert, generate a brief, natural language explanation for why a customer might churn based on their data.
    
    Customer data:
    {json.dumps(customer_info, indent=2)}
    
    Prediction: {"Customer will churn" if prediction == 1 else "Customer will not churn"}
    
    Top features by importance:
    {chr(10).join(formatted_features)}
    
    Provide a concise, data-driven explanation for this prediction in 1-2 sentences. For example:
    "Customer is likely to churn due to high monthly charges and low tenure."
    
    Your explanation:
    """
    
    # Call Gemini API
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    # Extract and clean the explanation
    explanation = response.text.strip()
    
    # Remove any quotation marks that might be in the response
    explanation = explanation.replace('"', '').replace("'", "")
    
    # Ensure the explanation is concise
    if len(explanation.split()) > 50:
        sentences = explanation.split('.')
        explanation = '.'.join(sentences[:2]) + '.'
    
    return explanation


def generate_rule_based_explanation(customer_data, prediction, feature_columns, model):
    """Generate a rule-based explanation using feature importances."""
    # For Telco dataset, we define meanings for common features
    telco_feature_meaning = {
        'tenure': 'length of time as customer',
        'MonthlyCharges': 'monthly bill amount',
        'TotalCharges': 'total amount billed',
        'Contract': 'contract type',
        'PaymentMethod': 'payment method',
        'OnlineSecurity': 'online security service',
        'TechSupport': 'tech support service',
        'InternetService': 'internet service type',
        'OnlineBackup': 'online backup service',
        'DeviceProtection': 'device protection',
        'PaperlessBilling': 'paperless billing'
    }
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = [feature_columns[i] for i in indices[:3]]
        
        # Get feature values for top features
        feature_values = {}
        for feature in top_features:
            if feature in customer_data:
                feature_values[feature] = customer_data[feature]
                
    elif hasattr(model, 'coef_'):
        coef = np.abs(model.coef_[0])
        indices = np.argsort(coef)[::-1]
        top_features = [feature_columns[i] for i in indices[:3]]
        
        # Get feature values for top features
        feature_values = {}
        for feature in top_features:
            if feature in customer_data:
                feature_values[feature] = customer_data[feature]
    else:
        # If no feature importance available
        return "Based on the customer profile analysis, our system has determined that this customer is " + \
               ("likely to churn" if prediction == 1 else "likely to stay with our service") + "."
    
    # Generate explanation based on prediction and top features
    if prediction == 1:  # Churn
        explanation_parts = []
        
        # Check for common churn patterns in Telco data
        for feature in top_features:
            if feature not in customer_data:
                continue
                
            value = customer_data[feature]
            if feature in telco_feature_meaning:
                feature_desc = telco_feature_meaning[feature]
            else:
                feature_desc = feature.replace('_', ' ')
            
            # Specific rules for Telco dataset
            if feature == 'tenure' and float(value) < 12:
                explanation_parts.append(f"short tenure ({value} months)")
            elif feature == 'MonthlyCharges' and float(value) > 70:
                explanation_parts.append(f"high monthly charges (${value:.2f})")
            elif feature == 'Contract' and (value == 0 or value == 'Month-to-month'):
                explanation_parts.append("month-to-month contract")
            elif feature == 'InternetService' and value in ['DSL', 'Fiber optic']:
                explanation_parts.append(f"{value} internet service")
            elif feature in ['OnlineSecurity', 'TechSupport', 'OnlineBackup'] and (value == 0 or value == 'No'):
                explanation_parts.append(f"no {feature_desc}")
            elif feature == 'PaperlessBilling' and (value == 1 or value == 'Yes'):
                explanation_parts.append("uses paperless billing")
            elif feature == 'PaymentMethod' and 'electronic' in str(value).lower():
                explanation_parts.append("uses electronic payment")
            else:
                if isinstance(value, (int, float)):
                    if float(value) > 0.5:
                        explanation_parts.append(f"high {feature_desc} ({value:.2f})")
                    else:
                        explanation_parts.append(f"low {feature_desc} ({value:.2f})")
                else:
                    explanation_parts.append(f"{feature_desc}: {value}")
        
        if len(explanation_parts) >= 2:
            explanation = f"Customer is likely to churn due to {explanation_parts[0]}, {explanation_parts[1]}"
            if len(explanation_parts) >= 3:
                explanation += f", and {explanation_parts[2]}"
            explanation += "."
        elif len(explanation_parts) == 1:
            explanation = f"Customer is likely to churn primarily due to {explanation_parts[0]}."
        else:
            explanation = "Customer is predicted to churn based on their overall profile."
            
    else:  # Not churn
        explanation_parts = []
        
        for feature in top_features:
            if feature not in customer_data:
                continue
                
            value = customer_data[feature]
            if feature in telco_feature_meaning:
                feature_desc = telco_feature_meaning[feature]
            else:
                feature_desc = feature.replace('_', ' ')
            
            # Specific rules for Telco dataset - retention factors
            if feature == 'tenure' and float(value) > 24:
                explanation_parts.append(f"long tenure ({value} months)")
            elif feature == 'Contract' and (value in [1, 2] or value in ['One year', 'Two year']):
                contract_type = value if isinstance(value, str) else ("one-year" if value == 1 else "two-year")
                explanation_parts.append(f"{contract_type} contract")
            elif feature in ['OnlineSecurity', 'TechSupport', 'OnlineBackup'] and (value == 1 or value == 'Yes'):
                explanation_parts.append(f"has {feature_desc}")
            elif feature == 'InternetService' and value == 'No':
                explanation_parts.append("does not have internet service")
            else:
                if isinstance(value, (int, float)):
                    if float(value) > 0.5:
                        explanation_parts.append(f"high {feature_desc} ({value:.2f})")
                    else:
                        explanation_parts.append(f"low {feature_desc} ({value:.2f})")
                else:
                    explanation_parts.append(f"{feature_desc}: {value}")
                    
        if len(explanation_parts) >= 2:
            explanation = f"Customer is likely to remain due to {explanation_parts[0]}, {explanation_parts[1]}"
            if len(explanation_parts) >= 3:
                explanation += f", and {explanation_parts[2]}"
            explanation += "."
        elif len(explanation_parts) == 1:
            explanation = f"Customer is likely to remain primarily due to {explanation_parts[0]}."
        else:
            explanation = "Customer is predicted to remain based on their overall profile."
    
    return explanation