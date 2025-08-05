# app/utils/data_cleaning.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


def analyze_dataset(df):
    """Analyze dataset and return metadata."""
    # Get column types
    column_types = df.dtypes.astype(str).to_dict()
    
    # Count missing values
    missing_values = df.isnull().sum().to_dict()
    
    # Get basic statistics for numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    statistics = {}
    if not numeric_columns.empty:
        statistics = df[numeric_columns].describe().to_dict()
    
    # Categorical column info
    categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_info = {}
    for col in categorical_columns:
        categorical_info[col] = {
            'unique_values': df[col].nunique(),
            'top_values': df[col].value_counts().head(5).to_dict()
        }
    
    # For Telco dataset - check for specific issues
    data_issues = []
    
    # Check if TotalCharges is string instead of numeric
    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == 'object':
        data_issues.append("'TotalCharges' column is not numeric. It will be converted to numeric during cleaning.")
    
    # Check if Churn column is in expected format
    if 'Churn' in df.columns:
        unique_values = df['Churn'].unique()
        data_issues.append(f"'Churn' column contains these values: {unique_values}")
        if set(unique_values) == {'Yes', 'No'}:
            data_issues.append("'Churn' will be encoded as 1 for 'Yes' and 0 for 'No'")
    
    return {
        'column_types': column_types,
        'missing_values': missing_values,
        'statistics': statistics,
        'categorical_info': categorical_info,
        'data_issues': data_issues,
        'preview': df.head(5).to_dict(orient='records')
    }


def perform_data_cleaning(df, missing_strategy='drop', categorical_encoding='label', scaling='standard', transformers=None):
    """
    Clean and preprocess the dataset based on specified parameters.
    
    Args:
        df: The DataFrame to clean
        missing_strategy: How to handle missing values ('drop' or 'impute')
        categorical_encoding: How to encode categorical variables ('label' or 'onehot')
        scaling: How to scale numerical features ('standard', 'minmax', or None)
        transformers: Dictionary of existing transformers to use instead of creating new ones
        
    Returns:
        processed_df: The cleaned and processed DataFrame
        cleaning_info: Information about the cleaning process
        transformers: Dictionary of transformation objects that can be reused
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    cleaning_info = {}
    
    # Initialize transformers dictionary if not provided
    if transformers is None:
        transformers = {}
    
    # Special handling for Telco dataset
    
    # 1. Remove customerID if present (not useful for prediction)
    if 'customerID' in processed_df.columns:
        processed_df = processed_df.drop('customerID', axis=1)
        cleaning_info['removed_columns'] = ['customerID']
    
    # 2. Convert TotalCharges to numeric if it's a string
    if 'TotalCharges' in processed_df.columns and processed_df['TotalCharges'].dtype == 'object':
        processed_df['TotalCharges'] = pd.to_numeric(processed_df['TotalCharges'], errors='coerce')
        cleaning_info['conversions'] = {'TotalCharges': 'numeric'}
    
    # Handle missing values
    initial_rows = processed_df.shape[0]
    initial_missing = processed_df.isnull().sum().sum()
    
    if missing_strategy == 'drop':
        processed_df = processed_df.dropna()
        cleaning_info['missing_values'] = {
            'strategy': 'drop',
            'rows_removed': initial_rows - processed_df.shape[0],
            'initial_missing': int(initial_missing)
        }
    elif missing_strategy == 'impute':
        # Impute numerical columns with mean
        numerical_cols = processed_df.select_dtypes(include=['number']).columns
        if not numerical_cols.empty:
            if 'num_imputer' not in transformers:
                transformers['num_imputer'] = SimpleImputer(strategy='mean')
                transformers['num_imputer'].fit(processed_df[numerical_cols])
            
            processed_df[numerical_cols] = transformers['num_imputer'].transform(processed_df[numerical_cols])
        
        # Impute categorical columns with mode
        categorical_cols = processed_df.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            for col in categorical_cols:
                if processed_df[col].isnull().sum() > 0:
                    if f'cat_imputer_{col}' not in transformers:
                        mode_value = processed_df[col].mode()[0]
                        transformers[f'cat_imputer_{col}'] = mode_value
                    else:
                        mode_value = transformers[f'cat_imputer_{col}']
                    
                    processed_df[col] = processed_df[col].fillna(mode_value)
        
        cleaning_info['missing_values'] = {
            'strategy': 'impute',
            'numerical_strategy': 'mean',
            'categorical_strategy': 'mode',
            'initial_missing': int(initial_missing),
            'remaining_missing': int(processed_df.isnull().sum().sum())
        }
    
    # Handle target column 'Churn' - Convert Yes/No to 1/0
    if 'Churn' in processed_df.columns and processed_df['Churn'].dtype == 'object':
        if set(processed_df['Churn'].unique()) == {'Yes', 'No'}:
            processed_df['Churn'] = (processed_df['Churn'] == 'Yes').astype(int)
            cleaning_info['target_encoding'] = {'Churn': 'Yes=1, No=0'}
    
    # Convert other binary Yes/No columns to 1/0
    binary_cols = []
    for col in processed_df.columns:
        if col != 'Churn' and processed_df[col].dtype == 'object':
            if set(processed_df[col].unique()).issubset({'Yes', 'No', None, np.nan}):
                processed_df[col] = (processed_df[col] == 'Yes').astype(int)
                binary_cols.append(col)
    
    if binary_cols:
        cleaning_info['binary_conversions'] = binary_cols
    
    # Handle remaining categorical encoding
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    
    if not categorical_cols.empty:
        if categorical_encoding == 'label':
            for col in categorical_cols:
                if f'label_encoder_{col}' not in transformers:
                    transformers[f'label_encoder_{col}'] = LabelEncoder()
                    transformers[f'label_encoder_{col}'].fit(processed_df[col].astype(str))
                
                processed_df[col] = transformers[f'label_encoder_{col}'].transform(
                    processed_df[col].astype(str)
                )
            
            cleaning_info['categorical_encoding'] = {
                'strategy': 'label',
                'encoded_columns': list(categorical_cols)
            }
        elif categorical_encoding == 'onehot':
            # Use OneHotEncoder for remaining categorical columns
            if 'onehot_encoder' not in transformers:
                transformers['onehot_encoder'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                transformers['onehot_encoder'].fit(processed_df[categorical_cols])
            
            encoded_data = transformers['onehot_encoder'].transform(processed_df[categorical_cols])
            
            # Create DataFrame with one-hot encoded columns
            encoded_df = pd.DataFrame(
                encoded_data, 
                columns=transformers['onehot_encoder'].get_feature_names_out(categorical_cols),
                index=processed_df.index
            )
            
            # Drop original categorical columns and join encoded ones
            processed_df = processed_df.drop(categorical_cols, axis=1)
            processed_df = pd.concat([processed_df, encoded_df], axis=1)
            
            cleaning_info['categorical_encoding'] = {
                'strategy': 'onehot',
                'original_columns': list(categorical_cols),
                'encoded_columns': list(transformers['onehot_encoder'].get_feature_names_out(categorical_cols))
            }
    
    # Apply scaling to numerical features
    numerical_cols = processed_df.select_dtypes(include=['number']).columns
    # Exclude the target column from scaling if it's there
    if 'Churn' in numerical_cols:
        numerical_cols = numerical_cols.drop('Churn')
    
    if not numerical_cols.empty and scaling:
        if scaling == 'standard':
            if 'scaler' not in transformers:
                transformers['scaler'] = StandardScaler()
                transformers['scaler'].fit(processed_df[numerical_cols])
            
            scaled_data = transformers['scaler'].transform(processed_df[numerical_cols])
            cleaning_info['scaling'] = {
                'strategy': 'standard',
                'scaled_columns': list(numerical_cols)
            }
        elif scaling == 'minmax':
            if 'scaler' not in transformers:
                transformers['scaler'] = MinMaxScaler()
                transformers['scaler'].fit(processed_df[numerical_cols])
            
            scaled_data = transformers['scaler'].transform(processed_df[numerical_cols])
            cleaning_info['scaling'] = {
                'strategy': 'minmax',
                'scaled_columns': list(numerical_cols)
            }
        
        # Replace numerical columns with scaled versions
        processed_df[numerical_cols] = scaled_data
    
    return processed_df, cleaning_info, transformers