# Customer Churn Prediction API

A comprehensive FastAPI application that predicts customer churn using machine learning models and provides AI-powered explanations for predictions.

## 🎯 Project Overview

This application provides a complete ML pipeline for customer churn prediction with the following features:

- **Data Upload & Analysis**: Upload CSV datasets and get comprehensive metadata
- **Data Cleaning & Preprocessing**: Handle missing values, categorical encoding, and feature scaling
- **Multi-Model Training**: Train Logistic Regression, Random Forest, and XGBoost models
- **AI-Powered Explanations**: Generate natural language explanations using Google's Gemini AI
- **Prediction Pipeline**: Make predictions on new data with explanations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Task
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file (optional - API key is already configured)
   echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
   ```

5. **Run the application**
   ```bash
   python run.py
   ```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Endpoints

#### 1. Upload Dataset
```http
POST /upload
Content-Type: multipart/form-data

Parameters:
- file: CSV file
- target_column: Target column name (optional)
```

**Response:**
```json
{
  "dataset_id": "uuid-string",
  "filename": "Telco-Customer-Churn.csv",
  "shape": [7043, 21],
  "metadata": {
    "column_types": {...},
    "missing_values": {...},
    "statistics": {...},
    "categorical_info": {...},
    "data_issues": [...],
    "preview": [...]
  }
}
```

#### 2. Data Cleaning
```http
POST /data_cleaning
Content-Type: application/json

{
  "dataset_id": "uuid-string",
  "missing_strategy": "impute",
  "categorical_encoding": "label",
  "scaling": "standard"
}
```

**Response:**
```json
{
  "message": "Data cleaning completed successfully",
  "cleaning_info": {
    "missing_values": {...},
    "categorical_encoding": {...},
    "scaling": {...}
  },
  "preview": [...]
}
```

#### 3. Train Models
```http
POST /train_model
Content-Type: application/json

{
  "dataset_id": "uuid-string",
  "model_types": ["logistic_regression", "random_forest", "xgboost"],
  "target_column": "Churn"
}
```

**Response:**
```json
{
  "model_id": "uuid-string",
  "message": "Models trained successfully",
  "performance": {
    "logistic_regression": {
      "metrics": {
        "accuracy": 0.8234,
        "precision": 0.7891,
        "recall": 0.7123,
        "f1": 0.7489
      }
    }
  }
}
```

#### 4. Generate Predictions
```http
POST /predict
Content-Type: multipart/form-data

Parameters:
- file: CSV file with customer data
- model_id: Trained model ID
- model_type: Model type to use (optional)
```

**Response:**
```json
{
  "message": "Predictions generated successfully",
  "model_used": "random_forest",
  "preview": [
    {
      "customerID": "1234-ABCD",
      "tenure": 12,
      "MonthlyCharges": 89.50,
      "Contract": "Month-to-month",
      "prediction": 1,
      "explanation": "Customer is likely to churn due to high monthly charges and short tenure."
    }
  ],
  "results_file": "app/uploads/results_test_data.csv"
}
```

## 📊 Sample Data

### Input CSV Format
The application works with the Telco Customer Churn dataset. Sample columns:

```csv
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,Yes,No,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.50,No
```

### Output Format
The prediction endpoint returns a CSV with original data + predictions + explanations:

```csv
customerID,tenure,MonthlyCharges,Contract,prediction,explanation
1234-ABCD,12,89.50,Month-to-month,1,"Customer is likely to churn due to high monthly charges and short tenure."
5678-EFGH,48,45.20,One year,0,"Customer is likely to remain due to long tenure and annual contract."
```

## 🤖 GenAI Integration

### Gemini AI Prompt Structure

The application uses Google's Gemini AI to generate natural language explanations. The prompt structure is:

```python
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
```

### Fallback System
If Gemini AI is unavailable, the system falls back to rule-based explanations using feature importance and domain-specific rules.

## 🏗️ Project Structure

```
customer-churn-prediction-api/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── utils/
│   │   ├── data_cleaning.py    # Data preprocessing
│   │   ├── model_training.py   # ML model training
│   │   └── genai.py           # AI explanations
│   ├── uploads/               # Temporary file storage
│   ├── models/                # Trained model storage
│   └── storage/               # Dataset metadata
├── data/
│   ├── Telco-Customer-Churn.csv  # Sample dataset
│   └── test_samples.csv          # Test data
├── test_app.py                # Complete workflow test
├── run.py                     # Application entry point
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key (optional - default key provided)

### Model Parameters
- **Missing Strategy**: `drop` or `impute`
- **Categorical Encoding**: `label` or `onehot`
- **Scaling**: `standard` or `minmax`
- **Model Types**: `logistic_regression`, `random_forest`, `xgboost`

## 🧪 Testing

### Run Complete Workflow
```bash
python test_app.py
```

This will:
1. Upload the sample dataset
2. Perform data cleaning
3. Train multiple models
4. Generate predictions with explanations
5. Display results

### Manual Testing
Use the interactive documentation at `http://localhost:8000/docs` to test individual endpoints.

## 📈 Performance Metrics

The application provides comprehensive model evaluation:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## 🔍 Key Features

### 1. Data Pipeline
- **Automatic Analysis**: Column types, missing values, statistics
- **Flexible Preprocessing**: Multiple strategies for data cleaning
- **Transformer Persistence**: Consistent preprocessing across training and prediction

### 2. Machine Learning
- **Multiple Algorithms**: Logistic Regression, Random Forest, XGBoost
- **Feature Importance**: Model interpretability
- **Performance Comparison**: Side-by-side model evaluation

### 3. AI Explanations
- **Gemini AI Integration**: Natural language explanations
- **Rule-based Fallback**: Reliable explanations when AI unavailable
- **Contextual Analysis**: Customer-specific reasoning

### 4. Production Ready
- **Error Handling**: Comprehensive error management
- **Data Persistence**: Survives server restarts
- **Scalable Architecture**: Handles multiple concurrent requests

## 🚀 Deployment

### Local Development
```bash
python run.py
```

### Production Deployment
```bash
# Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Using gunicorn (recommended for production)
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📝 API Usage Examples

### Python Client
```python
import requests

# Upload dataset
with open('data/Telco-Customer-Churn.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload',
        files={'file': f},
        data={'target_column': 'Churn'}
    )
dataset_id = response.json()['dataset_id']

# Clean data
cleaning_response = requests.post(
    'http://localhost:8000/data_cleaning',
    json={
        'dataset_id': dataset_id,
        'missing_strategy': 'impute',
        'categorical_encoding': 'label',
        'scaling': 'standard'
    }
)

# Train models
training_response = requests.post(
    'http://localhost:8000/train_model',
    json={
        'dataset_id': dataset_id,
        'model_types': ['random_forest'],
        'target_column': 'Churn'
    }
)
model_id = training_response.json()['model_id']

# Make predictions
with open('data/test_samples.csv', 'rb') as f:
    predict_response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f},
        data={'model_id': model_id}
    )
```

### cURL Examples
```bash
# Upload dataset
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/Telco-Customer-Churn.csv" \
  -F "target_column=Churn"

# Clean data
curl -X POST "http://localhost:8000/data_cleaning" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"your-dataset-id","missing_strategy":"impute","categorical_encoding":"label","scaling":"standard"}'
```
