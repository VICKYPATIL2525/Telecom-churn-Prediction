# 🚀 Telecom Churn Prediction - Deployment

This folder contains production-ready deployment applications for the Telecom Churn Prediction model.

## 📁 Contents

```
deployment/
├── app_flask.py           # Flask web application with UI
├── app_fastapi.py         # FastAPI REST API with Swagger docs
├── templates/
│   └── index.html         # Flask web interface
├── static/                # Static assets (CSS, JS, images)
└── README.md             # This file
```

## 🎯 Features

### Both Applications Include:
✅ **Multiple Model Support** - Choose from 5 trained models (XGBoost, LightGBM, Random Forest, Logistic Regression, SVM)
✅ **Predefined Test Cases** - 10 customer profiles for quick testing
✅ **Real-time Predictions** - Instant churn probability calculation
✅ **Risk Level Assessment** - Automated risk categorization
✅ **Actionable Recommendations** - Business insights based on predictions

### Flask App (Web UI)
- Beautiful, responsive web interface
- Form-based input
- Visual results with charts
- Auto-fill test cases via dropdown
- Model performance metrics display

### FastAPI (REST API)
- Interactive Swagger documentation
- RESTful endpoints
- JSON request/response
- Automatic API docs at `/docs`
- ReDoc documentation at `/redoc`

## 🔧 Installation

### Prerequisites
```bash
Python 3.13.2 or higher
```

### Install Dependencies
```bash
# From project root
pip install -r requirements.txt
```

## 🚀 Running the Applications

### Option 1: Flask Web Application

```bash
# Navigate to deployment folder
cd deployment

# Run Flask app
python app_flask.py
```

**Access at:** http://localhost:5000

### Option 2: FastAPI REST API

```bash
# Navigate to deployment folder
cd deployment

# Run FastAPI app
python app_fastapi.py

# OR using uvicorn directly
uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 8000
```

**Access at:**
- API Root: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📊 Available Models

| Model | Accuracy | Recall | Precision | F1-Score |
|-------|----------|--------|-----------|----------|
| **XGBoost** ⭐ | 96.25% | 81.44% | 91.86% | 86.34% |
| LightGBM | 96.10% | 78.35% | 93.83% | 85.39% |
| Random Forest | 95.95% | 78.35% | 92.68% | 84.92% |
| Logistic Regression | 86.06% | 76.29% | 51.39% | 61.41% |
| SVM | 91.00% | 70.10% | 68.69% | 69.39% |

**Recommended:** XGBoost (best recall for catching churners)

## 🧪 Predefined Test Cases

1. **High Risk - International Plan** - Customer with international plan and high usage
2. **Very High Risk - Multiple Service Calls** - 6+ service calls, international plan
3. **Low Risk - Voice Mail Subscriber** - Stable customer with voice mail
4. **Moderate Risk - High Bills** - High charges without international plan
5. **Low Risk - Stable Customer** - Long tenure, voice mail, low service calls
6. **High Risk - Heavy Usage** - Very high usage across all periods
7. **Moderate Risk - Some Service Calls** - 2 service calls, moderate usage
8. **Very Low Risk - Ideal Customer** - Long tenure, voice mail, no issues
9. **High Risk - International + High Bills** - Combination of risk factors
10. **Average Customer - Medium Risk** - Typical usage patterns

## 📡 API Usage Examples

### Using cURL

```bash
# Get all available models
curl http://localhost:8000/models

# Get test cases
curl http://localhost:8000/test-cases

# Get specific test case
curl http://localhost:8000/test-case/High%20Risk%20-%20International%20Plan

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "XGBoost",
    "customer_data": {
      "account_length": 128,
      "voice_mail_plan": 1,
      "voice_mail_messages": 25,
      "customer_service_calls": 1,
      "international_plan": 0,
      "day_calls": 110,
      "day_charge": 45.07,
      "evening_calls": 99,
      "evening_charge": 16.78,
      "night_calls": 91,
      "night_charge": 11.01,
      "international_calls": 3,
      "international_charge": 2.70,
      "total_charge": 75.56
    }
  }'
```

### Using Python Requests

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Customer data
payload = {
    "model": "XGBoost",
    "customer_data": {
        "account_length": 128,
        "voice_mail_plan": 1,
        "voice_mail_messages": 25,
        "customer_service_calls": 1,
        "international_plan": 0,
        "day_calls": 110,
        "day_charge": 45.07,
        "evening_calls": 99,
        "evening_charge": 16.78,
        "night_calls": 91,
        "night_charge": 11.01,
        "international_calls": 3,
        "international_charge": 2.70,
        "total_charge": 75.56
    }
}

# Make request
response = requests.post(url, json=payload)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Churn Probability: {result['churn_probability']:.2f}%")
print(f"Risk Level: {result['risk_level']}")
```

### Using JavaScript (Fetch)

```javascript
const url = 'http://localhost:8000/predict';

const data = {
  model: 'XGBoost',
  customer_data: {
    account_length: 128,
    voice_mail_plan: 1,
    voice_mail_messages: 25,
    customer_service_calls: 1,
    international_plan: 0,
    day_calls: 110,
    day_charge: 45.07,
    evening_calls: 99,
    evening_charge: 16.78,
    night_calls: 91,
    night_charge: 11.01,
    international_calls: 3,
    international_charge: 2.70,
    total_charge: 75.56
  }
};

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
  console.log('Prediction:', result.prediction);
  console.log('Churn Probability:', result.churn_probability);
  console.log('Risk Level:', result.risk_level);
});
```

## 🔍 API Endpoints (FastAPI)

### GET /models
Get list of available models with metrics

**Response:**
```json
{
  "models": ["XGBoost", "LightGBM", "Random Forest", "Logistic Regression", "SVM"],
  "metrics": { ... },
  "default_model": "XGBoost"
}
```

### GET /test-cases
List all predefined test cases

**Response:**
```json
{
  "test_cases": ["High Risk - International Plan", ...],
  "count": 10
}
```

### GET /test-case/{case_name}
Get specific test case data

**Response:**
```json
{
  "name": "High Risk - International Plan",
  "data": {
    "account_length": 120,
    "voice_mail_plan": 0,
    ...
  }
}
```

### POST /predict
Make a churn prediction

**Request:**
```json
{
  "model": "XGBoost",
  "customer_data": { ... }
}
```

**Response:**
```json
{
  "model": "XGBoost",
  "prediction": "Churn",
  "churn_probability": 85.67,
  "no_churn_probability": 14.33,
  "risk_level": "Very High Risk",
  "model_metrics": {
    "accuracy": 96.25,
    "recall": 81.44,
    "precision": 91.86,
    "f1": 86.34
  },
  "recommendations": [
    "⚠️ HIGH PRIORITY: Contact customer within 24 hours",
    "Offer retention discount or plan optimization",
    ...
  ]
}
```

## 🎨 Flask Web Interface Features

### Model Selection
- Dropdown to select from 5 models
- Live display of model metrics (accuracy, recall, precision, F1)

### Test Case Auto-Fill
- Select from 10 predefined customer profiles
- Automatically fills all form fields
- Quick testing without manual data entry

### Input Form
- 14 customer feature inputs
- Validation for all fields
- Clear labels and help text

### Results Display
- Visual churn probability (percentage)
- Risk level badge with color coding
- Prediction confidence meters
- Actionable recommendations for high-risk customers

## 🔒 Security Considerations

### For Production Deployment:
1. **Add Authentication**
   - API keys for FastAPI
   - User authentication for Flask

2. **Rate Limiting**
   - Prevent API abuse
   - Implement request throttling

3. **HTTPS**
   - Use SSL/TLS certificates
   - Encrypt data in transit

4. **Input Validation**
   - Strict data validation
   - Sanitize user inputs

5. **CORS Configuration**
   - Restrict allowed origins
   - Configure proper CORS headers

## 📦 Deployment Options

### Docker (Recommended)

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# For Flask
CMD ["python", "deployment/app_flask.py"]

# OR for FastAPI
# CMD ["uvicorn", "deployment.app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Platforms
- **AWS:** EC2, Elastic Beanstalk, Lambda
- **Google Cloud:** Cloud Run, App Engine
- **Azure:** App Service, Container Instances
- **Heroku:** Web dynos

### Gunicorn (Production WSGI Server)

```bash
# Flask
gunicorn -w 4 -b 0.0.0.0:5000 deployment.app_flask:app

# FastAPI
gunicorn -w 4 -k uvicorn.workers.UvicornWorker deployment.app_fastapi:app
```

## 🐛 Troubleshooting

### Models Not Loading
```bash
# Check models directory exists
ls ../models/

# Verify model files are present
ls ../models/*.pkl
```

### Port Already in Use
```bash
# Change port in code or kill existing process
# Flask: app.run(port=5001)
# FastAPI: uvicorn app_fastapi:app --port 8001
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📊 Performance Monitoring

### Metrics to Track:
- Request latency
- Prediction accuracy in production
- Model drift over time
- API error rates
- User engagement

### Logging
Both apps include basic logging. For production:
- Use structured logging (JSON)
- Integrate with monitoring services (DataDog, New Relic)
- Set up alerts for errors

## 🔄 Model Updates

To update models:
1. Train new models
2. Save to `models/` directory
3. Update `AVAILABLE_MODELS` dict in code
4. Update `MODEL_METRICS` with new performance
5. Restart application

## 📞 Support

For issues or questions:
- Check logs for error details
- Review API documentation at `/docs`
- Verify model files are accessible
- Ensure all dependencies are installed

---

**Version:** 1.0.0
**Last Updated:** February 6, 2026
**License:** Internal Use
