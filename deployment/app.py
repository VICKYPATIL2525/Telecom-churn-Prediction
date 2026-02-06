"""
Flask Web Application for Telecom Churn Prediction
Supports multiple ML models with interactive web interface
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
SCALER_PATH = MODELS_DIR / 'scaler.pkl'

AVAILABLE_MODELS = {
    'XGBoost': 'model_xgboost_acc_0.9625.pkl',
    'LightGBM': 'model_lightgbm_acc_0.9610.pkl',
    'Random Forest': 'model_random_forest_acc_0.9595.pkl',
    'Logistic Regression': 'model_logistic_regression_acc_0.8606.pkl',
    'SVM': 'model_svm_rbf_acc_0.9100.pkl'
}

MODEL_METRICS = {
    'XGBoost': {'accuracy': 96.25, 'recall': 81.44, 'precision': 91.86, 'f1': 86.34},
    'LightGBM': {'accuracy': 96.10, 'recall': 78.35, 'precision': 93.83, 'f1': 85.39},
    'Random Forest': {'accuracy': 95.95, 'recall': 78.35, 'precision': 92.68, 'f1': 84.92},
    'Logistic Regression': {'accuracy': 86.06, 'recall': 76.29, 'precision': 51.39, 'f1': 61.41},
    'SVM': {'accuracy': 91.00, 'recall': 70.10, 'precision': 68.69, 'f1': 69.39}
}

TEST_CASES = {
    'High Risk - International Plan': {
        'account_length': 120, 'voice_mail_plan': 0, 'voice_mail_messages': 0,
        'customer_service_calls': 4, 'international_plan': 1, 'day_calls': 100,
        'day_charge': 45.5, 'evening_calls': 90, 'evening_charge': 18.2,
        'night_calls': 85, 'night_charge': 10.5, 'international_calls': 8,
        'international_charge': 3.8, 'total_charge': 78.0
    },
    'Very High Risk - Multiple Service Calls': {
        'account_length': 85, 'voice_mail_plan': 0, 'voice_mail_messages': 0,
        'customer_service_calls': 6, 'international_plan': 1, 'day_calls': 110,
        'day_charge': 52.0, 'evening_calls': 100, 'evening_charge': 20.5,
        'night_calls': 95, 'night_charge': 11.8, 'international_calls': 10,
        'international_charge': 4.2, 'total_charge': 88.5
    },
    'Low Risk - Voice Mail Subscriber': {
        'account_length': 180, 'voice_mail_plan': 1, 'voice_mail_messages': 25,
        'customer_service_calls': 1, 'international_plan': 0, 'day_calls': 95,
        'day_charge': 28.5, 'evening_calls': 105, 'evening_charge': 15.8,
        'night_calls': 110, 'night_charge': 9.2, 'international_calls': 3,
        'international_charge': 2.1, 'total_charge': 55.6
    },
    'Moderate Risk - High Bills': {
        'account_length': 95, 'voice_mail_plan': 0, 'voice_mail_messages': 0,
        'customer_service_calls': 2, 'international_plan': 0, 'day_calls': 125,
        'day_charge': 48.0, 'evening_calls': 115, 'evening_charge': 19.5,
        'night_calls': 100, 'night_charge': 10.8, 'international_calls': 4,
        'international_charge': 2.5, 'total_charge': 80.8
    },
    'Low Risk - Stable Customer': {
        'account_length': 200, 'voice_mail_plan': 1, 'voice_mail_messages': 30,
        'customer_service_calls': 0, 'international_plan': 0, 'day_calls': 88,
        'day_charge': 25.0, 'evening_calls': 92, 'evening_charge': 14.2,
        'night_calls': 95, 'night_charge': 8.5, 'international_calls': 2,
        'international_charge': 1.8, 'total_charge': 49.5
    },
    'High Risk - Heavy Usage': {
        'account_length': 75, 'voice_mail_plan': 0, 'voice_mail_messages': 0,
        'customer_service_calls': 3, 'international_plan': 0, 'day_calls': 145,
        'day_charge': 55.5, 'evening_calls': 130, 'evening_charge': 22.0,
        'night_calls': 120, 'night_charge': 12.5, 'international_calls': 5,
        'international_charge': 2.8, 'total_charge': 92.8
    },
    'Moderate Risk - Some Service Calls': {
        'account_length': 110, 'voice_mail_plan': 1, 'voice_mail_messages': 15,
        'customer_service_calls': 2, 'international_plan': 0, 'day_calls': 102,
        'day_charge': 32.0, 'evening_calls': 98, 'evening_charge': 16.5,
        'night_calls': 105, 'night_charge': 9.8, 'international_calls': 3,
        'international_charge': 2.2, 'total_charge': 60.5
    },
    'Very Low Risk - Ideal Customer': {
        'account_length': 225, 'voice_mail_plan': 1, 'voice_mail_messages': 35,
        'customer_service_calls': 0, 'international_plan': 0, 'day_calls': 82,
        'day_charge': 22.5, 'evening_calls': 85, 'evening_charge': 13.0,
        'night_calls': 88, 'night_charge': 7.8, 'international_calls': 2,
        'international_charge': 1.5, 'total_charge': 44.8
    },
    'High Risk - International + High Bills': {
        'account_length': 65, 'voice_mail_plan': 0, 'voice_mail_messages': 0,
        'customer_service_calls': 3, 'international_plan': 1, 'day_calls': 135,
        'day_charge': 50.0, 'evening_calls': 120, 'evening_charge': 20.0,
        'night_calls': 110, 'night_charge': 11.5, 'international_calls': 12,
        'international_charge': 4.5, 'total_charge': 86.0
    },
    'Average Customer - Medium Risk': {
        'account_length': 101, 'voice_mail_plan': 0, 'voice_mail_messages': 0,
        'customer_service_calls': 1, 'international_plan': 0, 'day_calls': 100,
        'day_charge': 30.5, 'evening_calls': 100, 'evening_charge': 17.1,
        'night_calls': 100, 'night_charge': 9.0, 'international_calls': 4,
        'international_charge': 2.8, 'total_charge': 59.4
    }
}

print(f"Loading scaler from: {SCALER_PATH}")
scaler = joblib.load(SCALER_PATH)
print("Scaler loaded successfully!")

def load_model(model_name):
    """Load the selected model"""
    model_path = MODELS_DIR / AVAILABLE_MODELS[model_name]
    return joblib.load(model_path)

def prepare_features(data):
    """Prepare features with engineering and scaling"""
    df = pd.DataFrame([data])

    df['intl_plan_x_service_calls'] = df['international_plan'] * df['customer_service_calls']
    df['total_charge_x_service_calls'] = df['total_charge'] * df['customer_service_calls']

    day_mins = df['day_charge'] / 0.17
    evening_mins = df['evening_charge'] / 0.085
    night_mins = df['night_charge'] / 0.045
    international_mins = df['international_charge'] / 0.27

    df['total_minutes'] = day_mins + evening_mins + night_mins + international_mins
    df['charge_per_minute'] = df['total_charge'] / (df['total_minutes'] + 0.001)
    df['usage_intensity'] = df['total_minutes'] / (df['account_length'] + 1)

    df['high_service_calls'] = (df['customer_service_calls'] >= 4).astype(int)
    df['high_charges'] = (df['total_charge'] >= 75).astype(int)

    feature_order = [
        'account_length', 'voice_mail_plan', 'voice_mail_messages',
        'customer_service_calls', 'international_plan', 'day_calls',
        'day_charge', 'evening_calls', 'evening_charge', 'night_calls',
        'night_charge', 'international_calls', 'international_charge',
        'total_charge', 'intl_plan_x_service_calls', 'total_charge_x_service_calls',
        'total_minutes', 'charge_per_minute', 'usage_intensity',
        'high_service_calls', 'high_charges'
    ]

    features = df[feature_order]
    features_scaled = scaler.transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=feature_order)

    return features_scaled_df

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html',
                         models=AVAILABLE_MODELS.keys(),
                         test_cases=TEST_CASES.keys(),
                         metrics=MODEL_METRICS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        model_name = request.form.get('model')

        input_data = {
            'account_length': int(request.form.get('account_length')),
            'voice_mail_plan': int(request.form.get('voice_mail_plan')),
            'voice_mail_messages': int(request.form.get('voice_mail_messages')),
            'customer_service_calls': int(request.form.get('customer_service_calls')),
            'international_plan': int(request.form.get('international_plan')),
            'day_calls': int(request.form.get('day_calls')),
            'day_charge': float(request.form.get('day_charge')),
            'evening_calls': int(request.form.get('evening_calls')),
            'evening_charge': float(request.form.get('evening_charge')),
            'night_calls': int(request.form.get('night_calls')),
            'night_charge': float(request.form.get('night_charge')),
            'international_calls': int(request.form.get('international_calls')),
            'international_charge': float(request.form.get('international_charge')),
            'total_charge': float(request.form.get('total_charge'))
        }

        model = load_model(model_name)
        features = prepare_features(input_data)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        result = {
            'model': model_name,
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'churn_probability': float(probability[1]) * 100,
            'no_churn_probability': float(probability[0]) * 100,
            'risk_level': get_risk_level(probability[1]),
            'metrics': MODEL_METRICS[model_name],
            'input_data': input_data
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_test_case/<case_name>')
def get_test_case(case_name):
    """Return test case data"""
    if case_name in TEST_CASES:
        return jsonify(TEST_CASES[case_name])
    return jsonify({'error': 'Test case not found'}), 404

def get_risk_level(churn_prob):
    """Determine risk level based on churn probability"""
    if churn_prob >= 0.7:
        return 'Very High Risk'
    elif churn_prob >= 0.5:
        return 'High Risk'
    elif churn_prob >= 0.3:
        return 'Moderate Risk'
    elif churn_prob >= 0.15:
        return 'Low Risk'
    else:
        return 'Very Low Risk'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
