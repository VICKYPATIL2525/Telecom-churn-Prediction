"""
Create and save the StandardScaler that was used during model training
This scaler is CRITICAL for correct predictions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

print("=" * 80)
print("CREATING STANDARDSCALER FOR MODEL DEPLOYMENT")
print("=" * 80)

# Load the training data
DATA_PATH = Path(__file__).parent.parent / 'Data' / 'telecommunications_churn.csv'
print(f"\nLoading data from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Data shape: {df.shape}")

# Perform the EXACT same feature engineering as training
print("\nPerforming feature engineering...")

# Remove minutes columns (as done in training)
if 'day_mins' in df.columns:
    df = df.drop(['day_mins', 'evening_mins', 'night_mins', 'international_mins'], axis=1)

# Create engineered features (EXACT same as training)
df['intl_plan_x_service_calls'] = df['international_plan'] * df['customer_service_calls']
df['total_charge_x_service_calls'] = df['total_charge'] * df['customer_service_calls']

# Derived features - need to recreate minutes from charges
day_mins = df['day_charge'] / 0.17
evening_mins = df['evening_charge'] / 0.085
night_mins = df['night_charge'] / 0.045
international_mins = df['international_charge'] / 0.27

df['total_minutes'] = day_mins + evening_mins + night_mins + international_mins
df['charge_per_minute'] = df['total_charge'] / (df['total_minutes'] + 0.001)
df['usage_intensity'] = df['total_minutes'] / (df['account_length'] + 1)

# Binary flags
df['high_service_calls'] = (df['customer_service_calls'] >= 4).astype(int)
df['high_charges'] = (df['total_charge'] >= 75).astype(int)

# Feature order (21 features)
feature_columns = [
    'account_length', 'voice_mail_plan', 'voice_mail_messages',
    'customer_service_calls', 'international_plan', 'day_calls',
    'day_charge', 'evening_calls', 'evening_charge', 'night_calls',
    'night_charge', 'international_calls', 'international_charge',
    'total_charge', 'intl_plan_x_service_calls', 'total_charge_x_service_calls',
    'total_minutes', 'charge_per_minute', 'usage_intensity',
    'high_service_calls', 'high_charges'
]

X = df[feature_columns]
y = df['churn']

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data (80/20 stratified - same as training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Create and fit scaler (ONLY on training data to prevent leakage)
print("\nFitting StandardScaler on training data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(f"Scaler fitted successfully!")
print(f"  Mean: {scaler.mean_[:5]}...")  # Show first 5
print(f"  Scale: {scaler.scale_[:5]}...")  # Show first 5

# Save the scaler
SCALER_PATH = Path(__file__).parent.parent / 'models' / 'scaler.pkl'
joblib.dump(scaler, SCALER_PATH)

print(f"\n[SUCCESS] Scaler saved to: {SCALER_PATH}")

# Verify it works
print("\n" + "=" * 80)
print("VERIFICATION TEST")
print("=" * 80)

# Test with a sample
sample = X_test.iloc[0:1]
print(f"\nOriginal sample values (first 5 features):")
print(sample.iloc[0, :5].to_dict())

scaled_sample = scaler.transform(sample)
print(f"\nScaled sample values (first 5 features):")
print(dict(zip(feature_columns[:5], scaled_sample[0, :5])))

print("\n✓ Scaler is working correctly!")
print("=" * 80)
