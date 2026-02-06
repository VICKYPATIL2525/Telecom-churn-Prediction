# Telecom Churn Prediction - Machine Learning Project

## 🎯 Project Overview

A comprehensive machine learning project to predict customer churn in the telecommunications industry. The project includes:
- **Exploratory Data Analysis (EDA)** with insights and visualizations
- **Feature Engineering** with 7 new engineered features
- **Model Building** with 5 algorithms (XGBoost, Random Forest, LightGBM, Logistic Regression, SVM)
- **Class Imbalance Handling** using SMOTE and class weights
- **Threshold Optimization** for maximum business value
- **Business Impact Analysis** with ROI calculations
- **Production-Ready Deployment** with Flask web application

---

## ⭐ Key Highlights

| Highlight | Details |
|-----------|---------|
| **🏆 Best Model** | XGBoost with 96.25% accuracy and 81.44% recall |
| **💰 Business Value** | $205,607 annual savings potential (123% ROI) |
| **🎯 Churn Detection** | Catches 79 out of 97 churners (81.4% success rate) |
| **🚀 Deployment** | Production-ready Flask web app with 10 test cases |
| **📊 Data Size** | 3,333 customers with 21 features (14 original + 7 engineered) |
| **⚖️ Class Imbalance** | Successfully handled 5.9:1 imbalance using SMOTE |
| **🔍 Top Predictor** | Customer service calls (4+ calls = 60% churn risk) |
| **📈 Models Trained** | 5 algorithms compared (XGBoost, LightGBM, RF, LR, SVM) |

---

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Customers** | 3,333 |
| **Original Features** | 18 |
| **Engineered Features** | 7 |
| **Total Features Used** | 21 |
| **Churn Rate** | 14.49% (483 churned, 2,850 retained) |
| **Class Imbalance Ratio** | 5.9:1 (No Churn : Churn) |
| **Data Quality** | ✓ No missing values |

---

## 🏆 Model Performance Summary

### Best Model: **XGBoost**

| Metric | Score |
|--------|-------|
| **Recall** ⭐ | 81.4% (catches 79 out of 97 churners) |
| **Precision** | 91.9% (only 7 false alarms) |
| **F1-Score** | 86.3% (excellent balance) |
| **ROC-AUC** | 92.0% (strong discrimination) |
| **Accuracy** | 96.3% |
| **False Negatives** | 18 (missed churners) |
| **Annual Savings** | $205,607 |
| **ROI** | 123.4% |

### Model Comparison (All 5 Models)

| Model | Recall | Precision | F1-Score | ROC-AUC | Accuracy |
|-------|--------|-----------|----------|---------|----------|
| **XGBoost** | **0.8144** | **0.9186** | **0.8634** | **0.9196** | **0.9625** |
| LightGBM | 0.7835 | 0.9383 | 0.8539 | 0.9108 | 0.9610 |
| Random Forest | 0.7835 | 0.9268 | 0.8492 | 0.9007 | 0.9595 |
| Logistic Regression | 0.7629 | 0.5139 | 0.6141 | 0.8734 | 0.8606 |
| SVM | 0.7010 | 0.6869 | 0.6939 | 0.8978 | 0.9100 |

---

## 🔑 Key Findings from EDA

### Top 3 Churn Predictors

1. **Customer Service Calls** (Most Powerful)
   - 4+ calls = 60% churn probability
   - Critical intervention point at 3rd call

2. **International Plan** (Correlation: 0.26)
   - Customers with international plan have **3.5-4x higher churn rate**
   - Most critical service issue to address

3. **Total Charges** (Correlation: 0.23)
   - Price sensitivity evident
   - Customers with bills >$75 show elevated churn risk

### Protective Factor

- **Voice Mail Plan:** Reduces churn by ~50%
- Strong loyalty indicator for customers who adopt it

### High-Risk Customer Profile

Customers most likely to churn:
- 4+ customer service calls (60% churn rate)
- International plan subscriber (40% churn rate)
- Monthly charges >$75
- High daytime usage (>220 minutes)
- Newer customers (account length <6 months)

### Business Impact & Recommendations

- **Potential Annual Savings:** $205,607 (with current model)
- **Break-even:** Just 6 retained customers per month
- **Priority Action 1:** Review international plan pricing and service quality
- **Priority Action 2:** Implement service call alert system (trigger after 3 calls)
- **Priority Action 3:** Deploy ML model for automated churn prediction

---

## ⚠️ Class Imbalance Challenge & Solution

### The Challenge
- **Class Distribution:** 85.51% No Churn vs 14.49% Churn
- **Imbalance Ratio:** 5.9:1 (majority:minority)
- Without proper handling: Models predict "No Churn" for everyone, achieve 85% accuracy but miss ALL churners!

### Solution Implemented

✅ **SMOTE (Synthetic Minority Over-sampling)**
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# Reduces imbalance from 5.9:1 to 2:1
```

✅ **Class Weights**
```python
# Applied to all models
class_weight='balanced'  # Automatically adjusts weights
# Churners get 5.9x higher importance
```

✅ **Stratified Sampling**
```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# Maintains 14.49% churn rate in both train and test sets
```

✅ **Recall-Focused Evaluation**
- Primary metric: **Recall** (catch 80%+ of churners)
- Secondary metrics: Precision, F1-Score, ROC-AUC
- Business metric: Cost-benefit analysis

✅ **Threshold Tuning**
- Default threshold 0.5 is suboptimal for imbalanced data
- Optimal threshold: 0.35-0.40 for maximum business value
- Business justification: Retention cost ($50) << Customer lifetime value ($768)
- ROI = 15.4:1 → Can afford false positives to catch real churners

---

## 🛠️ Feature Engineering

### Original Features
- Account length, voice mail plan, international plan
- Day/evening/night/international call minutes and charges
- Customer service calls

### Engineered Features (7 new)

1. **Interaction Features**
   - `intl_plan_x_service_calls` - Combined effect of international plan and support calls
   - `total_charge_x_service_calls` - Combined effect of billing and support issues

2. **Derived Features**
   - `total_minutes` - Sum of all calling minutes
   - `charge_per_minute` - Efficiency metric (usage value ratio)
   - `usage_intensity` - Activity level per account month

3. **Binary Flags**
   - `high_service_calls` - Flag for 4+ customer service calls
   - `high_charges` - Flag for charges ≥$75

### Features Removed (Multicollinearity)
- `day_mins`, `evening_mins`, `night_mins`, `international_mins`
- Reason: 0.88+ correlation with corresponding charge features (data redundancy)

---

## 📁 Project Structure

```
Telecom-churn-Prediction/
├── README.md                                      # This file
├── requirements.txt                               # Python dependencies
├── .gitignore                                     # Git ignore patterns
├── Model_Building_Documentation.docx              # Model building documentation
├── data/
│   ├── telecommunications_churn.csv               # Dataset (3,333 customers)
│   └── Business_Requirement_P641.docx             # Business requirements document
├── notebooks/
│   ├── EDA_Telecom_Churn.ipynb                   # Exploratory Data Analysis
│   └── Model_Building_Churn_Prediction.ipynb     # Model building & evaluation
├── models/
│   ├── model_xgboost_acc_0.9625.pkl              # Best model (XGBoost)
│   ├── model_lightgbm_acc_0.9610.pkl             # 2nd best (LightGBM)
│   ├── model_random_forest_acc_0.9595.pkl        # 3rd best (Random Forest)
│   ├── model_logistic_regression_acc_0.8606.pkl  # Baseline (Logistic Regression)
│   ├── model_svm_rbf_acc_0.9100.pkl              # SVM model
│   └── scaler.pkl                                # StandardScaler for feature scaling
├── deployment/
│   ├── app.py                                     # Flask web application
│   ├── create_scaler.py                           # Script to create scaler
│   ├── README.md                                  # Deployment documentation
│   └── templates/
│       └── index.html                             # Web interface template
├── results/
│   ├── EDA_Report_Telecom_Churn.md                # Detailed EDA report (markdown)
│   ├── EDA_Report_Telecom_Churn.docx              # EDA report (Word format)
│   └── model_comparison_results.csv               # Model comparison metrics
└── myenv/                                         # Virtual environment
```

---

## 💰 Business Impact Analysis

### Financial Metrics (Test Set)

| Metric | Value |
|--------|-------|
| Retention offer cost | $50 per customer |
| Average monthly revenue | $64 per customer |
| Customer lifetime value | $768 (12 months) |
| ROI ratio | 15.4:1 |

### XGBoost Model Performance Impact

| Metric | Value |
|--------|-------|
| Churners caught | 79 out of 97 |
| Catch rate | 81.4% |
| Churners missed | 18 |
| False alarms | 7 |
| **Revenue saved** | $74,496 |
| **Revenue lost** | $13,824 |
| **Retention costs** | $4,300 |
| **Net benefit (test set)** | $56,372 |
| **ROI** | 1,211% |

### Extrapolated Annual Impact

Deploying XGBoost across the full customer base:
- **Total customers:** 3,333
- **Expected annual benefit:** $205,607
- **Break-even:** Just 6 retained customers
- **Payback period:** Immediate (first month)
- **Confidence level:** High (validated on holdout test set)

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Programming** | Python 3.13.2 |
| **Data Analysis** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Machine Learning** | scikit-learn 1.6.0, XGBoost 2.1.3, LightGBM 4.5.0 |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |
| **Web Framework** | Flask 3.1.0 |
| **Model Persistence** | joblib |
| **Development** | Jupyter Notebook, ipykernel |
| **Documentation** | python-docx, Markdown |

---

## 🚀 Setup & Installation

### Environment Setup

1. **Create Virtual Environment:**
   ```powershell
   python -m venv myenv
   .\myenv\Scripts\Activate.ps1
   ```

2. **Upgrade pip:**
   ```powershell
   python -m pip install --upgrade pip
   ```

3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

### Important Technical Notes

- **Python Version:** 3.13.2
- **numpy:** 2.1.3 (has pre-built wheels for Python 3.13.2)
- **scikit-learn:** 1.6.0 (compatible with numpy 2.x)
- **All packages:** Version-pinned (==) for reproducibility

### Dependencies

Core packages included:
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **ML:** scikit-learn, xgboost, lightgbm, imbalanced-learn
- **Jupyter:** notebook, ipykernel, ipython
- **Documentation:** python-docx

See `requirements.txt` for complete list with versions.

---

## 📚 Usage Guide

### 1. Run the Flask Web Application (Easiest)

```bash
# Navigate to deployment folder
cd deployment

# Run the web app
python app.py

# Access the web interface at http://localhost:5000
```

The Flask app provides:
- Interactive web interface for predictions
- 10 predefined test cases for quick testing
- Multiple model selection
- Visual results with risk level indicators
- Actionable recommendations

See `deployment/README.md` for detailed deployment guide.

### 2. Load and Use Models Programmatically

```python
import joblib
import pandas as pd

# Load artifacts
model = joblib.load('models/model_xgboost_acc_0.9625.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load your data
df = pd.read_csv('data/telecommunications_churn.csv')

# Preprocess and scale features
X = preprocess(df)  # Apply same preprocessing as training
X_scaled = scaler.transform(X)

# Get predictions
probabilities = model.predict_proba(X_scaled)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# Results
churn_risk_scores = pd.DataFrame({
    'customer_id': df.index,
    'churn_probability': probabilities,
    'predicted_churn': predictions
})
```

### 3. Reproduce Analysis

**For EDA:**
```bash
jupyter notebook notebooks/EDA_Telecom_Churn.ipynb
```

**For Model Building:**
```bash
jupyter notebook notebooks/Model_Building_Churn_Prediction.ipynb
```

---

## 📊 Key Methodology

### Data Preprocessing
1. **Stratified Train-Test Split** (80/20) - maintains 14.49% churn rate in both sets
2. **StandardScaler** - normalizes features (mean=0, std=1)
3. **SMOTE** - reduces imbalance from 5.9:1 to 2:1 ratio
4. **Multicollinearity Removal** - drops correlated minute features

### Model Training
1. **5 Algorithms Tested:**
   - XGBoost (best)
   - LightGBM
   - Random Forest
   - Logistic Regression
   - SVM

2. **Class Imbalance Handling:**
   - SMOTE applied to training data only
   - Class weights (balanced) in all models
   - Stratified cross-validation

3. **Threshold Optimization:**
   - Grid search over probability thresholds
   - Optimized for maximum recall
   - Business cost-benefit analysis

### Model Evaluation
- **Primary Metric:** Recall (catch 80%+ of churners)
- **Secondary Metrics:** Precision, F1-Score, ROC-AUC
- **Business Metric:** Revenue saved vs. retention costs

---

## 🚀 Deployment

### Flask Web Application

A production-ready Flask web application is available in the `deployment/` folder with the following features:

✅ **Interactive Web Interface**
- User-friendly form for customer data input
- Real-time churn prediction with probability scores
- Risk level assessment (Very Low, Low, Moderate, High, Very High)
- Actionable business recommendations

✅ **Multiple Model Support**
- Choose from 5 trained models (XGBoost recommended)
- Live display of model performance metrics
- Model comparison capabilities

✅ **10 Predefined Test Cases**
- Quick testing with realistic customer profiles
- Auto-fill functionality for faster testing
- Covers full risk spectrum from Very Low to Very High

### Quick Start

```bash
# Navigate to deployment folder
cd deployment

# Run the Flask app
python app.py

# Access at http://localhost:5000
```

### Deployment Features

| Feature | Description |
|---------|-------------|
| **Models Available** | XGBoost, LightGBM, Random Forest, Logistic Regression, SVM |
| **Input Method** | Form-based with validation |
| **Output** | Churn probability, risk level, recommendations |
| **Test Cases** | 10 predefined customer profiles |
| **Performance** | Real-time prediction (<100ms) |

### Production Deployment Options

The Flask app can be deployed to:
- **Docker** - Containerized deployment
- **AWS** - EC2, Elastic Beanstalk, Lambda
- **Google Cloud** - Cloud Run, App Engine
- **Azure** - App Service, Container Instances
- **Heroku** - Web dynos

See `deployment/README.md` for detailed deployment instructions, API documentation, and production deployment guides.

---

## 🎯 Project Status & Roadmap

### ✅ Completed Phases

- [x] **Phase 0: EDA** - Comprehensive analysis with 50+ visualizations
- [x] **Phase 1: Data Preprocessing** - Feature engineering, scaling, stratified split
- [x] **Phase 2: Model Building** - 5 algorithms with SMOTE + class weights
- [x] **Phase 3: Model Evaluation** - Comprehensive metrics and comparison
- [x] **Phase 4: Threshold Optimization** - Precision-recall curve analysis
- [x] **Phase 5: Business Impact Analysis** - ROI calculations and recommendations
- [x] **Phase 6: Web Application Deployment** - Flask app with interactive UI

### 📋 Recommended Next Steps (Optional Enhancements)

**Phase 7: Model Optimization (1-2 weeks)**
- [ ] Hyperparameter tuning using GridSearchCV or BayesianOptimization
- [ ] Ensemble methods (Stacking, Voting Classifier)
- [ ] Cross-validation for more robust estimates
- [ ] Cost-sensitive learning with explicit false negative penalties

**Phase 8: Advanced Deployment (2-3 weeks)**
- [ ] Build REST API (FastAPI) for real-time churn prediction
- [ ] Deploy to production cloud platform (AWS/Azure/GCP)
- [ ] Add authentication and rate limiting
- [ ] Integrate with CRM system
- [ ] Create real-time churn risk dashboard
- [ ] Implement batch prediction pipeline

**Phase 9: Monitoring & Maintenance (Ongoing)**
- [ ] Monitor recall and precision trends
- [ ] Track business KPIs (retention rate, revenue saved)
- [ ] Implement model drift detection
- [ ] Retrain model quarterly with new data
- [ ] A/B test retention strategies
- [ ] Analyze misclassifications for improvement
- [ ] Set up automated alerts for high-risk customers

---

## 💡 Business Recommendations by Customer Segment

### High Service Calls Segment (4+ calls)
- **Churn Rate:** 60%
- **Actions:**
  - Immediate escalation to retention specialist
  - Service quality investigation and remediation
  - Proactive outreach after 3rd call (preventive)
  - VIP support for high-value accounts

### International Plan Customers
- **Churn Rate:** 40%
- **Actions:**
  - Competitive rate analysis and price matching
  - International calling bundles and discounts
  - Education on plan benefits
  - Loyalty program for frequent international callers

### High Bill Customers (>$75/month)
- **Churn Rate:** 25%+
- **Actions:**
  - Proactive billing review and optimization
  - Loyalty discounts or price adjustments
  - Alternative plan recommendations
  - Usage alerts to prevent bill shock

---

## 📖 Documentation Files

| File | Description |
|------|-------------|
| `README.md` | This file - project overview and guide |
| `requirements.txt` | Python dependencies and versions |
| `notebooks/EDA_Telecom_Churn.ipynb` | Detailed exploratory data analysis (50+ visualizations) |
| `notebooks/Model_Building_Churn_Prediction.ipynb` | Model building, evaluation, and analysis |
| `results/EDA_Report_Telecom_Churn.md` | Comprehensive EDA report (markdown) |
| `results/EDA_Report_Telecom_Churn.docx` | EDA report (Word format) |
| `Model_Building_Documentation.docx` | Model building documentation (Word) |
| `deployment/README.md` | Deployment guide for Flask web app |
| `data/telecommunications_churn.csv` | Dataset (3,333 customers, 19 features) |
| `data/Business_Requirement_P641.docx` | Business requirements document |

---

## 🔍 Model Comparison Details

### Why XGBoost Won

1. **Highest Recall:** 81.4% (catches 79 of 97 churners)
2. **Best F1-Score:** 86.3% (optimal precision-recall balance)
3. **Strong ROC-AUC:** 92.0% (excellent discrimination)
4. **Lowest False Negatives:** 18 missed churners (vs. 21-29 for other models)
5. **High Precision:** 91.9% (minimal false alarms)
6. **Robust:** Handles imbalance, multicollinearity, and feature importance

### Trade-offs Considered

- **Recall vs. Precision:** Higher recall (catching more churners) increases false positives
- **Accuracy vs. Business Value:** High accuracy (96%+) doesn't mean catching churners
- **Model Complexity:** Tree-based models more complex but vastly superior performance
- **Threshold Selection:** Default 0.5 threshold sacrifices recall; optimal threshold improves business outcome

---

## 🔧 Troubleshooting

### Virtual Environment Issues
- **numpy compilation error:** Use numpy==2.1.3 (has pre-built wheels)
- **Missing dependencies:** Run `pip install -r requirements.txt --upgrade`

### Model Loading Issues
- Ensure joblib is installed: `pip install joblib`
- Verify model file path is correct
- Check Python version matches training environment (3.13.2+)

### Jupyter Kernel Issues
- Activate virtual environment before launching Jupyter
- Install ipykernel in the virtual environment

---

## 📧 Contact & Support

For questions about this project:
- Review the detailed notebooks in `Notebooks/`
- Check documentation in `results/`
- Examine the source code in model building scripts

---

## 📝 Project Notes

### Version History
- **v1.0** - Initial release with 5 ML models and comprehensive EDA
- **v1.1** - Added Flask web application for deployment
- **Current Version:** 1.1
- **Last Updated:** February 6, 2026

### Project Status
- **Status:** ✅ Production-ready with deployment
- **Best Model:** XGBoost (96.25% accuracy, 81.44% recall)
- **Deployment:** Flask web app available
- **Documentation:** Comprehensive (README, notebooks, Word docs)

### Maintenance Recommendations
- **Model Retraining:** Quarterly with new data
- **Performance Monitoring:** Track recall, precision, and business metrics
- **Data Quality:** Validate input data consistency
- **Model Drift:** Monitor prediction distribution changes
- **A/B Testing:** Test retention strategies based on predictions

---

*This project successfully demonstrates how machine learning can deliver significant business value (123% ROI) even with challenging class imbalance problems.*
