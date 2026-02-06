# Telecom-churn-Prediction

## Project Overview
Machine Learning project to predict customer churn in telecommunications industry.

## 🎯 Key Findings from EDA

### Dataset Overview
- **Total Customers:** 3,333
- **Features:** 19 variables
- **Target Variable:** Churn (Binary)
- **Churn Rate:** 14.49% (483 churned, 2,850 retained)
- **Data Quality:** ✓ No missing values

### Top 3 Churn Predictors
1. **International Plan** (correlation: 0.26)
   - Customers with international plan have **3.5-4x higher churn rate**
   - Most critical predictor identified
   
2. **Total Charges** (correlation: 0.23)
   - Price sensitivity evident
   - Customers with bills >$75 show elevated churn risk
   
3. **Customer Service Calls** (correlation: 0.21)
   - Strong dissatisfaction indicator
   - 4+ calls = high-risk churn candidate

### Protective Factor
- **Voice Mail Plan:** Reduces churn by ~50%

### High-Risk Customer Profile
Customers most likely to churn:
- International plan subscriber
- 4+ customer service calls
- Monthly charges >$75
- High daytime usage (>220 minutes)

### Business Impact & ROI
- **Potential Annual Savings:** $94,000-$113,000 (with 20% churn reduction)
- **Priority Action:** Review international plan pricing and service quality
- **Quick Win:** Implement customer service call alert system (trigger after 3 calls)

---

## ⚠️ CRITICAL: Class Imbalance for Machine Learning

### The Challenge
- **Class Distribution:** 85.51% No Churn vs 14.49% Churn
- **Imbalance Ratio:** 5.9:1 (majority:minority)

### Why This Matters
❌ **Without handling imbalance:**
- Models will predict "No Churn" for everyone
- Can achieve 85.51% accuracy but miss ALL churners!
- Business goal = CATCH churners, not just high accuracy

✅ **What We MUST Implement:**

#### 1. Resampling Techniques
```python
# SMOTE - Create synthetic minority samples
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.5)  # Bring churn to 50% of majority

# Or combination approach
from imblearn.combine import SMOTETomek
```

#### 2. Class Weights
```python
# In sklearn models
class_weight='balanced'  # Automatically adjusts weights
# Or manual: {0: 1, 1: 5.9}  # Give churners 5.9x importance
```

#### 3. Evaluation Metrics (NOT Accuracy!)
- ✅ **Recall** (catch 80-90% of churners)
- ✅ **Precision-Recall Curve**
- ✅ **F1-Score**
- ✅ **ROC-AUC**
- ✅ **Confusion Matrix** (focus on false negatives)

#### 4. Stratified Sampling
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

#### 5. Threshold Tuning
- Default threshold = 0.5 may not be optimal
- Lower to 0.3-0.4 to catch more churners
- **Business logic:** Retention cost ($50) << Customer value ($768/year)
- ROI = 15:1 → Can afford false positives to catch real churners

### Recommended Strategy
1. Use SMOTE to balance training data (40-50% minority)
2. Set `class_weight='balanced'` in models
3. Optimize for **Recall** (minimize false negatives)
4. Use stratified 5-fold cross-validation
5. Tune probability threshold based on business cost-benefit analysis

---

## Setup Instructions

### Environment Setup
1. **Create Virtual Environment:**
   ```powershell
   python -m venv myenv
   .\myenv\Scripts\Activate.ps1
   ```

2. **Upgrade pip (Important!):**
   ```powershell
   python -m pip install --upgrade pip
   ```

3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

### Important Notes
- **Python Version:** 3.13.2
- **Key Dependency Fix:** Using numpy==2.1.3 instead of 1.26.4 to avoid compilation errors
  - numpy 1.26.4 doesn't have pre-built wheels for Python 3.13.2
  - numpy 2.1.3 has pre-built wheels - installs without C compiler
- **scikit-learn:** Updated to 1.6.0 for numpy 2.x compatibility
- All packages use version pinning (==) to ensure reproducibility

### Project Structure
```
├── myenv/                             # Virtual environment
├── telecommunications_churn.csv       # Dataset (3,333 customers, 19 features)
├── EDA_Telecom_Churn.ipynb           # Exploratory Data Analysis notebook
├── EDA_Report_Telecom_Churn.docx     # Comprehensive EDA report (32+ pages)
├── EDA_Report_Telecom_Churn.md       # Markdown version of EDA report
├── generate_eda_report.py            # Script to generate Word document
├── requirements.txt                   # Python dependencies (version pinned)
└── README.md                          # This file
```

## Next Steps

### 1. Data Preprocessing & Feature Engineering
- ✓ EDA completed (see EDA_Telecom_Churn.ipynb)
- [ ] Handle class imbalance (SMOTE + class weights)
- [ ] Feature engineering (interaction features, binning)
- [ ] Feature scaling/normalization
- [ ] Remove multicollinear features (keep day_charge OR day_mins)

### 2. Model Building (With Imbalance Handling)
- [ ] Baseline models with `class_weight='balanced'`
  - Logistic Regression
  - Random Forest
  - XGBoost
- [ ] Apply SMOTE to training data
- [ ] Optimize for Recall (>80% target)
- [ ] Stratified 5-fold cross-validation

### 3. Model Evaluation (Focus on Recall!)
- [ ] Confusion Matrix analysis
- [ ] Precision-Recall curves
- [ ] ROC-AUC score
- [ ] F1-Score
- [ ] Cost-benefit analysis (false positive vs false negative)

### 4. Business Integration
- [ ] Churn risk scoring system
- [ ] Retention intervention triggers
- [ ] Customer service call alert system (after 3 calls)
- [ ] International plan customer review

### 5. Deployment & Monitoring
- [ ] Model deployment pipeline
- [ ] Real-time prediction API
- [ ] Performance monitoring dashboard
- [ ] Model retraining schedule

### Priority Actions (Immediate)
1. 🔴 **Review international plan** - Most critical churn driver
2. 🟡 **Implement service call alerts** - Early warning system
3. 🟢 **Build ML models with imbalance handling** - Optimize for recall

---

## Documentation
- **EDA Report:** `EDA_Report_Telecom_Churn.docx` (32+ pages comprehensive analysis)
- **Jupyter Notebook:** `EDA_Telecom_Churn.ipynb` (Interactive analysis with visualizations)

---
*Note: This setup was tested and working on Windows with Python 3.13.2*
