# Customer Churn Prediction - Project Report

## 1. Executive Summary

This project builds a machine learning pipeline to predict customer churn for a telecom company. Using customer demographic, account, and service data, we trained and compared three models (Logistic Regression, Random Forest, XGBoost) to identify at-risk customers before they leave.

**Key Result:** The XGBoost model achieved 92% accuracy and 0.97 ROC-AUC, enabling the business to identify at-risk customers with high precision and recall.

---

## 2. Business Problem

Customer acquisition costs are far higher than retention costs. A telecom company needs to:

- Identify customers likely to churn in the near future
- Understand the key drivers of churn
- Enable proactive retention campaigns
- Reduce revenue loss from customer attrition

---

## 3. Dataset Description

**Source:** Synthetic dataset modeled after real telecom customer data
**Size:** 5,000 customers, 20 features

| Category | Features |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account Info | tenure, Contract, PaperlessBilling, PaymentMethod |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Financial | MonthlyCharges, TotalCharges |
| Target | Churn (0 = No, 1 = Yes) |

---

## 4. Methodology

### 4.1 Exploratory Data Analysis (EDA)

Key findings from EDA:

- **Class imbalance:** Churn rate is approximately 19%, creating a moderately imbalanced classification problem
- **Contract type is a strong predictor:** Month-to-month contracts show 2-3x higher churn
- **Tenure inversely correlated with churn:** Newer customers churn more
- **Payment method matters:** Electronic check users have elevated churn
- **Value-added services reduce churn:** Online security and tech support users churn less
- **Fiber optic internet:** Higher churn rate compared to DSL

### 4.2 Data Preprocessing

1. **Feature encoding:** Binary categorical features mapped to 0/1; multi-class features label-encoded
2. **Scaling:** Numeric features (tenure, MonthlyCharges, TotalCharges) standardized using StandardScaler
3. **Train-test split:** 80/20 stratified split to preserve class distribution
4. **Class imbalance handling:** SMOTE (Synthetic Minority Oversampling Technique) applied to training data

### 4.3 Models Trained

| Model | Type | Key Parameters |
|---|---|---|
| Logistic Regression | Linear baseline | max_iter=1000, class_weight=balanced |
| Random Forest | Ensemble (bagging) | n_estimators=200, max_depth=12, class_weight=balanced |
| XGBoost | Gradient boosting | n_estimators=200, max_depth=6, learning_rate=0.05 |

### 4.4 Evaluation Metrics

Given the business context, we prioritize:

1. **Recall:** Minimize false negatives (customers we miss who actually churn)
2. **F1-Score:** Balanced metric between precision and recall
3. **ROC-AUC:** Overall discriminatory power

---

## 5. Results

### 5.1 Model Performance

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| Accuracy | 0.8570 | 0.9050 | 0.9200 |
| Precision | 0.5830 | 0.7209 | 0.7434 |
| Recall | 0.8684 | 0.8158 | 0.8842 |
| F1-Score | 0.6977 | 0.7654 | 0.8077 |
| ROC-AUC | 0.9300 | 0.9553 | 0.9700 |

### 5.2 Best Model: XGBoost

XGBoost achieved the highest performance across all metrics. With an ROC-AUC of 0.97 and F1-Score of 0.81, it provides excellent discriminatory power for identifying at-risk customers. Notably, it achieves 88% recall, meaning it catches nearly 9 out of 10 churners while maintaining 74% precision.

### 5.3 Feature Importance Analysis

The top features driving churn predictions (in order):

1. Contract type (Month-to-month vs annual)
2. Tenure (monthly length of service)
3. Monthly charges
4. Internet service type
5. Payment method

---

## 6. Business Recommendations

### 6.1 Short-Term Actions

1. **Target Month-to-Month Customers:** Email/phone campaign offering 10-15% discount for switching to annual contracts
2. **Onboarding Program:** Implement 30-60-90 day check-in cadence for new customers (tenure < 12 months)
3. **Payment Incentives:** $3/month discount for switching from electronic check to auto-pay

### 6.2 Medium-Term Actions

4. **Service Bundling:** Create attractive bundles combining internet with online security and tech support
5. **Fiber Quality Investigation:** Audit fiber optic service quality and address infrastructure gaps
6. **Predictive Scoring:** Integrate the ML model into CRM for real-time churn risk scoring

### 6.3 ROI Estimation

| Metric | Value |
|---|---|
| Average Customer Lifetime Value | $1,500/year |
| Monthly Churners (at current rate) | ~95 customers |
| Model Recall (predicted correctly) | ~88% = 84 customers |
| Intervention Success Rate (estimated) | 25% |
| Customers Saved per Month | ~21 |
| Annual Revenue Saved | ~$378,000 |

---

## 7. Technical Implementation

### 7.1 Project Structure

```
project/
  data/           Dataset (CSV)
  notebooks/      Jupyter notebook with full analysis
  src/            Python modules (preprocessing, modeling)
  app/            Streamlit interactive dashboard
  models/         Saved trained models
  reports/        Generated visualizations
```

### 7.2 Running the Project

```bash
# Generate data
python src/generate_data.py

# Train models
python src/modeling.py

# Run dashboard
streamlit run app/streamlit_app.py
```

### 7.3 Dependencies

- pandas, numpy — Data manipulation
- scikit-learn — ML algorithms and preprocessing
- xgboost — Gradient boosting
- imbalanced-learn — SMOTE for class imbalance
- matplotlib, seaborn, plotly — Visualizations
- streamlit — Interactive dashboard

---

## 8. Future Improvements

1. **Hyperparameter tuning:** Use GridSearchCV or Optuna for systematic optimization
2. **Feature engineering:** Create interaction features (e.g., tenure * monthly charges)
3. **More models:** Add LightGBM, CatBoost, or neural networks
4. **Model explainability:** Integrate SHAP/LIME for individual prediction explanations
5. **Production deployment:** Create REST API endpoint for real-time scoring
6. **A/B testing:** Measure actual retention impact of model-driven interventions

---

## 9. Conclusion

This project successfully demonstrates the end-to-end data science workflow:

- Data exploration and visualization to understand churn patterns
- Preprocessing and feature engineering for ML readiness
- Multi-model training with proper evaluation methodology
- Actionable business insights derived from analysis
- Interactive dashboard for stakeholder presentation

The predictive model enables the business to move from reactive to proactive customer retention, potentially saving hundreds of thousands in annual revenue.
