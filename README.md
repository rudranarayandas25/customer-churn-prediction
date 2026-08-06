# Customer Churn Prediction

A complete data science project for predicting customer churn in a telecom company. Built as a capstone project for a Data Science certification.

## Project Overview

- **Business Problem:** Identify customers likely to churn and enable proactive retention
- **Dataset:** 5,000 telecom customers with demographic, account, and service features
- **Models:** Logistic Regression, Random Forest, XGBoost
- **Key Techniques:** EDA, feature engineering, SMOTE, model comparison, feature importance
- **Deliverables:** Jupyter notebook, trained models, interactive dashboard, project report

## Project Structure

```
customer-churn-prediction/
  data/
    telco_churn.csv              Dataset
  notebooks/
    churn_analysis.ipynb         Full EDA + modeling notebook
  src/
    generate_data.py             Synthetic data generation
    preprocessing.py             Data loading, cleaning, encoding, scaling
    modeling.py                  Model training, evaluation, comparison
  app/
    streamlit_app.py             Interactive prediction dashboard
  models/                        Saved trained models and scaler
  reports/
    project_report.md            Comprehensive project report
    *.png                        Generated visualizations
  requirements.txt               Python dependencies
  run_pipeline.py                One-click pipeline runner
  README.md                      This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python run_pipeline.py
```

This will:
- Generate the synthetic dataset
- Preprocess and encode the data
- Train all three models
- Evaluate and compare performance
- Save the best model

### 3. Explore the Analysis

```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

### 4. Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard includes:
- Dataset overview and statistics
- Interactive EDA visualizations
- Single-customer churn prediction
- Business insights and recommendations

## Key Findings

- Month-to-month contracts show 2-3x higher churn than annual plans
- Electronic check payment users have elevated churn
- Customers with value-added services (security, tech support) churn less
- Tenure is inversely correlated with churn risk
- The best ML model achieves strong predictive performance

## Technologies Used

| Category | Tools |
|---|---|
| Data Manipulation | pandas, numpy |
| Machine Learning | scikit-learn, XGBoost, imbalanced-learn |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Environment | Jupyter Notebook |

## License

This project is for educational/certification purposes.
