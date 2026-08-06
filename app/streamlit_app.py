import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import load_data, clean_data, encode_categorical

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

st.title("Customer Churn Prediction Dashboard")
st.markdown("Analyze customer data and predict churn probability using machine learning.")

df = load_data(os.path.join(os.path.dirname(__file__), "..", "data", "telco_churn.csv"))

tab1, tab2, tab3, tab4 = st.tabs(
    ["Data Overview", "EDA Visualizations", "Churn Predictor", "Business Insights"]
)

with tab1:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Churn Rate", f"{df['Churn'].mean():.1%}")
    with col3:
        st.metric("Avg. Monthly Charge", f"${df['MonthlyCharges'].mean():.2f}")
    with col4:
        st.metric("Avg. Tenure", f"{df['tenure'].mean():.0f} months")

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Column Information")
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.values,
        "Non-Null": df.count().values,
        "Unique Values": df.nunique().values,
    })
    st.dataframe(col_info, use_container_width=True)

with tab2:
    st.header("Exploratory Data Analysis")

    viz_option = st.selectbox(
        "Select Visualization",
        [
            "Churn Distribution",
            "Churn by Contract Type",
            "Churn by Internet Service",
            "Churn by Payment Method",
            "Tenure Distribution by Churn",
            "Monthly Charges vs Churn",
            "Churn by Demographics",
            "Correlation Heatmap",
        ],
    )

    if viz_option == "Churn Distribution":
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                df,
                names=df["Churn"].map({0: "No Churn", 1: "Churn"}),
                title="Churn Distribution",
                color_discrete_sequence=["#2ecc71", "#e74c3c"],
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            churn_counts = df["Churn"].value_counts()
            fig = px.bar(
                x=["No Churn", "Churn"],
                y=churn_counts.values,
                title="Churn Count",
                color=["No Churn", "Churn"],
                color_discrete_sequence=["#2ecc71", "#e74c3c"],
            )
            st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "Churn by Contract Type":
        churn_by_contract = (
            df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
        )
        fig = px.bar(
            x=churn_by_contract.index,
            y=churn_by_contract.values,
            title="Churn Rate by Contract Type",
            labels={"x": "Contract Type", "y": "Churn Rate (%)"},
            color=churn_by_contract.values,
            color_continuous_scale="RdYlGn_r",
            text=churn_by_contract.values,
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "Churn by Internet Service":
        churn_by_internet = (
            df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False) * 100
        )
        fig = px.bar(
            x=churn_by_internet.index,
            y=churn_by_internet.values,
            title="Churn Rate by Internet Service",
            labels={"x": "Internet Service", "y": "Churn Rate (%)"},
            color=churn_by_internet.values,
            color_continuous_scale="RdYlGn_r",
            text=churn_by_internet.values,
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "Churn by Payment Method":
        churn_by_payment = (
            df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False) * 100
        )
        fig = px.bar(
            x=churn_by_payment.index,
            y=churn_by_payment.values,
            title="Churn Rate by Payment Method",
            labels={"x": "Payment Method", "y": "Churn Rate (%)"},
            color=churn_by_payment.values,
            color_continuous_scale="RdYlGn_r",
            text=churn_by_payment.values,
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "Tenure Distribution by Churn":
        fig = px.histogram(
            df,
            x="tenure",
            color=df["Churn"].map({0: "No Churn", 1: "Churn"}),
            title="Tenure Distribution by Churn Status",
            labels={"tenure": "Tenure (Months)", "color": "Status"},
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
            opacity=0.7,
            barmode="overlay",
            nbins=40,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "Monthly Charges vs Churn":
        fig = px.box(
            df,
            x=df["Churn"].map({0: "No Churn", 1: "Churn"}),
            y="MonthlyCharges",
            title="Monthly Charges Distribution by Churn",
            labels={"x": "Churn Status", "y": "Monthly Charges ($)"},
            color=df["Churn"].map({0: "No Churn", 1: "Churn"}),
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "Churn by Demographics":
        col1, col2 = st.columns(2)
        with col1:
            churn_by_senior = (
                df.groupby("SeniorCitizen")["Churn"].mean() * 100
            )
            churn_by_senior.index = ["Not Senior", "Senior"]
            fig = px.bar(
                x=churn_by_senior.index,
                y=churn_by_senior.values,
                title="Churn by Senior Citizen Status",
                labels={"x": "", "y": "Churn Rate (%)"},
                color=churn_by_senior.values,
                color_continuous_scale="RdYlGn_r",
                text=churn_by_senior.values,
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            churn_by_partner = (
                df.groupby("Partner")["Churn"].mean() * 100
            )
            fig = px.bar(
                x=churn_by_partner.index,
                y=churn_by_partner.values,
                title="Churn by Partner Status",
                labels={"x": "", "y": "Churn Rate (%)"},
                color=churn_by_partner.values,
                color_continuous_scale="RdYlGn_r",
                text=churn_by_partner.values,
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "Correlation Heatmap":
        from sklearn.preprocessing import LabelEncoder

        df_encoded = df.drop("customerID", axis=1).copy()
        le = LabelEncoder()
        for col in df_encoded.select_dtypes(include="object").columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        corr = df_encoded.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Predict Churn for a Single Customer")

    model_path = os.path.join(os.path.dirname(__file__), "..", "models")

    model_files = [f for f in os.listdir(model_path) if f.endswith(".joblib") and f != "scaler.joblib"]

    if not model_files:
        st.warning("No trained model found. Please run the training pipeline first.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Customer Details")

            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
            total = st.number_input("Total Charges ($)", 0.0, 9000.0, tenure * monthly, 0.5)

        with col2:
            st.subheader("Prediction Result")

            input_data = pd.DataFrame(
                [
                    {
                        "gender": gender,
                        "SeniorCitizen": 1 if senior == "Yes" else 0,
                        "Partner": partner,
                        "Dependents": dependents,
                        "tenure": tenure,
                        "PhoneService": phone,
                        "MultipleLines": multiple_lines,
                        "InternetService": internet,
                        "OnlineSecurity": online_security,
                        "OnlineBackup": online_backup,
                        "DeviceProtection": device_protection,
                        "TechSupport": tech_support,
                        "StreamingTV": streaming_tv,
                        "StreamingMovies": streaming_movies,
                        "Contract": contract,
                        "PaperlessBilling": paperless,
                        "PaymentMethod": payment,
                        "MonthlyCharges": monthly,
                        "TotalCharges": total,
                    }
                ]
            )

            input_encoded = encode_categorical(input_data)

            scaler = joblib.load(os.path.join(model_path, "scaler.joblib"))
            numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
            input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

            for model_file in model_files:
                model = joblib.load(os.path.join(model_path, model_file))
                proba = model.predict_proba(input_encoded)[0]
                prediction = model.predict(input_encoded)[0]

                model_name = model_file.replace(".joblib", "")

                st.markdown(f"### Model: {model_name}")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    risk_color = "red" if proba[1] > 0.5 else "green"
                    st.metric("Churn Probability", f"{proba[1]:.1%}")
                with col_b:
                    st.metric(
                        "Prediction",
                        "CHURN" if prediction == 1 else "NO CHURN",
                    )
                with col_c:
                    risk = "High Risk" if proba[1] > 0.5 else "Low Risk"
                    st.metric("Risk Level", risk)

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=proba[1] * 100,
                        title={"text": "Churn Risk"},
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#e74c3c" if proba[1] > 0.5 else "#2ecc71"},
                            "steps": [
                                {"range": [0, 30], "color": "#2ecc71"},
                                {"range": [30, 60], "color": "#f1c40f"},
                                {"range": [60, 100], "color": "#e74c3c"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 3},
                                "thickness": 0.75,
                                "value": proba[1] * 100,
                            },
                        },
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Business Insights & Recommendations")

    st.subheader("Key Drivers of Churn")

    insights = {
        "Contract Type": {
            "finding": "Month-to-month customers churn at 2-3x the rate of annual contract holders.",
            "action": "Offer discounts for switching to annual contracts. Create a loyalty rewards program.",
        },
        "Tenure": {
            "finding": "Customers with less than 12 months of tenure are significantly more likely to churn.",
            "action": "Implement a 30-60-90 day onboarding program with proactive check-ins.",
        },
        "Payment Method": {
            "finding": "Electronic check users have the highest churn rate among all payment methods.",
            "action": "Incentivize switching to auto-pay with small monthly discounts ($2-5).",
        },
        "Internet Service": {
            "finding": "Fiber optic customers churn more than DSL or non-internet customers.",
            "action": "Investigate service quality. Bundle premium support with fiber plans.",
        },
        "Value-Added Services": {
            "finding": "Customers with online security and tech support churn significantly less.",
            "action": "Bundle these services at attractive introductory prices.",
        },
    }

    for title, insight in insights.items():
        with st.expander(f"{title}"):
            st.markdown(f"**Finding:** {insight['finding']}")
            st.markdown(f"**Recommended Action:** {insight['action']}")

    st.subheader("ROI Estimation")
    st.markdown(
        """
    Assuming the model can identify churners with 80%+ accuracy:
    - Average customer lifetime value: $1,500/year
    - Potential churners identified per month: ~60 (at current churn rate)
    - If we retain just 20% through intervention: 12 customers/month
    - **Annual savings: ~$216,000**
    """
    )
