import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from utils import create_input_dataframe

# =========================================================
# Load Model & Preprocessor
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "final_xgboost_model.pkl")
preprocessor = joblib.load(BASE_DIR / "models" / "preprocessor.pkl")

# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a telecom customer is likely to churn."
)

# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("📊 Dashboard")

st.sidebar.markdown("---")

st.sidebar.subheader("Model")
st.sidebar.write("XGBoost")

st.sidebar.subheader("Performance")
st.sidebar.write("Accuracy : 96.38%")
st.sidebar.write("F1 Score : 0.929")
st.sidebar.write("ROC-AUC : 0.991")

st.sidebar.markdown("---")

st.sidebar.subheader("Developer")
st.sidebar.write("Your Name")

st.sidebar.markdown("---")

st.sidebar.info(
    "Predict customer churn using Machine Learning."
)

st.markdown("---")

# =========================================================
# Customer Information
# =========================================================

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.slider(
        "Age",
        18,
        80,
        35
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        1,
        72,
        12
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-Month",
            "One Year",
            "Two Year"
        ]
    )

with col2:

    internet_service = st.selectbox(
        "Internet Service",
        [
            "Yes",
            "No"
        ]
    )

    internet_type = st.selectbox(
        "Internet Type",
        [
            "Fiber Optic",
            "DSL",
            "Cable",
            "No Internet"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Credit Card",
            "Bank Withdrawal",
            "Mailed Check"
        ]
    )

    monthly_charge = st.slider(
        "Monthly Charge",
        18.0,
        120.0,
        70.0
    )

    satisfaction_score = st.slider(
        "Satisfaction Score",
        1,
        5,
        3
    )

# =========================================================
# Predict Button
# =========================================================

st.markdown("---")

predict_button = st.button("Predict Churn")

# =========================================================
# Prediction
# =========================================================

if predict_button:

    user_input = {

        "Gender": gender,
        "Age": age,
        "Under 30": "Yes" if age < 30 else "No",
        "Senior Citizen": "Yes" if age >= 65 else "No",
        "Married": married,
        "Dependents": dependents,
        "Number of Dependents": 1 if dependents == "Yes" else 0,

        "City": "Los Angeles",
        "Zip Code": 90001,
        "Latitude": 34.05,
        "Longitude": -118.24,
        "Population": 100000,

        "Referred a Friend": "No",
        "Number of Referrals": 0,

        "Tenure in Months": tenure,

        "Offer": None,

        "Phone Service": "Yes",

        "Avg Monthly Long Distance Charges": 20,

        "Multiple Lines": "No",

        "Internet Service": internet_service,

        "Internet Type": (
            None if internet_service == "No"
            else internet_type
        ),

        "Avg Monthly GB Download": 20,

        "Online Security": online_security,

        "Online Backup": "No",

        "Device Protection Plan": "No",

        "Premium Tech Support": "No",

        "Streaming TV": "No",

        "Streaming Movies": "No",

        "Streaming Music": "No",

        "Unlimited Data": "No",

        "Contract": contract,

        "Paperless Billing": "Yes",

        "Payment Method": payment_method,

        "Monthly Charge": monthly_charge,

        "Total Charges": monthly_charge * tenure,

        "Total Refunds": 0,

        "Total Extra Data Charges": 0,

        "Total Long Distance Charges": 20 * tenure,

        "Total Revenue": monthly_charge * tenure,

        "Satisfaction Score": satisfaction_score,

        "CLTV": 4500

    }

    # -----------------------------------------------------
    # Preprocess & Predict
    # -----------------------------------------------------

    input_df = create_input_dataframe(user_input)

    processed = preprocessor.transform(input_df)

    prediction = model.predict(processed)[0]

    probability = model.predict_proba(processed)[0][1]

    confidence = max(probability, 1 - probability)

    # -----------------------------------------------------
    # Prediction Result
    # -----------------------------------------------------

    st.markdown("---")

    st.header("Prediction Result")

    st.write(
        "The model predicts whether the customer is likely to churn based on the information provided."
    )

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

    with col2:
        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

    with col3:

        if probability < 0.30:
            risk = "🟢 Low"

        elif probability < 0.70:
            risk = "🟡 Medium"

        else:
            risk = "🔴 High"

        st.metric(
            "Risk Level",
            risk
        )

    # -----------------------------------------------------
    # Progress Bar
    # -----------------------------------------------------

    st.markdown("### Churn Probability")

    st.progress(float(probability))

    # -----------------------------------------------------
    # Customer Summary
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("Customer Summary")

    summary = pd.DataFrame({

        "Feature": [

            "Gender",
            "Age",
            "Contract",
            "Tenure",
            "Internet Type",
            "Monthly Charge",
            "Satisfaction Score"

        ],

        "Value": [

            gender,
            age,
            contract,
            tenure,
            internet_type,
            monthly_charge,
            satisfaction_score

        ]

    })

    st.table(summary)

    # -----------------------------------------------------
    # Recommendation
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("Recommendation")

    if prediction == 1:

        st.warning(
            """
Recommended Actions

• Offer loyalty discounts

• Contact customer proactively

• Improve customer support

• Recommend a long-term contract
"""
        )

    else:

        st.success(
            """
Customer appears loyal.

Continue providing quality service.

Offer premium plans to increase revenue.
"""
        )

    # -----------------------------------------------------
    # Model Information
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("Model Information")

    st.write("""
**Model:** XGBoost

**Accuracy:** 96.38%

**F1 Score:** 0.929

**ROC-AUC:** 0.991

**Explainability:** SHAP
""")

    st.markdown("---")

    st.caption(
        "Customer Churn Prediction System | Built using Python, Scikit-learn, XGBoost, SHAP and Streamlit"
    )