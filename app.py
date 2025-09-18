import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -------------------------------
# Load model artifacts
# -------------------------------
model = joblib.load("logistic_model_final.pkl")
scaler = joblib.load("scaler_final.pkl")
explainer = joblib.load("shap_explainer_final.pkl")

# Custom decision threshold
THRESHOLD = 0.39

# Feature order
feature_columns = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'AvgMonthlySpend', 'gender_Male',
    'InternetService_Fiber optic', 'InternetService_No',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'tenure_group_13-24m', 'tenure_group_25-48m',
    'tenure_group_49-72m', 'tenure_group_7-12m'
]

# Business-friendly labels
feature_labels = {
    "gender_Male": "Gender: Male",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Has Partner",
    "Dependents": "Has Dependents",
    "tenure": "Tenure (months)",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "PaperlessBilling": "Paperless Billing",
    "MonthlyCharges": "Monthly Charges ($)",
    "TotalCharges": "Total Charges ($)",
    "AvgMonthlySpend": "Avg Monthly Spend",
    "InternetService_Fiber optic": "Fiber Optic Internet",
    "InternetService_No": "No Internet",
    "Contract_One year": "One Year Contract",
    "Contract_Two year": "Two Year Contract",
    "PaymentMethod_Credit card (automatic)": "Credit Card (Auto)",
    "PaymentMethod_Electronic check": "Electronic Check",
    "PaymentMethod_Mailed check": "Mailed Check",
    "tenure_group_7-12m": "Tenure: 7–12 months",
    "tenure_group_13-24m": "Tenure: 13–24 months",
    "tenure_group_25-48m": "Tenure: 25–48 months",
    "tenure_group_49-72m": "Tenure: 49–72 months",
}

# Retention recommendations
feature_actions = {
    "MonthlyCharges": "Offer a discount or repackage the plan to reduce monthly costs.",
    "TotalCharges": "Highlight customer’s long history with loyalty rewards.",
    "tenure": "Encourage renewal with contract benefits.",
    "tenure_group_7-12m": "Proactively engage — many customers leave in their first year.",
    "tenure_group_13-24m": "Reinforce value before the 2-year mark.",
    "tenure_group_25-48m": "Introduce loyalty perks to extend retention.",
    "tenure_group_49-72m": "Upsell advanced services to long-tenured customers.",
    "InternetService_Fiber optic": "Check satisfaction — fiber customers often leave if expectations aren’t met.",
    "InternetService_No": "Promote internet bundles to lock in services.",
    "Contract_One year": "Encourage upgrading to a 2-year contract.",
    "Contract_Two year": "Maintain engagement — 2-year contracts reduce churn risk.",
    "TechSupport": "Promote support bundles as a value-add.",
    "PaymentMethod_Electronic check": "Encourage switching to auto-payment methods.",
    "PaymentMethod_Credit card (automatic)": "Keep them engaged — auto-pay customers are more stable.",
    "PaymentMethod_Mailed check": "Promote easier digital payment options.",
    "SeniorCitizen": "Provide personalised assistance and simplify billing.",
    "Partner": "Offer family/partner bundles.",
    "Dependents": "Promote family plans and discounts."
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("T-Mobile Retention Analytics Platform")
st.header("🔍 Customer Churn Prediction")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.checkbox("Senior Citizen?")
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.5)
payment = st.selectbox("Payment Method", [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check"
])

# Feature engineering
tenure_groups = {
    "tenure_group_7-12m": 0,
    "tenure_group_13-24m": 0,
    "tenure_group_25-48m": 0,
    "tenure_group_49-72m": 0
}
if 7 <= tenure <= 12:
    tenure_groups["tenure_group_7-12m"] = 1
elif 13 <= tenure <= 24:
    tenure_groups["tenure_group_13-24m"] = 1
elif 25 <= tenure <= 48:
    tenure_groups["tenure_group_25-48m"] = 1
elif tenure >= 49:
    tenure_groups["tenure_group_49-72m"] = 1

total = monthly * tenure
avg_monthly_spend = total / tenure if tenure > 0 else monthly

data = {
    'SeniorCitizen': 1 if senior else 0,
    'Partner': 0, 'Dependents': 0,
    'tenure': tenure,
    'PhoneService': 1, 'MultipleLines': 0,
    'OnlineSecurity': 0, 'OnlineBackup': 0,
    'DeviceProtection': 0,
    'TechSupport': 1 if tech_support == "Yes" else 0,
    'StreamingTV': 0, 'StreamingMovies': 0,
    'PaperlessBilling': 1,
    'MonthlyCharges': monthly,
    'TotalCharges': total,
    'AvgMonthlySpend': avg_monthly_spend,
    'gender_Male': 1 if gender == "Male" else 0,
    'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
    'InternetService_No': 1 if internet == "No" else 0,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment == "Credit card (automatic)" else 0,
    'PaymentMethod_Electronic check': 1 if payment == "Electronic check" else 0,
    'PaymentMethod_Mailed check': 1 if payment == "Mailed check" else 0
}
data.update(tenure_groups)

X_input = pd.DataFrame([data], columns=feature_columns)

# -------------------------------
# Prediction & Explanation
# -------------------------------
if st.button("Predict"):
    X_scaled = scaler.transform(X_input)
    prob = model.predict_proba(X_scaled)[0][1]
    risk_pct = round(prob * 100, 1)
    prediction = 1 if prob >= THRESHOLD else 0

    # Risk Score
    st.subheader("📈 Churn Risk Score")
    fig, ax = plt.subplots(figsize=(4, 1))
    ax.barh(["Customer"], [risk_pct], color="crimson" if prediction else "seagreen")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Risk %")
    st.pyplot(fig, clear_figure=True)

    if prediction:
        st.error(f"⚠️ High Risk: This customer has a **{risk_pct}% chance of leaving**.")
    else:
        st.success(f"✅ Low Risk: This customer has only a **{risk_pct}% chance of leaving**.")

    # SHAP Explanation
    st.subheader("🔎 Why this prediction?")
    shap_values = explainer(X_scaled)

    shap_df = pd.DataFrame({
        "Feature": feature_columns,
        "SHAP Value": shap_values.values[0]
    })
    shap_df["Readable Feature"] = shap_df["Feature"].map(lambda x: feature_labels.get(x, x))

    # Top 5 drivers
    top5 = shap_df.reindex(shap_df["SHAP Value"].abs().sort_values(ascending=False).index).head(5)

    st.markdown("This chart shows the **top 5 factors** influencing churn risk:")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(
        top5["Readable Feature"],
        top5["SHAP Value"],
        color=["crimson" if v > 0 else "steelblue" for v in top5["SHAP Value"]]
    )
    ax.set_xlabel("Impact on Churn Risk")
    ax.set_title("Top 5 Drivers of Churn")
    st.pyplot(fig, clear_figure=True)

    # Recommended actions
    st.subheader("💡 Recommended Actions")
    for i, row in top5.iterrows():
        action = feature_actions.get(row["Feature"], "Monitor and engage this customer proactively.")
        st.write(f"- **{row['Readable Feature']}** → {action}")

    # Quick summary
    main_driver = top5.iloc[0]["Readable Feature"]
    main_action = feature_actions.get(top5.iloc[0]["Feature"], "Take proactive engagement.")
    st.info(f"📌 Quick Summary: This customer is at **{risk_pct}% churn risk**, mainly due to **{main_driver}**. Suggested action: {main_action}")
