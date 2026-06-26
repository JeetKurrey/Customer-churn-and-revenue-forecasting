import streamlit as st
import requests

st.set_page_config(page_title="TeleConnect Systems", layout="wide")

st.title("TeleConnect Customer Analytics System")
st.write("Customer Churn Prediction & Revenue Forecasting")
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("Customer Information")

tenure          = st.sidebar.slider("Tenure (months)", min_value=0, max_value=72, value=24)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=20.0, max_value=150.0, value=70.0)
total_charges   = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1500.0)
contract        = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_svc    = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with st.sidebar.expander("Additional Details (optional)"):
    gender           = st.selectbox("Gender",            ["Male", "Female"])
    senior_citizen   = st.selectbox("Senior Citizen",    ["No", "Yes"])
    partner          = st.selectbox("Partner",           ["No", "Yes"])
    dependents       = st.selectbox("Dependents",        ["No", "Yes"])
    phone_service    = st.selectbox("Phone Service",     ["Yes", "No"])
    multiple_lines   = st.selectbox("Multiple Lines",    ["No", "Yes", "No phone service"])
    online_security  = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
    online_backup    = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
    device_prot      = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support     = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])
    paperless        = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method   = st.selectbox("Payment Method",    [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

# ── Payload ─────────────────────────────────────────────────────────────────
payload = {
    # Numerical Features
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,

    # Engineered Features
    "AvgMonthlySpend": monthly_charges / (tenure + 1),
    "ContractValue": monthly_charges * tenure,

    # Service Count
    "ServiceCount": sum([
        phone_service == "Yes",
        online_security == "Yes",
        online_backup == "Yes",
        device_prot == "Yes",
        tech_support == "Yes",
        streaming_tv == "Yes",
        streaming_movies == "Yes"
    ]),

    # One-Hot Encoded Features
    "gender_Male": int(gender == "Male"),

    "Partner_Yes": int(partner == "Yes"),
    "Dependents_Yes": int(dependents == "Yes"),

    "PhoneService_Yes": int(phone_service == "Yes"),

    "MultipleLines_No phone service":
        int(multiple_lines == "No phone service"),

    "MultipleLines_Yes":
        int(multiple_lines == "Yes"),

    "InternetService_Fiber optic":
        int(internet_svc == "Fiber optic"),

    "InternetService_No":
        int(internet_svc == "No"),

    "OnlineSecurity_No internet service":
        int(online_security == "No internet service"),

    "OnlineSecurity_Yes":
        int(online_security == "Yes"),

    "OnlineBackup_No internet service":
        int(online_backup == "No internet service"),

    "OnlineBackup_Yes":
        int(online_backup == "Yes"),

    "DeviceProtection_No internet service":
        int(device_prot == "No internet service"),

    "DeviceProtection_Yes":
        int(device_prot == "Yes"),

    "TechSupport_No internet service":
        int(tech_support == "No internet service"),

    "TechSupport_Yes":
        int(tech_support == "Yes"),

    "StreamingTV_No internet service":
        int(streaming_tv == "No internet service"),

    "StreamingTV_Yes":
        int(streaming_tv == "Yes"),

    "StreamingMovies_No internet service":
        int(streaming_movies == "No internet service"),

    "StreamingMovies_Yes":
        int(streaming_movies == "Yes"),

    "Contract_One year":
        int(contract == "One year"),

    "Contract_Two year":
        int(contract == "Two year"),

    "PaperlessBilling_Yes":
        int(paperless == "Yes"),

    "PaymentMethod_Credit card (automatic)":
        int(payment_method == "Credit card (automatic)"),

    "PaymentMethod_Electronic check":
        int(payment_method == "Electronic check"),

    "PaymentMethod_Mailed check":
        int(payment_method == "Mailed check")
}

API_URL = "http://127.0.0.1:5000/api/predict"

col1, col2 = st.columns(2)

# ── Churn Prediction ─────────────────────────────────────────────────────────
with col1:
    if st.button("Predict Churn", use_container_width=True, type="primary"):
        with st.spinner("Analysing customer churn risk…"):
            try:
                response = requests.post(API_URL, json=payload, timeout=15)
                res = response.json()   # always parse before branching

                if response.status_code == 200:
                    st.subheader("Prediction Outputs")
                    st.write(f"Raw Class Output: **{res['prediction']}**")
                    st.write(f"Churn Probability: **{res['churn_probability']:}**")
                    st.write(f"Risk Level: **{res['risk_level']}**")

                    st.markdown("### Recommendation")
                    risk = res["risk_level"]
                    #st.write(res)

                    if risk == "Critical":
                        st.error("🚨 Offer a yearly contract discount immediately.")

                    elif risk == "High":
                        st.warning("⚠️ Provide a personalised retention offer soon.")

                    elif risk == "Medium":
                        st.info("ℹ️ Monitor customer engagement and satisfaction.")

                    else:
                        st.success("✅ Customer likely to stay — low risk profile.")
                else:
                    st.error(f"Backend error: {res.get('message', response.text)}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach backend. Make sure Flask is running on port 5000.")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The backend may be overloaded.")

# ── Revenue Prediction ───────────────────────────────────────────────────────
with col2:
    if st.button("Predict Revenue", use_container_width=True):
        with st.spinner("Forecasting expected monthly revenue…"):
            try:
                response = requests.post(API_URL, json=payload, timeout=15)
                res = response.json()   # always parse before branching

                if response.status_code == 200:
                    st.subheader("Expected Revenue Output")
                    st.metric(
                        label="Forecasted Monthly Revenue",
                        value=f"${res['expected_revenue']:.2f}"
                    )
                    st.write(f"Churn Risk: **{res['risk_level']}** "
                             f"({res['churn_probability']:.2%})")
                else:
                    st.error(f"Backend error: {res.get('message', response.text)}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach backend. Make sure Flask is running on port 5000.")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The backend may be overloaded.")