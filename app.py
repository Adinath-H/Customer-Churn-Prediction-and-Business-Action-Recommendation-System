import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Retail Churn Intelligence System",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.main-title {
    font-size: 40px;
    font-weight: bold;
    color: #1f77b4;
}
.sub-text {
    font-size: 18px;
    color: gray;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Header Section
# -----------------------------
st.markdown('<p class="main-title">📊 Retail Customer Churn Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">AI Powered Prediction with Business Action Recommendations</p>', unsafe_allow_html=True)

st.divider()

# -----------------------------
# Sidebar - Input Panel
# -----------------------------
st.sidebar.header("📥 Customer Input Panel")

from datetime import date

today = date.today()

# -----------------------------
# 1️⃣ First & Last Purchase Date
# -----------------------------
first_purchase_date = st.sidebar.date_input(
    "📅 First Purchase Date",
    value=date(2023, 1, 1)
)

last_purchase_date = st.sidebar.date_input(
    "📅 Last Purchase Date",
    value=today
)

# Validate dates
if last_purchase_date < first_purchase_date:
    st.sidebar.error("Last purchase date cannot be before first purchase date")
    tenure_days = 0
    recency = 0
else:
    tenure_days = (last_purchase_date - first_purchase_date).days
    recency = (today - last_purchase_date).days

tenure_years = tenure_days / 365 if tenure_days > 0 else 0

# -----------------------------
# 2️⃣ Frequency & Monetary
# -----------------------------
orders_input = st.sidebar.text_input(
    "🛒 Enter Order Amounts (comma separated)",
    "200,800,450,200"
)

try:
    orders = [float(x.strip()) for x in orders_input.split(",") if x.strip() != ""]
except:
    st.sidebar.error("Please enter valid numbers separated by commas")
    orders = []

frequency = len(orders)
monetary = sum(orders)

if frequency > 0:
    aov = monetary / frequency
else:
    aov = 0

# -----------------------------
# 3️⃣ CLV (Data-driven lifespan)
# -----------------------------
life_span = tenure_years   # Instead of fixed 2
clv = aov * frequency * life_span

predict_button = st.sidebar.button("🚀 Predict Now")


# -----------------------------
# Dashboard Layout
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("📅 Recency", recency)
    st.metric("🔁 Frequency", frequency)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("💰 Monetary", f"₹{round(monetary, 2)}")
    st.metric("🛒 AOV", f"₹{round(aov, 2)}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("💎 CLV", f"₹{round(clv, 2)}")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()


# -----------------------------
# Prediction Section
# -----------------------------
if predict_button:

    input_df = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary],
        'AOV': [aov],
        'CLV': [clv]
    })

    input_scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=input_df.columns
    )

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    # Risk Indicator
    if probability > 0.7:
        st.error("🔴 High Risk – Customer Likely to Churn")
    elif probability > 0.4:
        st.warning("🟡 Medium Risk – Needs Attention")
    else:
        st.success("🟢 Low Risk – Loyal Customer")

    # Probability Display
    st.markdown("### 📈 Churn Probability")
    st.progress(float(probability))
    st.write(f"### {round(probability*100,2)}% Probability of Churn")

    # -----------------------------
    # Intelligent Business Action
    # -----------------------------
    st.divider()
    st.subheader("📌 Intelligent Business Recommendation")

    # CLV Segmentation
    if clv > 1000:
        value_segment = "High Value Customer"
    elif clv > 500:
        value_segment = "Medium Value Customer"
    else:
        value_segment = "Low Value Customer"

    st.write(f"### 💎 Customer Segment: {value_segment}")

    # Combined Probability + CLV Logic
    if probability > 0.7:

        if clv > 1000:
            st.error("🔴 Critical Customer – Immediate Action Required")
            st.markdown("""
            ### 🚨 Action Plan:
            - Offer 25–30% retention discount
            - Assign dedicated relationship manager
            - Personal phone call follow-up
            - Exclusive VIP benefits
            """)
        else:
            st.warning("🔴 High Churn Risk")
            st.markdown("""
            ### ⚠️ Action Plan:
            - Send automated 15% discount email
            - Limited-time offer campaign
            - Reminder notifications
            """)

    elif probability > 0.4:

        if clv > 1000:
            st.warning("🟡 Important Customer – Preventive Action")
            st.markdown("""
            ### 📢 Action Plan:
            - Offer loyalty reward points
            - Early access to new products
            - Personalized email engagement
            """)
        else:
            st.info("🟡 Moderate Risk Customer")
            st.markdown("""
            ### 📩 Action Plan:
            - Send promotional emails
            - Encourage repeat purchase
            - Small coupon incentive
            """)

    else:

        if clv > 1000:
            st.success("🟢 Loyal VIP Customer")
            st.markdown("""
            ### 🎉 Action Plan:
            - Offer premium membership
            - Exclusive product previews
            - Reward appreciation program
            """)
        else:
            st.success("🟢 Stable Customer")
            st.markdown("""
            ### 👍 Action Plan:
            - Regular marketing campaigns
            - Cross-sell recommendations
            - Customer satisfaction survey
            """)

    st.divider()

    # -----------------------------
    # Visualization
    # -----------------------------
    st.subheader("📊 Customer Feature Visualization")

    chart_df = pd.DataFrame({
        "Features": ["Recency", "Frequency", "Monetary", "AOV", "CLV"],
        "Values": [recency, frequency, monetary, aov, clv]
    })

    st.bar_chart(chart_df.set_index("Features"))

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---
👨‍💻 Developed for Retail Customer Behavior Prediction Project  
📊 Machine Learning Model: Logistic Regression  
🚀 Intelligent Business Decision Engine Integrated
""")
