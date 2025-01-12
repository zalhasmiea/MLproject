import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load the saved ensemble model
model_path = 'ensemble_model.pkl'  # Update with the correct path
ensemble_model = joblib.load(model_path)

# Add a title and a banner
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: #4CAF50;
        font-size: 36px;
    }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<h1 class="title">CHD Risk Prediction App</h1>', unsafe_allow_html=True)

# Add a banner image
st.image("banner_img.jpg", use_container_width=True)   # Ensure this path is correct

st.write("""
### Predict the Risk of Coronary Heart Disease (CHD) within 10 Years
Fill in the patient information below and click **Predict Risk**.
""")

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=50, step=1, help="Enter the patient's age in years.")
    sex = st.selectbox("Sex", options=["Male", "Female"])
    education = st.selectbox("Education Level", options=[1, 2, 3, 4], help="1: High School, 2: Diploma, 3: Bachelor, 4: Master/PhD")
    current_smoker = st.selectbox("Current Smoker", options=["Yes", "No"])
    cigs_per_day = st.number_input("Cigarettes per Day", min_value=0, max_value=100, value=10, step=1)

with col2:
    tot_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=500, value=200, step=1)
    bp_meds = st.selectbox("On BP Medications", options=["Yes", "No"])
    prevalent_stroke = st.selectbox("History of Stroke", options=["Yes", "No"])
    prevalent_hyp = st.selectbox("Hypertension", options=["Yes", "No"])
    diabetes = st.selectbox("Diabetes", options=["Yes", "No"])
    sys_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=300, value=120, step=1)

bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=70, step=1)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=100, step=1)

# Map inputs
sex_map = {"Male": 1, "Female": 0}
binary_map = {"Yes": 1, "No": 0}

# Prepare input data
user_data = pd.DataFrame({
    "sex": [sex_map[sex]],
    "age": [age],
    "education": [education],
    "currentSmoker": [binary_map[current_smoker]],
    "cigsPerDay": [cigs_per_day],
    "BPMeds": [binary_map[bp_meds]],
    "prevalentStroke": [binary_map[prevalent_stroke]],
    "prevalentHyp": [binary_map[prevalent_hyp]],
    "diabetes": [binary_map[diabetes]],
    "totChol": [tot_chol],
    "sysBP": [sys_bp],
    "diaBP": [bmi],
    "BMI": [bmi],
    "heartRate": [heart_rate],
    "glucose": [glucose]
})

# Prediction button
if st.button("Predict Risk"):
    prediction = ensemble_model.predict(user_data)
    risk = "High Risk" if prediction[0] == 1 else "Low Risk"
    if risk == "High Risk":
        st.error(f"Predicted Risk: **{risk}**")
    else:
        st.success(f"Predicted Risk: **{risk}**")

    # Risk probability as a gauge chart
    probabilities = ensemble_model.predict_proba(user_data)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilities[0][1] * 100,
        title={"text": "Risk Probability (%)"},
        gauge={'axis': {'range': [0, 100]}},
    ))
    st.plotly_chart(fig)
