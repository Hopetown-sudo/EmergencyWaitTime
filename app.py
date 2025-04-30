import streamlit as st
import pandas as pd
import numpy as np
import joblib
from openai import OpenAI
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Sample values for dropdowns
urgency_levels = ['1', '2', '3']
time_of_day = ['Morning', 'Afternoon', 'Evening', 'Late Night']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
regions = ['Urban', 'Rural']

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("waitwise_model_pipeline_tuned.joblib")

# GenAI explanation using OpenAI
client = OpenAI(api_key=openai_api_key)

# Generate explanation with OpenAI
def generate_genai_explanation(inputs, predicted_time):
    prompt = f"""
A patient arrived at the ER with the following conditions:
- Urgency Level: {inputs['Urgency Level']}
- Time of Day: {inputs['Time of Day']}
- Nurse-to-Patient Ratio: {inputs['Nurse-to-Patient Ratio']}
- Specialist Availability: {inputs['Specialist Availability']}
- Region: {inputs['Region']}
- Facility Size: {inputs['Facility Size (Beds)']}
The predicted wait time is {int(predicted_time)} minutes.

In **3 sentences**, explain the likely reason for the wait in simple, kind, and empathetic language. Be brief and reassuring.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a hospital assistant. Your job is to explain ER wait times in an empathetic and simple way."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# UI layout
st.title("⏱️ WaitWise: ER Wait Time Predictor + GenAI Explanation")
st.markdown("Use patient and facility details to estimate ER wait time and receive a short, empathetic explanation.")

# Sidebar inputs
st.sidebar.header("Patient Visit Information")
region = st.sidebar.selectbox("Region", regions)
day_of_week = st.sidebar.selectbox("Day of Week", days_of_week)
season = st.sidebar.selectbox("Season", seasons)
time_day = st.sidebar.selectbox("Time of Day", time_of_day)
urgency = st.sidebar.selectbox("Urgency Level", urgency_levels)
nurse_ratio = st.sidebar.slider("Nurse-to-Patient Ratio", 1, 10, 4)
specialists = st.sidebar.slider("Specialist Availability", 0, 10, 3)
beds = st.sidebar.slider("Facility Size (Beds)", 10, 150, 75)

# Prepare input
input_data = pd.DataFrame([{
    "Region": region,
    "Day of Week": day_of_week,
    "Season": season,
    "Time of Day": time_day,
    "Urgency Level": urgency,
    "Nurse-to-Patient Ratio": nurse_ratio,
    "Specialist Availability": specialists,
    "Facility Size (Beds)": beds
}])

# Predict button
if st.button("🔍 Predict Wait Time"):
    model = load_model()
    predicted_wait_time = model.predict(input_data)[0]

    st.success(f"🕒 Predicted Wait Time: **{int(predicted_wait_time)} minutes**")

    with st.spinner("Generating explanation..."):
        explanation_text = generate_genai_explanation(input_data.iloc[0], predicted_wait_time)

    st.markdown("### 🧠 Explanation")
    st.info(explanation_text)