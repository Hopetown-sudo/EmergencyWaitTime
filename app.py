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
openai_api_key = os.getenv("OPENAI_KEY")

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

In **3 sentences**, explain the likely reason for the wait in simple, kind, and empathetic language. 
Be brief and reassuring and do not use boilerplate language.
Do not say exactly the numbers of Facility Sizes, Nurse-to-Patient Ratio and Specialist Availability, but you can describe them.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a hospital assistant. Your job is to explain ER wait times in an empathetic and simple way."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content, prompt

# UI layout
st.title("⏱️ WaitWise: ER Wait Time Predictor + GenAI Explanation")
st.markdown("Use patient and facility details to estimate ER wait time and receive a short, empathetic explanation.")


# Randomization function
def randomize_inputs():
    return {
        "Region": np.random.choice(regions),
        "Day of Week": np.random.choice(days_of_week),
        "Season": np.random.choice(seasons),
        "Time of Day": np.random.choice(time_of_day),
        "Urgency Level": np.random.choice(urgency_levels),
        "Nurse-to-Patient Ratio": np.random.randint(1, 11),
        "Specialist Availability": np.random.randint(0, 11),
        "Facility Size (Beds)": np.random.randint(10, 151)
    }

# Initialize session state for random values
if "random_inputs" not in st.session_state:
    st.session_state.random_inputs = randomize_inputs()

# Sidebar inputs
st.sidebar.header("Patient Visit Information")

# Button to randomize inputs
if st.sidebar.button("🎲 Randomize Inputs"):
    st.session_state.random_inputs = randomize_inputs()


# Sidebar widgets using session state
region = st.sidebar.selectbox("Region", regions, index=regions.index(st.session_state.random_inputs["Region"]))
day_of_week = st.sidebar.selectbox("Day of Week", days_of_week, index=days_of_week.index(st.session_state.random_inputs["Day of Week"]))
season = st.sidebar.selectbox("Season", seasons, index=seasons.index(st.session_state.random_inputs["Season"]))
time_day = st.sidebar.selectbox("Time of Day", time_of_day, index=time_of_day.index(st.session_state.random_inputs["Time of Day"]))
urgency = st.sidebar.selectbox("Urgency Level (1=Lowest)", urgency_levels, index=urgency_levels.index(st.session_state.random_inputs["Urgency Level"]))
nurse_ratio = st.sidebar.slider("Nurse-to-Patient Ratio", 1, 10, st.session_state.random_inputs["Nurse-to-Patient Ratio"])
specialists = st.sidebar.slider("Specialist Availability", 0, 10, st.session_state.random_inputs["Specialist Availability"])
beds = st.sidebar.slider("Facility Size (Beds)", 10, 150, st.session_state.random_inputs["Facility Size (Beds)"])


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
        explanation_text, used_prompt = generate_genai_explanation(input_data.iloc[0], predicted_wait_time)

    st.markdown("### 🧠 Explanation")
    st.info(explanation_text)

    with st.expander("🔍 Show Prompt Used for Explanation"):
        st.success(used_prompt.strip())

