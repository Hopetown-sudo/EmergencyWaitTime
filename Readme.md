# WaitWise: Predicting and Explaining ER Wait Times with AI

WaitWise is an interactive Streamlit-based web app that predicts Emergency Room (ER) wait times based on patient visit characteristics and staffing levels. It uses a machine learning model to forecast wait time and generates human-friendly explanations using OpenAI's GPT-4.

---

## 🚀 Features

- Predict ER wait time using a trained Random Forest model
- Accept inputs such as urgency level, time of day, staffing levels, and region
- Generate empathetic, patient-friendly explanations using GPT-4
- Built for hospital administrators, emergency teams, and transparency for patients

---

## 📁 Project Structure

```
Emegergencywaittime/
├── app.py                 # Streamlit frontend app
├── waitwise_model_pipeline_tuned.joblib  # Trained ML pipeline
├── README.md
├── .env                            # Environment variable with OpenAI key
```

---

## 📦 Installation

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/Emegergencywaittime.git
cd Emegergencywaittime
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your OpenAI API key
Create a file `.env` with the following:
```env
OPENAI_API_KEY=your-openai-api-key
```

---

## ▶️ Run the App
```bash
streamlit run app.py
```

Make sure to load the `.env` values at the top of your script using `python-dotenv`:
```python
from dotenv import load_dotenv
load_dotenv()
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
```

---

## 🔍 Model Summary
- Model: Random Forest Regressor (sklearn)
- Features: Region, Day of Week, Season, Time of Day, Urgency Level (mapped to numeric), Nurse-to-Patient Ratio, Specialist Availability, Facility Size (Beds)
- Evaluation: R² ~ 0.94, RMSE ~ 16 minutes
- One-hot encoded categorical variables, numeric urgency retained

---

## 💡 Example GenAI Output
> "Due to multiple high-urgency cases and limited specialist availability in a rural facility during the evening, your expected wait time is approximately 122 minutes. We appreciate your patience."

---


