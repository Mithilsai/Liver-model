import streamlit as st
import numpy as np
import pandas as pd
import joblib
import boto3
import json
from datetime import datetime

# Load the model and scaler
scaler = joblib.load('scaler.pkl')
xgb_model = joblib.load('xgb_model.pkl')

# AWS S3 Configuration
s3 = boto3.client('s3')
bucket_name = 'liver-disease-detection-risk'

# Define the Streamlit app
st.title("Liver Disease Risk Prediction Tool")

st.write("""
### Welcome to the Liver Disease Risk Prediction Tool!
Enter your health metrics below to assess your risk of developing liver disease.
""")

# Get user input
age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])

# BMI calculation option
calculate_bmi = st.checkbox("Calculate BMI")

if calculate_bmi:
    height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight_kg = st.number_input("Weight (kg)", min_value=10, max_value=200, value=70)
    if height_cm > 0:
        bmi = weight_kg / ((height_cm / 100) ** 2)
        st.write(f"Calculated BMI: {bmi:.2f}")
else:
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

alcohol_consumption = st.slider("Alcohol Consumption (units per week)", min_value=0, max_value=100, value=0)
smoking = st.selectbox("Smoking", ["Non-smoker", "Smoker"])
genetic_risk = st.selectbox("Family history of liver disease", ["No", "Yes"])
physical_activity = st.slider("Physical Activity (hours per week)", min_value=0, max_value=168, value=1)
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
liver_function_test = st.number_input("Most recent liver function test result (0.0 - 100.0)", min_value=0.0, max_value=100.0, value=0.0)

# Convert categorical inputs to numerical values
gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Smoker" else 0
genetic_risk = 1 if genetic_risk == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
hypertension = 1 if hypertension == "Yes" else 0

# Prepare the input data with the correct feature names
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'BMI': [bmi],
    'AlcoholConsumption': [alcohol_consumption],
    'Smoking': [smoking],
    'GeneticRisk': [genetic_risk],
    'PhysicalActivity': [physical_activity],
    'Diabetes': [diabetes],
    'Hypertension': [hypertension],
    'LiverFunctionTest': [liver_function_test]
})

# Ensure feature names match the training data
input_data_scaled = scaler.transform(input_data)

# Function to convert numpy data types to native Python types
def convert_to_native_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

# Make prediction
if st.button("Predict"):
    prediction = xgb_model.predict_proba(input_data_scaled)[:, 1]
    risk = prediction[0]

    # Categorize risk level
    if risk < 0.2:
        risk_level = "Low"
    elif risk < 0.5:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # Display the result
    st.write(f"### Predicted Risk of Liver Disease: {risk_level}")
    st.write(f"Your predicted risk score is: {risk:.2f}")

    # Provide tips based on risk level
    if risk_level == "Low":
        st.write("Great! Your risk of liver disease is low. Keep maintaining a healthy lifestyle.")
    elif risk_level == "Moderate":
        st.write("Your risk of liver disease is moderate. Consider regular check-ups and adopting healthier habits.")
    else:
        st.write("Your risk of liver disease is high. It is recommended to consult with a healthcare professional and take preventive measures.")
    
    # Save the input data and prediction to a dictionary
    data_to_save = {
        'Age': age,
        'Gender': gender,
        'BMI': bmi,
        'AlcoholConsumption': alcohol_consumption,
        'Smoking': smoking,
        'GeneticRisk': genetic_risk,
        'PhysicalActivity': physical_activity,
        'Diabetes': diabetes,
        'Hypertension': hypertension,
        'LiverFunctionTest': liver_function_test,
        'PredictedRisk': risk_level,
        'RiskScore': risk,
        'Timestamp': datetime.utcnow().isoformat()
    }

    # Convert the dictionary to JSON serializable types
    data_to_save = {k: convert_to_native_types(v) for k, v in data_to_save.items()}

    # Convert the dictionary to a JSON string
    data_json = json.dumps(data_to_save)

    # Define the filename for the S3 object
    filename = f"user_data_{datetime.utcnow().isoformat()}.json"

    # Upload the JSON data to S3
    try:
        s3.put_object(Bucket=bucket_name, Key=filename, Body=data_json)
        st.write("Data successfully saved to S3.")
    except Exception as e:
        st.write(f"An error occurred while saving data to S3: {e}")

# Additional information
st.write("""
### What do these terms mean?
- **Age**: Your age in years.
- **Gender**: Your gender (Male or Female).
- **BMI**: Body Mass Index, a measure of body fat based on height and weight.
- **Alcohol Consumption**: Amount of alcohol consumed per week (units).
- **Smoking**: Whether you are a smoker or non-smoker.
- **Genetic Risk**: Family history of liver disease.
- **Physical Activity**: Hours of physical activity per week.
- **Diabetes**: Whether you have diabetes.
- **Hypertension**: Whether you have high blood pressure.
- **Liver Function Test**: A blood test to check how well the liver is working (value between 0.0 and 100.0).
""")
