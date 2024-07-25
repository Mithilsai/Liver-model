import streamlit as st
import numpy as np
import pandas as pd
import boto3
import joblib
import json

# Load scaler
scaler = joblib.load('scaler.pkl')

# Initialize the SageMaker runtime client
sm_runtime = boto3.client('sagemaker-runtime')

# SageMaker endpoint name
endpoint_name = 'your-sagemaker-endpoint'

def predict(data):
    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(data)
    )
    result = json.loads(response['Body'].read().decode())
    return result

# Streamlit app
st.title("Liver Disease Risk Prediction Tool")

st.write("""
### Welcome to the Liver Disease Risk Prediction Tool!
Enter your health metrics below to assess your risk of developing liver disease.
""")

# Function to calculate BMI
def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

# Get user input
age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])

bmi_choice = st.selectbox("Do you know your BMI?", ["Yes, I know my BMI", "No, help me calculate my BMI"])
if bmi_choice == "Yes, I know my BMI":
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=50.0, value=25.0)
else:
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
    bmi = calculate_bmi(weight, height)
    st.write(f"Your calculated BMI is: {bmi:.2f} kg/m²")

alcohol_consumption = st.slider("Alcohol Consumption (units per week)", min_value=0, max_value=100, value=0)
smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
genetic_risk = st.selectbox("Is there a family history of liver disease?", ["No", "Yes"])
physical_activity = st.slider("Physical Activity (hours per week)", min_value=0, max_value=168, value=1)
diabetes = st.selectbox("Do you have diabetes?", ["No", "Yes"])
hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
liver_function_test = st.number_input("Most recent liver function test result (ALT/AST level)", min_value=0.0, max_value=100.0, value=0.0)

# Convert categorical inputs to numerical values
gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
genetic_risk = 1 if genetic_risk == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
hypertension = 1 if hypertension == "Yes" else 0

# Prepare the input data with the correct feature names
input_data = {
    'Age': age,
    'Gender': gender,
    'BMI': bmi,
    'AlcoholConsumption': alcohol_consumption,
    'Smoking': smoking,
    'GeneticRisk': genetic_risk,
    'PhysicalActivity': physical_activity,
    'Diabetes': diabetes,
    'Hypertension': hypertension,
    'LiverFunctionTest': liver_function_test
}

# Ensure feature names match the training data
input_data_scaled = scaler.transform(pd.DataFrame([input_data]))

# Convert scaled data back to a list for JSON serialization
input_data_scaled = input_data_scaled.tolist()

# Make prediction
if st.button("Predict"):
    prediction = predict(input_data_scaled)
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

st.write("""
### What do these terms mean?
- **BMI**: Body Mass Index, a measure of body fat based on height and weight.
- **Alcohol Consumption**: Amount of alcohol consumed per week.
- **Smoking**: Whether the user smokes or not.
- **Genetic Risk**: Family history of liver disease.
- **Physical Activity**: Hours of physical activity per week.
- **Diabetes**: Whether the user has diabetes.
- **Hypertension**: Whether the user has high blood pressure.
- **Liver Function Test**: A blood test to check how well the liver is working.
""")
