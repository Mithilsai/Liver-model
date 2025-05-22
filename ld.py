import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the model and scaler
scaler = joblib.load('scaler.pkl')
xgb_model = joblib.load('xgb_model.pkl')

# Define the landing page
def landing_page():
    st.title("Welcome to the Liver Disease Risk Prediction Tool")
    st.image("https://img.freepik.com/free-vector/health-professional-team_52683-36023.jpg", use_column_width=True)  # Replace with your image URL
    st.write("""
    ## Take control of your liver health
    This tool allows you to input your health metrics and assess your risk of developing liver disease.
    """)
    if st.button("Get Started"):
        st.session_state.page = "main_app"

# Define the main application page
def main_app():
    st.title("Liver Disease Risk Prediction Tool")

    st.write("""
    ### Enter your health metrics below to assess your risk of developing liver disease.
    """)

    # Get user input
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])

    # Option to calculate BMI
    bmi_choice = st.selectbox("Do you know your BMI?", ["Yes, I know my BMI", "No, help me calculate my BMI"])
    if bmi_choice == "Yes, I know my BMI":
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=25.0, help="Body Mass Index. Normal range: 18.5-24.9. Calculated as weight (kg) / (height (m))^2.")
    else:
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        bmi = calculate_bmi(weight, height)
        st.write(f"Your calculated BMI is: {bmi:.2f} kg/m²")

    alcohol_consumption = st.slider("Alcohol Consumption (units per week)", min_value=0, max_value=100, value=0, help="Estimate units per week (1 unit ≈ 10ml or 8g of pure alcohol).")
    smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
    genetic_risk = st.selectbox("Is there a family history of liver disease?", ["No", "Yes"])
    physical_activity = st.slider("Physical Activity (hours per week)", min_value=0, max_value=50, value=1, help="Moderate to vigorous physical activity.")
    diabetes = st.selectbox("Do you have diabetes?", ["No", "Yes"])
    hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
    liver_function_test = st.number_input("Most recent liver function test result (ALT/AST level)", min_value=0.0, max_value=500.0, value=20.0, help="Enter your latest ALT or AST level (typically in U/L). If unsure, use the higher value or consult your doctor.")

    # Convert categorical inputs to numerical values
    gender = 1 if gender == "Male" else 0
    smoking = 1 if smoking == "Yes" else 0
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

    # Additional information
    st.write("""
    ### What do these terms mean?
    - **BMI**: Body Mass Index, a measure of body fat based on height and weight. Normal range is typically 18.5-24.9. Note: BMI is a general indicator and may be less accurate for individuals with high muscle mass.
    - **Alcohol Consumption**: Self-reported average weekly consumption. Standard units vary by region (e.g., a UK unit is ~8g or 10ml of pure alcohol).
    - **Smoking**: Whether the user currently smokes tobacco products.
    - **Genetic Risk**: Indicates if there is a known family history of liver disease (e.g., parents, siblings).
    - **Physical Activity**: Hours of moderate to vigorous physical activity per week.
    - **Diabetes**: A condition characterized by high blood sugar levels.
    - **Hypertension**: Also known as high blood pressure.
    - **Liver Function Test**: Refers to common liver enzyme tests like Alanine Aminotransferase (ALT) or Aspartate Aminotransferase (AST). Units are typically in U/L. Higher values can indicate liver stress or damage.
    """)

# Main logic to switch between pages
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

if st.session_state.page == 'landing':
    landing_page()
else:
    main_app()
