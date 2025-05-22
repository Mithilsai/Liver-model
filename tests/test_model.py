import pytest
import joblib
import pandas as pd
import numpy as np
import os

# Ensure the test runner can find the model and scaler files.
# Assuming 'scaler.pkl' and 'xgb_model.pkl' are in the root directory.
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.pkl')

# Test Artifact Loading
def test_load_scaler():
    """Test if scaler.pkl can be loaded."""
    try:
        scaler = joblib.load(SCALER_PATH)
        assert scaler is not None, "Scaler should not be None after loading."
    except FileNotFoundError:
        pytest.fail(f"Scaler file not found at {SCALER_PATH}. Ensure it's in the root directory.")
    except Exception as e:
        pytest.fail(f"Error loading scaler.pkl: {e}")

def test_load_xgb_model():
    """Test if xgb_model.pkl can be loaded."""
    try:
        model = joblib.load(MODEL_PATH)
        assert model is not None, "XGBoost model should not be None after loading."
    except FileNotFoundError:
        pytest.fail(f"Model file not found at {MODEL_PATH}. Ensure it's in the root directory.")
    except Exception as e:
        pytest.fail(f"Error loading xgb_model.pkl: {e}")

# Test Prediction Pipeline (Base Case)
def test_prediction_pipeline_base_case():
    """Test the full prediction pipeline with a typical base case."""
    # Load scaler and model first
    try:
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        pytest.fail(f"Failed to load model or scaler for pipeline test: {e}")

    sample_input = {
        'Age': [50],                # Example: 50 years
        'Gender': [1],              # Example: Male (1)
        'BMI': [25.0],              # Example: 25.0 kg/m^2
        'AlcoholConsumption': [10], # Example: 10 units/week
        'Smoking': [0],             # Example: No (0)
        'GeneticRisk': [0],         # Example: No (0)
        'PhysicalActivity': [3],    # Example: 3 hours/week
        'Diabetes': [0],            # Example: No (0)
        'Hypertension': [1],        # Example: Yes (1)
        'LiverFunctionTest': [40.0] # Example: ALT/AST level 40.0
    }
    input_df = pd.DataFrame(sample_input)

    # Ensure column order matches training data if scaler is sensitive to it
    # (StandardScaler by default is not sensitive to column order if fit on a DataFrame
    # and then transform is also called on a DataFrame with same column names)
    expected_columns = ['Age', 'Gender', 'BMI', 'AlcoholConsumption', 'Smoking', 
                        'GeneticRisk', 'PhysicalActivity', 'Diabetes', 'Hypertension', 'LiverFunctionTest']
    input_df = input_df[expected_columns]


    try:
        scaled_input = scaler.transform(input_df)
        prediction_proba = model.predict_proba(scaled_input)
        
        # Probability of the positive class (liver disease)
        risk_probability = prediction_proba[:, 1][0] 

        assert isinstance(risk_probability, float), "Prediction probability should be a float."
        assert 0.0 <= risk_probability <= 1.0, "Prediction probability must be between 0.0 and 1.0."
    except Exception as e:
        pytest.fail(f"Error during prediction pipeline (base case): {e}")


# Test Prediction Pipeline (Edge Cases)
@pytest.mark.parametrize("case_name, sample_input_dict", [
    ("min_values_case", {
        'Age': [18],        # Plausible minimum age for such a model
        'Gender': [0],      # Female
        'BMI': [15.0],      # Low BMI
        'AlcoholConsumption': [0],
        'Smoking': [0],
        'GeneticRisk': [0],
        'PhysicalActivity': [0], # Min physical activity
        'Diabetes': [0],
        'Hypertension': [0],
        'LiverFunctionTest': [10.0] # Low LFT result
    }),
    ("max_values_case", {
        'Age': [80],        # Plausible maximum age
        'Gender': [1],      # Male
        'BMI': [40.0],      # High BMI
        'AlcoholConsumption': [50], # High alcohol consumption
        'Smoking': [1],     # Yes
        'GeneticRisk': [1], # Yes
        'PhysicalActivity': [10], # High physical activity
        'Diabetes': [1],    # Yes
        'Hypertension': [1],# Yes
        'LiverFunctionTest': [80.0] # High LFT result
    })
])
def test_prediction_pipeline_edge_cases(case_name, sample_input_dict):
    """Test the prediction pipeline with edge case inputs."""
    try:
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        pytest.fail(f"Failed to load model or scaler for pipeline test ({case_name}): {e}")

    input_df = pd.DataFrame(sample_input_dict)
    
    expected_columns = ['Age', 'Gender', 'BMI', 'AlcoholConsumption', 'Smoking', 
                        'GeneticRisk', 'PhysicalActivity', 'Diabetes', 'Hypertension', 'LiverFunctionTest']
    input_df = input_df[expected_columns]

    try:
        scaled_input = scaler.transform(input_df)
        prediction_proba = model.predict_proba(scaled_input)
        risk_probability = prediction_proba[:, 1][0]

        assert isinstance(risk_probability, float), f"Prediction probability should be a float for {case_name}."
        assert 0.0 <= risk_probability <= 1.0, f"Prediction probability must be between 0.0 and 1.0 for {case_name}."
    except Exception as e:
        pytest.fail(f"Error during prediction pipeline ({case_name}): {e}")

# To run these tests:
# 1. Ensure pytest is installed: pip install pytest joblib pandas numpy scikit-learn xgboost
# 2. Navigate to the root directory of your project.
# 3. Run the command: pytest
#
# Ensure scaler.pkl and xgb_model.pkl are in the root directory.
# The test file structure should be:
# your-repo-root/
# |-- ld.py
# |-- scaler.pkl
# |-- xgb_model.pkl
# |-- tests/
# |   |-- test_utils.py
# |   |-- test_model.py
# |-- (other files)
#
# Note on column order for scaler:
# If the scaler was fit on a NumPy array, the order of columns in input_df for transform matters.
# If it was fit on a Pandas DataFrame, StandardScaler is generally robust to column order IF
# the DataFrame passed to transform() has the same column names.
# The provided Model.ipynb uses a DataFrame for fitting, so this should be fine.
# Explicitly reordering columns in the test (input_df = input_df[expected_columns])
# is a good safeguard.
#
# The feature names in `expected_columns` must exactly match those used when the
# scaler was fit and the model was trained in Model.ipynb.
# Based on Model.ipynb, the columns are:
# X = data.drop(columns=['Diagnosis'])
# X.columns are: 'Age', 'Gender', 'BMI', 'AlcoholConsumption', 'Smoking', 'GeneticRisk', 
#                'PhysicalActivity', 'Diabetes', 'Hypertension', 'LiverFunctionTest'
# This order is used in the test.
#
# The base case and edge case values are illustrative.
# They should ideally reflect plausible boundaries or typical values seen in the dataset
# or expected in the application.
# The `min_value` and `max_value` from Streamlit's `number_input` and `slider`
# in `ld.py` can also guide these edge case values.
# For example:
# Age: min_value=0, max_value=120 (app allows 0, but plausible for model might be 18)
# BMI: min_value=10.0, max_value=50.0
# AlcoholConsumption: min_value=0, max_value=100
# PhysicalActivity: min_value=0, max_value=168
# LiverFunctionTest: min_value=0.0, max_value=100.0
# The edge cases defined use these as a rough guide.
#
# The Gender, Smoking, GeneticRisk, Diabetes, Hypertension are binary (0 or 1).
# Gender: Male=1, Female=0 in ld.py
# Smoking: Yes=1, No=0
# GeneticRisk: Yes=1, No=0
# Diabetes: Yes=1, No=0
# Hypertension: Yes=1, No=0
# The edge cases reflect these binary assignments.
#
# Final check on model output: model.predict_proba(scaled_input) returns a 2D array like [[P(class_0), P(class_1)]].
# We need P(class_1), which is prediction_proba[:, 1]. Since we pass one sample, it's [0].
#
# Added more detailed comments and error messages for pytest.fail.
# Ensured paths to model files are constructed robustly.
# Added explicit reordering of DataFrame columns to match training order as a safeguard.
# Parameterized edge case tests for cleaner code.
# Added check for FileNotFoundError during artifact loading.
