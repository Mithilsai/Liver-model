import pytest
import sys
import os

# Add the project root directory to sys.path to allow imports from ld.py
# Assumes ld.py is in the root directory and this test file is in tests/
# Correct path to project root: os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Simplified for execution in current environment:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ld import calculate_bmi

def test_calculate_bmi_valid():
    """Test BMI calculation with typical valid inputs."""
    weight = 70  # kg
    height = 170 # cm
    # Expected BMI: 70 / (1.70 * 1.70) = 70 / 2.89 = 24.221453287...
    expected_bmi = 24.221453287
    assert calculate_bmi(weight, height) == pytest.approx(expected_bmi)

def test_calculate_bmi_zero_height():
    """
    Test BMI calculation when height is zero.
    The calculate_bmi function in ld.py raises ZeroDivisionError if height is 0.
    """
    weight = 70 # kg
    height = 0  # cm
    with pytest.raises(ZeroDivisionError):
        calculate_bmi(weight, height)

def test_calculate_bmi_zero_weight():
    """Test BMI calculation when weight is zero and height is valid."""
    weight = 0   # kg
    height = 170 # cm
    expected_bmi = 0.0
    assert calculate_bmi(weight, height) == pytest.approx(expected_bmi)

# Notes for running:
# 1. Ensure pytest is installed: pip install pytest
# 2. From the project root directory, run: pytest
#
# Directory structure assumed:
# your-repo-root/
# |-- ld.py
# |-- tests/
# |   |-- test_utils.py
# |-- (other files)
#
# The calculate_bmi function in ld.py (after the earlier refactoring) is:
# def calculate_bmi(weight, height):
#     """
#     Calculates BMI (Body Mass Index).
#     Weight in kg, Height in cm.
#     Raises ZeroDivisionError if height is 0.
#     """
#     if height == 0:
#         raise ZeroDivisionError("Height cannot be zero.")
#     return weight / ((height / 100) ** 2)
# This matches the expectations for the tests above.
# The Streamlit number_input for height in ld.py has min_value=100.0,
# so ZeroDivisionError would not be triggered through the UI's typical path for that specific input,
# but testing the function directly ensures its standalone robustness.
