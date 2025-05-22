# Liver Disease Prediction Tool

## Overview

This project provides a web-based tool to predict an individual's risk of developing liver disease. Users can input various health metrics, and the tool utilizes a machine learning model (XGBoost) to provide a risk assessment (Low, Moderate, High) along with a risk score.

## Features

*   **Interactive Web Interface**: Built with Streamlit for easy user input and clear presentation of results.
*   **Machine Learning Model**: Uses an XGBoost classifier trained on synthetic liver disease data.
*   **Risk Assessment**: Provides a qualitative risk level (Low, Moderate, High) and a quantitative risk score.
*   **BMI Calculation**: Includes an option to calculate BMI if the user does not know it.
*   **Explanatory Information**: Offers descriptions of the input terms and tips based on the predicted risk level.
*   **Comprehensive Testing**: Includes unit tests for utility functions and model pipeline components.
*   **Reproducible Model Training**: A Jupyter notebook (`Model.ipynb`) details the complete model training and evaluation process.

## Setup and Installation

### Prerequisites

*   Python 3.8 or newer.
*   `pip` (Python package installer).

### Steps

1.  **Clone the repository (if applicable)**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment**:
    *   On macOS and Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```

3.  **Install dependencies**:
    Ensure your virtual environment is activated, then run:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Once the setup is complete and dependencies are installed:

1.  Ensure your virtual environment is activated.
2.  Navigate to the project's root directory (where `ld.py` is located).
3.  Run the Streamlit application:
    ```bash
    streamlit run ld.py
    ```
    This will typically open the application in your default web browser.

## Retraining the Model

The machine learning model used by this application can be retrained or fine-tuned using the `Model.ipynb` Jupyter notebook.

Key steps in the notebook include:
*   **Data Loading**: The model is trained using `syn_liver_disease_data.csv` (this file would need to be present).
*   **Preprocessing**: Includes steps like feature scaling using `StandardScaler`.
*   **Hyperparameter Tuning**: `GridSearchCV` is used to find the optimal hyperparameters for the XGBoost model.
*   **Model Saving**: The trained `XGBClassifier` is saved as `xgb_model.pkl`, and the `StandardScaler` is saved as `scaler.pkl`. These files are then used by `ld.py`.

To run the notebook, you can use Jupyter Lab or Jupyter Notebook:
```bash
# If you don't have jupyter lab installed:
# pip install jupyterlab
jupyter lab Model.ipynb
```

## Running Tests

The project includes unit tests to ensure the reliability of its components.

1.  Ensure your virtual environment is activated and all dependencies (including `pytest`) from `requirements.txt` are installed.
2.  Navigate to the project's root directory.
3.  Run the tests using pytest:
    ```bash
    pytest tests/
    ```
    This command will discover and run all tests located in the `tests/` directory.

---
*This README provides a comprehensive guide to setting up, running, and developing the Liver Disease Prediction Tool.*
