import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the XGBoost model
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the Streamlit app
def main():
    st.title("Fraud Detection Predictor")
    st.write("Predict whether a transaction is fraudulent or legitimate.")

    # Input fields for the categorical and numerical features
    st.header("Input Transaction Details")
    transaction_type = st.selectbox("Transaction Type", ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'])
    amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f", value=0.0)
    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, format="%.2f", value=0.0)
    newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, format="%.2f", value=0.0)

    # Prepare input for the model
    if st.button("Predict"):
        # Map transaction type to numerical encoding
        type_mapping = {'CASH_OUT': 0, 'PAYMENT': 1, 'CASH_IN': 2, 'TRANSFER': 3, 'DEBIT': 4}
        type_encoded = type_mapping[transaction_type]

        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'type': [type_encoded],
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig]
        })

        # Align input with model's expected features
        try:
            feature_names = model.feature_names_in_
        except AttributeError:
            feature_names = model.get_booster().feature_names

        # Add missing features with default values and reorder columns
        for col in feature_names:
            if col not in input_data:
                input_data[col] = 0
        input_data = input_data[feature_names]

        # Ensure input data is in correct format (e.g., float32 for numerical features)
        input_data = input_data.astype('float32')

        # Make a prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1]

        # Display the results
        if prediction[0] == 1:
            st.error(f"The transaction is predicted to be **Fraudulent** with a probability of {prediction_proba[0]:.2f}.")
        else:
            st.success(f"The transaction is predicted to be **Legitimate** with a probability of {1 - prediction_proba[0]:.2f}.")

if __name__ == "__main__":
    main()


