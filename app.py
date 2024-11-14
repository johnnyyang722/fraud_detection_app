import streamlit as st
import numpy as np
import joblib

# Load your trained Random Forest model
model = joblib.load("rf_model.pkl")  # Ensure the model filename is correct

# Title and Description
st.title("Fraud Detection App")
st.write("Enter transaction details to predict whether it's fraudulent or not.")

# Collect user input for transaction details
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
type_transaction = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

# Encode transaction type (assuming your model was trained with one-hot encoded types)
type_encoded = [1 if type_transaction == t else 0 for t in ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]]

# Create input array (only 'amount' and encoded transaction types)
input_data = [amount] + type_encoded
input_data = np.array(input_data).reshape(1, -1)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0, 1]

    # Display results
    if prediction[0] == 1:
        st.error(f"Warning: Fraudulent transaction detected with probability {probability:.2%}")
    else:
        st.success(f"Transaction is legitimate with probability {(1 - probability):.2%}")
