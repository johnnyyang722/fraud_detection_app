import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

# Load preprocessor and model
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

def main():
    st.title("Fraud Detection Predictor")
    
    # Input fields
    transaction_type = st.selectbox("Transaction Type", ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'])
    amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f", value=0.0)
    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, format="%.2f", value=0.0)
    newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, format="%.2f", value=0.0)
    
    if st.button("Predict"):
        # Prepare input data
        input_data = pd.DataFrame({
            'type': [transaction_type],  # Keep the categorical value as is
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig]
        })

        input_data = input_data.astype({'amount': 'float64', 'oldbalanceOrg': 'float64', 'newbalanceOrig': 'float64'})
        input_data = input_data.fillna(0)

        # Transform the input data
        input_transformed = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(input_transformed)
        prediction_proba = model.predict_proba(input_transformed)[:, 1]

        # Display results
        if prediction[0] == 1:
            st.error(f"The transaction is predicted to be **Fraudulent** with a probability of {prediction_proba[0]:.2f}.")
        else:
            st.success(f"The transaction is predicted to be **Legitimate** with a probability of {1 - prediction_proba[0]:.2f}.")

if __name__ == "__main__":
    main()


