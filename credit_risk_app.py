import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('credit_card_risk_model.joblib')

# Create a Streamlit app
st.title('Credit Card Risk Prediction App')

st.write('Please enter the following information to predict the Risk Level:')

# Input fields
annual_income = st.number_input('Annual Income', min_value=0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850)
debt_to_income = st.number_input('Debt-to-Income Ratio', min_value=0.1, max_value=0.6, step=0.01)
employment_duration = st.number_input('Employment Duration (years)', min_value=0)

# Predict button
if st.button('Predict Risk Level'):
    # Create a DataFrame with the user input
    user_data = pd.DataFrame([[annual_income, credit_score, debt_to_income, employment_duration]],
                             columns=['Annual Income', 'Credit Score', 'Debt-to-Income Ratio', 'Employment Duration'])

    # Make a prediction using the model
    prediction = model.predict(user_data)[0]

    risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
    result = f'The predicted risk level is: {risk_levels[prediction]}'
    st.write(result)
