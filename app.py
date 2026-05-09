import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('📡 Telecom Customer Churn Predictor')
st.write('Enter customer details to predict if they will churn.')

# Input fields
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider('Tenure in Months', 0, 80, 12)
    monthly_charge = st.number_input('Monthly Charge ($)', 0.0, 200.0, 65.0)
    total_charges = st.number_input('Total Charges ($)', 0.0, 10000.0, 500.0)
    contract = st.selectbox('Contract Type', ['Month-to-Month', 'One Year', 'Two Year'])
    internet_service = st.selectbox('Internet Service', ['Yes', 'No'])
    internet_type = st.selectbox('Internet Type', ['Fiber Optic', 'DSL', 'Cable', 'None'])

with col2:
    payment_method = st.selectbox('Payment Method', ['Bank Withdrawal', 'Credit Card', 'Mailed Check'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    age = st.slider('Age', 18, 80, 35)
    number_of_referrals = st.slider('Number of Referrals', 0, 10, 0)
    offer = st.selectbox('Offer', ['None', 'Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E'])
    married = st.selectbox('Married', ['Yes', 'No'])

# Encode inputs
contract_map = {'Month-to-Month': 0, 'One Year': 1, 'Two Year': 2}
internet_type_map = {'DSL': 0, 'Cable': 1, 'Fiber Optic': 2, 'None': 3}
payment_map = {'Bank Withdrawal': 0, 'Credit Card': 1, 'Mailed Check': 2}
yes_no = {'Yes': 1, 'No': 0}
offer_map = {'None': 0, 'Offer A': 1, 'Offer B': 2, 'Offer C': 3, 'Offer D': 4, 'Offer E': 5}

# Build input array (34 features — fill rest with 0)
features = np.zeros(34)
features[0] = 0  # Gender
features[1] = age
features[2] = 0  # Under 30
features[3] = 0  # Senior Citizen
features[4] = yes_no[married]
features[5] = 0  # Dependents
features[6] = 0  # Number of Dependents
features[7] = 0  # Referred a Friend
features[8] = number_of_referrals
features[9] = tenure
features[10] = offer_map[offer]
features[11] = yes_no[internet_service]
features[12] = 0  # Avg Monthly Long Distance
features[13] = 0  # Multiple Lines
features[14] = yes_no[internet_service]
features[15] = internet_type_map[internet_type]
features[16] = 0  # Avg Monthly GB
features[17] = 0  # Online Security
features[18] = 0  # Online Backup
features[19] = 0  # Device Protection
features[20] = 0  # Premium Tech Support
features[21] = 0  # Streaming TV
features[22] = 0  # Streaming Movies
features[23] = 0  # Streaming Music
features[24] = 0  # Unlimited Data
features[25] = contract_map[contract]
features[26] = yes_no[paperless_billing]
features[27] = payment_map[payment_method]
features[28] = monthly_charge
features[29] = total_charges
features[30] = 0  # Total Refunds
features[31] = 0  # Total Extra Data
features[32] = 0  # Total Long Distance
features[33] = total_charges  # Total Revenue

# Scale and predict
features_scaled = scaler.transform([features])
prediction = model.predict(features_scaled)[0]
probability = model.predict_proba(features_scaled)[0][1]

st.divider()

if st.button('Predict Churn'):
    if prediction == 1:
        st.error(f' This customer is likely to CHURN — {probability*100:.1f}% probability')
    else:
        st.success(f' This customer is likely to STAY — {(1-probability)*100:.1f}% probability')

    st.metric('Churn Probability', f'{probability*100:.1f}%')