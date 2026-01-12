import streamlit as st
import pandas as pd
import pickle
import os
import sys

# Ensure src can be found if needed for unpickling
sys.path.append('../src')

paths = {
    'pipeline': './models/models.pkl',
    'model': './models/lin_reg.pkl',
    'data': './data/bengaluru_house_prices.csv'
}

# Verify files exist
for name, path in paths.items():
    if not os.path.exists(path):
        st.error(f"Missing File: {path} (PWD: {os.getcwd()})")
        st.stop()

# Load Models
with open(paths['pipeline'], 'rb') as f:
    full_pipeline = pickle.load(f)
    preprocessor = full_pipeline.named_steps['preprocessor']

with open(paths['model'], 'rb') as f:
    lin_model = pickle.load(f)

# Load and Prep Data for Dropdowns
data = pd.read_csv(paths['data'])

data[['location', 'society']] = data[['location', 'society']].fillna('unknown')
data['availability'] = data['availability'].fillna('unknown')
data['area_type'] = data['area_type'].fillna('unknown')

for col in ['area_type', 'availability', 'location', 'society']:
    data[col] = data[col].astype(str).str.strip().str.lower()

st.title("House Price Prediction")

# Inputs
total_sqft = st.number_input("Total Square Feet", value=1000.0)
bath = st.number_input("Bathrooms", min_value=1, value=2, step=1)
bhk = st.number_input("BHK", min_value=1, value=2, step=1)
balcony = st.number_input("Balcony", min_value=0, value=1, step=1)

area_type = st.selectbox("Area Type", sorted(data['area_type'].unique()))
availability = st.selectbox("Availability", sorted(data['availability'].unique()))
location = st.selectbox("Location", sorted(data['location'].unique()))
society = st.selectbox("Society", sorted(data['society'].unique()))

if st.button("Predict"):
    input_data = pd.DataFrame({
        'total_sqft': [total_sqft],
        'bath': [bath],
        'bhk': [bhk],
        'balcony': [balcony],
        'area_type': [area_type],
        'availability': [availability],
        'location': [location],
        'society': [society],
        'size': [f"{int(bhk)} bhk"]
    })

    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col] = input_data[col].str.strip().str.lower()

    try:
        processed_data = preprocessor.transform(input_data)
        prediction = lin_model.predict(processed_data)
        st.header(f"Result: {prediction[0]:.2f} Lakhs")
    except Exception as e:
        st.error(f"Prediction Error: {e}")