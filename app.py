import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(page_title="Car Sales Prediction", layout="wide")

# Load models and preprocessors
@st.cache_resource
def load_artifacts():
    with open("saved_models/artifacts.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()
model2 = artifacts["model2"]
reason_model = artifacts["reason_model"]
le = artifacts["le"]
ohe = artifacts["ohe"]
ohe_cols = artifacts["ohe_cols"]
X_train_balanced = artifacts["X_train_balanced"]
reason_model_features = artifacts.get("reason_model_features", reason_model.feature_names_in_)

# App header
st.title("Car Sales Prediction")
st.markdown("Enter customer details below to predict if they will purchase a car.")

# Main prediction page
st.header("Customer Prediction")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    age = st.number_input("Age", min_value=18, max_value=70, value=42)
    gender = st.selectbox("Gender", ["Male", "Female"])
    income = st.number_input("Annual Income", min_value=200000, max_value=9999999, value=800000)
    
with col2:
    st.subheader("Vehicle Information")
    city = st.selectbox("City", ["Pune", "Delhi", "Hyderabad", "Mumbai", "Bangalore"])
    brand = st.selectbox("Preferred Brand", ["Toyota", "Kia", "Hyundai", "Honda", "Tata", "Maruti"])
    car_model = st.selectbox("Car Model", ["Sedan", "SUV", "Hatchback"])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "EV"])

col3, col4 = st.columns(2)

with col3:
    st.subheader("Behavioral Data")
    test_drive_taken = st.checkbox("Test Drive Taken", value=False)
    competitor_visit = st.checkbox("Visited Competitor", value=False)
    mileage = st.slider("Expected Mileage (km/liter)", min_value=10, max_value=30, value=15)
    
with col4:
    st.subheader("Commercial Details")
    budget = st.number_input("Budget", min_value=400000, max_value=1500000, value=800000)
    car_price = st.number_input("Car Price", min_value=500000, max_value=1600000, value=750000)
    discount_offered = st.number_input("Discount Offered", min_value=5000, max_value=150000, value=8000)
    followup_efficiency = st.slider("Follow-up Efficiency", min_value=0.0, max_value=5.0, value=3.0)
    brand_loyalty_score = st.slider("Brand Loyalty Score", min_value=1, max_value=10, value=6)

# Prepare input data and show predictions
try:
    # Create input dictionary
    user_dict = {
        "age": age,
        "gender": 1 if gender == "Male" else 0,
        "income": income,
        "mileage": mileage,
        "test_drive_taken": int(test_drive_taken),
        "discount_offered": discount_offered,
        "competitor_visit": int(competitor_visit),
        "budget_gap": budget - car_price,
        "followup_efficiency": followup_efficiency,
        "brand_loyalty_score": brand_loyalty_score,
    }
    
    # Create one-hot encoded features
    for col in ["city", "brand", "car_model", "fuel_type"]:
        for cat in ohe.categories_[ohe_cols.index(col)]:
            user_dict[f"{col}_{cat}"] = 0
    
    # Set the selected categories
    user_dict[f"city_{city}"] = 1
    user_dict[f"brand_{brand}"] = 1
    user_dict[f"car_model_{car_model}"] = 1
    user_dict[f"fuel_type_{fuel_type}"] = 1
    
    # Create DataFrame and ensure column alignment
    user_df = pd.DataFrame([user_dict])
    user_df = user_df.reindex(columns=model2.feature_names_in_, fill_value=0)
    
    # Make prediction
    pred = model2.predict(user_df)[0]
    prob = model2.predict_proba(user_df)[0][1]
    
    # Display results
    st.markdown("---")
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if pred == 1:
            st.success("Will Buy")
        else:
            st.error("Will Not Buy")
    
    with col2:
        st.metric("Confidence Level", f"{prob*100:.1f}%")
    
    
    # If customer won't buy, predict the reason
    if pred == 0:
        st.markdown("---")
        st.subheader("Reason for Not Buying")
        
        # Prepare data for reason model - use only the features it was trained on
        reason_df = user_df[reason_model_features]
        reason_pred = reason_model.predict(reason_df)[0]
        reason_proba = reason_model.predict_proba(reason_df)[0]
        reason_text = le.inverse_transform([reason_pred])[0]
        reason_confidence = reason_proba[reason_pred]
        
        col1, col2 = st.columns(2)
        with col1:
            st.warning(f"**Predicted Reason:** {reason_text}")
        with col2:
            st.metric("Reason Confidence", f"{reason_confidence*100:.1f}%")
    
    # Summary box
    st.markdown("---")
    st.subheader("Summary")
    summary_data = {
        "Customer Info": f"{age} yr old {gender.lower()} from {city}",
        "Budget": f"{budget:,.0f} Inr",
        "Target Car": f"{brand} {car_model}",
        "Price": f"{car_price:,.0f} Inr",
        "Prediction": "YES - Will Buy" if pred == 1 else f"NO - Will Not Buy ({reason_text})",
        "Confidence": f"{prob*100:.1f}%"
    }
    
    for key, value in summary_data.items():
        st.write(f"**{key}:** {value}")

except Exception as e:
    st.error(f"Error making prediction: {str(e)}")