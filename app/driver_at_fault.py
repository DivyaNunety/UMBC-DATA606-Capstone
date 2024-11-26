#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# App Title
st.title("🚗 Driver Fault Prediction with Logistic Regression")
st.markdown("### Predict if a driver is at fault based on car crash data.")

# File uploader widget
uploaded_file = st.file_uploader("Upload your Drivers_Data.csv file", type=["csv"])

@st.cache_data
def load_data(file):
    if file is not None:
        data = pd.read_csv(file)
        return data
    else:
        return None

# Load the dataset
if uploaded_file:
    car_crash_df = load_data(uploaded_file)
    st.write(f"Dataset contains *{car_crash_df.shape[0]} rows* and *{car_crash_df.shape[1]} columns*.")
    if st.checkbox("Show data preview"):
        st.write(car_crash_df.head())
else:
    st.warning("Please upload a dataset to proceed.")
    st.stop()

@st.cache_data
def preprocess_data(df):
    df = df.drop(['Report Number', 'Off-Road Description', 'Municipality', 'Related Non-Motorist', 
                  'Non-Motorist Substance Abuse', 'Circumstance'], axis=1, errors='ignore').dropna()
    
    # Encode categorical variables
    le = LabelEncoder()
    variables = ['ACRS Report Type', 'Collision Type', 'Weather', 'Surface Condition', 'Light', 
                 'Driver Substance Abuse', 'Injury Severity', 'Vehicle Damage Extent', 'Vehicle Body Type']
    
    # Ensure only existing columns are encoded
    variables = [var for var in variables if var in df.columns]
    for var in variables:
        df[var] = le.fit_transform(df[var])
    df['Driver_At_Fault_encoded'] = le.fit_transform(df['Driver At Fault'])
    return df, le, variables

@st.cache_resource
def train_model(df):
    # Define features and target
    X = df[['ACRS Report Type', 'Collision Type', 'Weather', 'Surface Condition', 'Light', 'Driver Substance Abuse',
            'Injury Severity', 'Speed Limit', 'Vehicle Damage Extent', 'Vehicle Body Type']]
    y = df['Driver_At_Fault_encoded']
    
    # Train-test split and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Preprocess and train
dropped_df, le, variables = preprocess_data(car_crash_df)
model, log_reg_accuracy = train_model(dropped_df)

# Display Model Accuracy
st.write(f"*Logistic Regression Model Accuracy:* {log_reg_accuracy * 100 :.2f}%")

# Prediction Section
st.subheader("Predict Driver At Fault")
st.write("Enter crash data below:")

# Input fields for prediction
acrs_report_type = st.selectbox("ACRS Report Type", range(len(car_crash_df['ACRS Report Type'].unique())))
collision_type = st.selectbox("Collision Type", range(len(car_crash_df['Collision Type'].unique())))
weather = st.selectbox("Weather", range(len(car_crash_df['Weather'].unique())))
surface_condition = st.selectbox("Surface Condition", range(len(car_crash_df['Surface Condition'].unique())))
light = st.selectbox("Light", range(len(car_crash_df['Light'].unique())))
driver_substance_abuse = st.selectbox("Driver Substance Abuse", range(len(car_crash_df['Driver Substance Abuse'].unique())))
injury_severity = st.selectbox("Injury Severity", range(len(car_crash_df['Injury Severity'].unique())))
speed_limit = st.slider("Speed Limit", min_value=int(car_crash_df['Speed Limit'].min()), 
                        max_value=int(car_crash_df['Speed Limit'].max()), step=5)
vehicle_damage_extent = st.selectbox("Vehicle Damage Extent", range(len(car_crash_df['Vehicle Damage Extent'].unique())))
vehicle_body_type = st.selectbox("Vehicle Body Type", range(len(car_crash_df['Vehicle Body Type'].unique())))

# Prediction button
if st.button("Predict"):
    prediction = model.predict(np.array([[acrs_report_type, collision_type, weather, surface_condition, light,
                                          driver_substance_abuse, injury_severity, speed_limit,
                                          vehicle_damage_extent, vehicle_body_type]]))
    fault_status = "At Fault" if prediction[0] == 1 else "Not At Fault"
    st.success(f"The model predicts that the driver is: *{fault_status}*")
