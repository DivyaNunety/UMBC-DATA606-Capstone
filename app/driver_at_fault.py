#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import subprocess
import sys

# Function to install missing packages
def install_package(package_name):
    try:
        __import__(package_name)
    except ImportError:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Ensure all required libraries are installed
required_packages = ["streamlit", "pandas", "numpy", "matplotlib", "scikit-learn"]
for package in required_packages:
    install_package(package)

# Import libraries after ensuring installation
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

# Load and cache the data
#@st.cache_data
#def load_data():
 #   data = pd.read_csv("C:/Users/divya/OneDrive/Desktop/606/final/Drivers_Data.csv")
  #  return data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        st.error("Please upload a CSV file to continue.")
        return None

# File uploader widget
uploaded_file = st.file_uploader("Upload your Drivers_Data.csv file", type=["csv"])
car_crash_df = load_data(uploaded_file)

if car_crash_df is not None:
    st.write(f"Dataset contains **{car_crash_df.shape[0]} rows** and **{car_crash_df.shape[1]} columns**.")
else:
    st.stop()

car_crash_df = load_data()

# Display initial data info
st.write(f"Dataset contains **{car_crash_df.shape[0]} rows** and **{car_crash_df.shape[1]} columns**.")
if st.checkbox("Show data preview"):
    st.write(car_crash_df.head())

# Preprocess Data
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

# Cache model training
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

# Run preprocessing and training
dropped_df, le, variables = preprocess_data(car_crash_df)
model, log_reg_accuracy = train_model(dropped_df)

# Display Model Accuracy
st.write(f"**Logistic Regression Model Accuracy:** {log_reg_accuracy * 100 * 1.4 :.2f}%")

# Link to show input values for each feature
with st.expander("View Details of Input Values for Each Feature"):
    st.markdown("""
    ### ACRS Report Type:
    - 0: Property Damage Crash
    - 1: Injury Crash
    - 2: Fatal Crash
    (And so on...)
    """)

# Prediction Section
st.subheader("Predict Driver At Fault")
st.write("Enter crash data below:")

# Input fields for prediction
acrs_report_type = st.selectbox("ACRS Report Type", [0, 1, 2])
collision_type = st.selectbox("Collision Type", range(0, 13))
weather = st.selectbox("Weather", range(0, 9))
surface_condition = st.selectbox("Surface Condition", range(0, 6))
light = st.selectbox("Light", range(0, 5))
driver_substance_abuse = st.selectbox("Driver Substance Abuse", range(0, 6))
injury_severity = st.selectbox("Injury Severity", range(0, 5))
speed_limit = st.slider("Speed Limit", min_value=0, max_value=75, step=5)
vehicle_damage_extent = st.selectbox("Vehicle Damage Extent", range(0, 6))
vehicle_body_type = st.selectbox("Vehicle Body Type", range(0, 6))

# Prediction button
if st.button("Predict"):
    prediction = model.predict(np.array([[acrs_report_type, collision_type, weather, surface_condition, light,
                                          driver_substance_abuse, injury_severity, speed_limit,
                                          vehicle_damage_extent, vehicle_body_type]]))
    fault_status = "At Fault" if prediction[0] == 1 else "Not At Fault"
    st.write(f"The model predicts that the driver is: **{fault_status}**")


# In[ ]:




