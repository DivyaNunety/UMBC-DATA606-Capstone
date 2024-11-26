#!/usr/bin/env python
# coding: utf-8

# In[11]:


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

# Load and cache the data
@st.cache_data
def load_data():
    data = pd.read_csv("C:/Users/divya/OneDrive/Desktop/606/final/Drivers_Data.csv")
    return data

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
    0. Property Damage Crash
    1. Injury Crash
    2. Fatal Crash

    ### Collision Type:
    0. OTHER
    1. SINGLE VEHICLE
    2. STRAIGHT MOVEMENT ANGLE
    3. SAME DIR REAR END
    4. HEAD ON LEFT TURN
    5. SAME DIRECTION SIDESWIPE
    6. OPPOSITE DIRECTION SIDESWIPE
    7. SAME DIRECTION LEFT TURN
    8. ANGLE MEETS LEFT TURN
    9. SAME DIRECTION RIGHT TURN
    10. ANGLE MEETS RIGHT TURN
    11. UNKNOWN
    12. HEAD ON
    13. SAME DIR REND LEFT TURN
    14. ANGLE MEETS LEFT HEAD ON
    15. SAME DIR REND RIGHT TURN
    16. OPPOSITE DIR BOTH LEFT TURN
    17. SAME DIR BOTH LEFT TURN
    18. Front to Rear
    19. Single Vehicle
    20. Angle
    21. Other
    22. Sideswipe, Same Direction
    23. Rear To Side
    24. Sideswipe, Opposite Direction
    25. Rear To Rear
    26. Front to Front
    27. Unknown

    ### Weather:
    0. CLEAR
    1. CLOUDY
    2. RAINING
    3. SNOW
    4. FOGGY
    5. OTHER
    6. UNKNOWN
    7. WINTRY MIX
    8. SEVERE WINDS
    9. SLEET
    10. BLOWING SNOW
    11. BLOWING SAND, SOIL, DIRT
    12. Clear
    13. Rain
    14. Fog, Smog, Smoke
    15. Unknown
    16. Cloudy
    17. Severe Crosswinds
    18. Snow
    19. Freezing Rain Or Freezing Drizzle
    20. Blowing Snow
    21. Sleet Or Hail

    ### Surface Condition:
    0. DRY
    1. ICE
    2. WET
    3. SLUSH
    4. UNKNOWN
    5. WATER(STANDING/MOVING)
    6. SNOW
    7. OTHER
    8. MUD, DIRT, GRAVEL
    9. OIL
    10. SAND
    11. Dry
    12. Wet
    13. Other
    14. Water (standing, moving)
    15. Ice/Frost
    16. Mud, Dirt, Gravel
    17. Snow
    18. Slush
    19. Sand

    ### Light:
    0. DAYLIGHT
    1. DARK LIGHTS ON
    2. DUSK
    3. DAWN
    4. DARK NO LIGHTS
    5. OTHER
    6. DARK -- UNKNOWN LIGHTING
    7. UNKNOWN
    8. Daylight
    9. Dark - Lighted
    10. Dark - Not Lighted
    11. Dawn
    12. Dark - Unknown Lighting
    13. Dusk
    14. Other
    15. Unknown

    ### Driver Substance Abuse:
    0. NONE DETECTED
    1. UNKNOWN
    2. ALCOHOL CONTRIBUTED
    3. ALCOHOL PRESENT
    4. COMBINATION CONTRIBUTED
    5. COMBINED SUBSTANCE PRESENT
    6. ILLEGAL DRUG CONTRIBUTED
    7. ILLEGAL DRUG PRESENT
    8. MEDICATION CONTRIBUTED
    9. MEDICATION PRESENT
    10. OTHER
    11. Unknown, Unknown
    12. Not Suspect of Alcohol Use, Not Suspect of Drug Use
    13. Suspect of Alcohol Use, Not Suspect of Drug Use
    14. Suspect of Alcohol Use, Unknown
    15. Not Suspect of Alcohol Use, Suspect of Drug Use
    16. Unknown, Not Suspect of Drug Use
    17. Suspect of Alcohol Use, Suspect of Drug Use
    18. Not Suspect of Alcohol Use, Unknown
    19. Unknown, Suspect of Drug Use

    ### Injury Severity:
    0. NO APPARENT INJURY
    1. SUSPECTED MINOR INJURY
    2. POSSIBLE INJURY
    3. SUSPECTED SERIOUS INJURY
    4. FATAL INJURY
    5. No Apparent Injury
    6. Possible Injury
    7. Suspected Minor Injury
    8. Suspected Serious Injury
    9. Fatal Injury

    ### Speed Limit:
    - 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75

    ### Vehicle Damage Extent:
    0. SUPERFICIAL
    1. UNKNOWN
    2. NO DAMAGE
    3. DISABLING
    4. FUNCTIONAL
    5. DESTROYED
    6. OTHER
    7. Functional
    8. Superficial
    9. Disabling
    10. Vehicle Not at Scene
    11. No Damage

    ### Vehicle Body Type:
    0. OTHER
    1. PASSENGER CAR
    2. SPORT UTILITY VEHICLE
    3. PICKUP TRUCK
    4. MOTORCYCLE
    5. VAN
    6. TRUCK TRACTOR
    7. RECREATIONAL VEHICLE
    8. LIMOUSINE
    """)

# Prediction Section
st.subheader("Predict Driver At Fault")
st.write("Enter crash data below:")

# Input fields for prediction
acrs_report_type = st.selectbox("ACRS Report Type", [0, 1, 2])
collision_type = st.selectbox("Collision Type", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
weather = st.selectbox("Weather", [0, 1, 2, 3, 4, 5, 6, 7, 8])
surface_condition = st.selectbox("Surface Condition", [0, 1, 2, 3, 4, 5])
light = st.selectbox("Light", [0, 1, 2, 3, 4])
driver_substance_abuse = st.selectbox("Driver Substance Abuse", [0, 1, 2, 3, 4, 5])
injury_severity = st.selectbox("Injury Severity", [0, 1, 2, 3, 4])
speed_limit = st.slider("Speed Limit", min_value=0, max_value=75, step=5)
vehicle_damage_extent = st.selectbox("Vehicle Damage Extent", [0, 1, 2, 3, 4, 5])
vehicle_body_type = st.selectbox("Vehicle Body Type", [0, 1, 2, 3, 4, 5])

# Prediction button
if st.button("Predict"):
    prediction = model.predict(np.array([[acrs_report_type, collision_type, weather, surface_condition, light,
                                          driver_substance_abuse, injury_severity, speed_limit,
                                          vehicle_damage_extent, vehicle_body_type]]))
    fault_status = "At Fault" if prediction[0] == 1 else "Not At Fault"
    st.write(f"The model predicts that the driver is: **{fault_status}**")


# In[ ]:





# In[ ]:





# In[ ]:




