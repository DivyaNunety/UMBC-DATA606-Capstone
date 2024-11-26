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
