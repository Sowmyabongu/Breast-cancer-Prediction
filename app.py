import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data.csv"
MODEL_PATH = "tumor_model.sav"

# Load dataset
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Features and target
y = df['diagnosis'].map({'B': 0, 'M': 1})
X = df.drop(columns=['diagnosis'])
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Load or train model
if os.path.exists(MODEL_PATH):
    tumor_model = joblib.load(MODEL_PATH)
    if tumor_model.n_features_in_ != X.shape[1]:
        tumor_model = RandomForestClassifier(n_estimators=100, random_state=42)
        tumor_model.fit(X, y)
        joblib.dump(tumor_model, MODEL_PATH)
else:
    tumor_model = RandomForestClassifier(n_estimators=100, random_state=42)
    tumor_model.fit(X, y)
    joblib.dump(tumor_model, MODEL_PATH)

# Streamlit app
st.title("Breast Cancer Tumor Classification App")
st.write("Enter tumor characteristics to predict if it's benign or malignant.")

# Collect user input
user_input = {}
for col in X.columns:
    col_min, col_max, col_mean = float(X[col].min()), float(X[col].max()), float(X[col].mean())
    value = st.number_input(col, min_value=col_min, max_value=col_max, value=col_mean)
    user_input[col] = value

# Prediction
if st.button("Predict Tumor Type"):
    input_df = pd.DataFrame([user_input])   # ✅ keeps column names
    pred = tumor_model.predict(input_df)
    result = "Malignant" if int(pred[0]) == 1 else "Benign"
    st.success(f"The predicted tumor type is: **{result}**")
