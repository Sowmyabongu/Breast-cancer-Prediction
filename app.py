import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

# Paths
DATA_PATH = "data.csv"
MODEL_PATH = "tumor_model.sav"

# Load and preprocess data
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
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

# Streamlit app UI
st.title("Breast Cancer Tumor Classification App")
st.write("Enter tumor characteristics to predict if it's benign or malignant.")

# User inputs
user_input = []
st.subheader("Enter Tumor Features:")

for col in X.columns:
    col_min, col_max = float(X[col].min()), float(X[col].max())
    # Set default to 0.0 for empty input
    value = st.number_input(
        label=col,
        min_value=col_min,
        max_value=col_max,
        value=0.0,       # Start with 0
        step=0.01        # Optional: small increment
    )
    user_input.append(value)

# Prediction button
if st.button("Predict Tumor Type"):
    arr = np.array(user_input, dtype=float).reshape(1, -1)
    pred = tumor_model.predict(arr)
    result = "Malignant" if int(pred[0]) else "Benign"
    st.success(f"The predicted tumor type is: **{result}**")
