import streamlit as st
import pickle
import pandas as pd
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from feature_engineering import (
    engineer_features,
    InteractionTransformer
)

st.title("🍷 Wine Quality Prediction")

try:
    with open("model/rf_pipeline.pkl", "rb") as file:
        model = joblib.load(file)
except FileNotFoundError:
    st.error("Model file not found.")
    st.stop()


with st.sidebar:
    fixed_acidity = st.number_input("Fixed Acidity:", min_value=0)
    volatile_acidity = st.number_input("Volatile Acidity:", min_value = 0)
    citric_acid = st.number_input("Citric Acid:", min_value = 0)
    residual_sugar = st.number_input("Residual Sugar:", min_value = 0)
    chlorides = st.number_input("Chlorides:", min_value = 0)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide:", min_value=0)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide:",min_value = 0)
    density = st.number_input("Density:", min_value = 0)
    pH = st.number_input("pH:", min_value = 0)
    sulphates = st.number_input("Sulphates:", min_value = 0)
    alcohol = st.number_input("Alcohol:", min_value = 0)

prediction_data = {"fixed acidity": fixed_acidity, "volatile acidity": volatile_acidity, "citric acid":citric_acid, "residual sugar":residual_sugar, "chlorides":chlorides, "free sulfur dioxide":free_sulfur_dioxide, "total sulfur dioxide":total_sulfur_dioxide,"density":density, "pH":pH, "sulphates": sulphates,"alcohol":alcohol}

pred_df = pd.DataFrame([prediction_data])

with st.expander("Input Features"):
    st.write("**Input Wine:**")
    st.dataframe(pred_df)

if st.button("Predict:"):
    prediction = model.predict(pred_df)[0]
    prediction = (((prediction - 3)/5)*9)+1
    st.subheader("Predicted Quality")
    st.success(str(prediction))