import streamlit as st
import pickle
import pandas as pd

st.title("🍷 Wine Quality Prediction")

try:
    with open("model/rf_pipeline.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found.")
    st.stop()


with st.sidebar:
    fixed_acidity = st.number_input("Fixed Acidity:")
    volatile_acidity = st.number_input("Volatile Acidity:")
    citric_acid = st.number_input("Citric Acid:")
    residual_sugar = st.number_input("Residual Sugar:")
    chlorides = st.number_input("Chlorides:")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide:")
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide:")
    density = st.number_input("Density:")
    pH = st.number_input("pH:")
    sulphates = st.number_input("Sulphates:")
    alcohol = st.number_input("Alcohol:")

prediction_data = {"fixed acidity": fixed_acidity, "volatile acidity": volatile_acidity, "citric acid":citric_acid, "residual sugar":residual_sugar, "chlorides":chlorides, "free sulfur dioxide":free_sulfur_dioxide, "total sulfur dioxide":total_sulfur_dioxide,"density":density, "pH":pH, "sulphates": sulphates,"alcohol":alcohol}

pred_df = pd.DataFrame([prediction_data])

with st.expander("Input Features"):
    st.write("**Input Wine:**")
    st.dataframe(pred_df)

prediction = model.predict(pred_df)[0]

st.subheader("Predicted Quality")
st.success(str(prediction))