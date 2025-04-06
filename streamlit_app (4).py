
import streamlit as st
import pandas as pd
import pickle

st.title("🧠 Transformer Health Prediction")

# Load models
with open('SVC_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# User Input
fal = st.number_input("Enter 2-FAL value", min_value=0.0)
dp = st.number_input("Enter DP value", min_value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([[fal, dp]], columns=['2-FAL', 'DP'])
    prediction = model.predict(input_df)
    st.success(f"Predicted Health Condition: {prediction[0]}")
