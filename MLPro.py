import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("m1.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.title("🩺 Breast Cancer Detection App")

st.write("Enter the values for the following features:")

inputs = []
for feat in features:
    val = st.number_input(feat, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    data = np.array(inputs).reshape(1, -1)
    scaled_data = scaler.transform(data)

    pred = model.predict(scaled_data)

    if pred[0] == 1:
        st.error("⚠️ Possibility of Cancer")
    else:
        st.success("✅ No Possibility of Cancer")

