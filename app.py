import streamlit as st
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "m1.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
features = pickle.load(open(os.path.join(BASE_DIR, "features.pkl"), "rb"))

st.title("🩺 Breast Cancer Detection System")

st.write("Enter the feature values to predict cancer possibility:")

inputs = []
for feature in features:
    val = st.number_input(feature, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    data = np.array(inputs).reshape(1, -1)
    data = scaler.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ Possibility of Cancer")
    else:
        st.success("✅ No Possibility of Cancer")
