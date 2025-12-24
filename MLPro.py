import streamlit as st
import pickle
import numpy as np

# Load files
model = pickle.load(open("m1.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.title("Breast Cancer Detection App")
st.write("Enter feature values to predict cancer")

# Input fields
inputs = []
for f in features:
    val = st.number_input(f, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    data = np.array(inputs).reshape(1, -1)
    data = scaler.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Possibility of Cancer")
    else:
        st.success("No Possibility of Cancer")

