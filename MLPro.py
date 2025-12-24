import uvicorn
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

classifier = pickle.load(open("m1.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

@app.get("/")
def index():
    return {"message": "Breast Cancer Detection API"}

@app.post("/predict")
def predict(
    f1: float, f2: float, f3: float, f4: float, f5: float,
    f6: float, f7: float, f8: float, f9: float, f10: float
):
    input_data = np.array([[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]])
    input_scaled = scaler.transform(input_data)

    prediction = classifier.predict(input_scaled)

    if prediction[0] == 1:
        result = "Possibility of Cancer"
    else:
        result = "No Possibility of Cancer"

    return {"prediction": result}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
