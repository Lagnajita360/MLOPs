from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("iris_model.pkl")

class InputData(BaseModel):
    data: list[float]

@app.post("/predict")
def predict(input_data: InputData):
    prediction = model.predict([input_data.data])
    return {"prediction": prediction.tolist()}
