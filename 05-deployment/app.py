from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load pipeline
with open("pipeline_v1.bin", "rb") as f:
    pipeline = pickle.load(f)

class ClientData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI()

@app.post("/predict")
def predict(client: ClientData):
    record = client.dict()
    prediction = pipeline.predict([record])[0]  # 0 или 1
    proba = pipeline.predict_proba([record])[0, 1]  # вероятность класса 1
    return {
        "prediction": int(prediction),
        "probability": float(proba)
    }