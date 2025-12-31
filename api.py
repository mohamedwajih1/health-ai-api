from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("health_model.pkl")

class HealthInput(BaseModel):
    avg_sleep_hours: float
    steps_per_day: int
    exercise_minutes: int
    calories_intake: int
    water_liters: float
    habit_completion_rate: float

@app.post("/predict")
def predict(data: HealthInput):
    features = [[
        data.avg_sleep_hours,
        data.steps_per_day,
        data.exercise_minutes,
        data.calories_intake,
        data.water_liters,
        data.habit_completion_rate
    ]]
    pred = model.predict(features)[0]
    return {"health_status": int(pred)}
