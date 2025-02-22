from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
import pandas as pd
import joblib
import pickle

app = FastAPI()

OPTIMAL_THRESHOLD = 0.6278
ever_married_mapping = {"No": 0, "Yes": 1}
residence_type_mapping = {"Rural": 0, "Urban": 1}

model = joblib.load("LogisticRegression_best_model.pkl")

class StrokePredictionInput(BaseModel):
    gender: str
    age: float = Field(..., ge=0, le=100)
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float = Field(..., ge=0, le=500)
    bmi: float = Field(..., ge=0, le=100)
    smoking_status: str
    stroke: Optional[int] = None

    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('gender must be either "Male" or "Female"')
        return v

    @validator('ever_married')
    def validate_ever_married(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('ever_married must be "Yes" or "No"')
        return v

    @validator('work_type')
    def validate_work_type(cls, v):
        valid_work_types = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
        if v not in valid_work_types:
            raise ValueError(f'work_type must be one of {valid_work_types}')
        return v

    @validator('residence_type')
    def validate_residence_type(cls, v):
        if v not in ['Urban', 'Rural']:
            raise ValueError('residence_type must be "Urban" or "Rural"')
        return v

    @validator('smoking_status')
    def validate_smoking_status(cls, v):
        valid_smoking_status = ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
        if v not in valid_smoking_status:
            raise ValueError(f'smoking_status must be one of {valid_smoking_status}')
        return v

import pandas as pd

def make_prediction(input_data: StrokePredictionInput):
    input_data_dict = input_data.dict()
    
    input_data_dict['ever_married'] = ever_married_mapping[input_data_dict['ever_married']]
    input_data_dict['residence_type'] = residence_type_mapping[input_data_dict['residence_type']]
    
    input_data_df = pd.DataFrame([input_data_dict])
    
    expected_columns = [
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
        'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
    ]
    
    input_data_df = input_data_df[expected_columns]
    
    try:
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_data_df)[0][1]
            prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0
        else:
            raise AttributeError("The model does not have 'predict_proba' method.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
    
    return prediction, probability


@app.post("/predict")
def predict(input_data: StrokePredictionInput):
    prediction, probability = make_prediction(input_data)
    
    return {
        "prediction": prediction,
        "probability": probability
    }

