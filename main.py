from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
from typing import List
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
model = joblib.load("model.pkl")

app = FastAPI(title="House Price Prediction API", description="API for house price prediction using a trained ML model")

# Add CORS middleware to allow requests from any origin (or specific ones)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (use specific URLs for more security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define input schema
class PredictionInput(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int
    mainroad_yes: int
    basement_yes: int
    airconditioning_yes: int
    prefarea_yes: int
    furnishingstatus_semi_furnished: int
    furnishingstatus_unfurnished: int
    guestroom_yes: int
    hotwaterheating_yes: int

class PredictionOutput(BaseModel):
    prediction: float
    confidence: float = None

@app.get("/")
def health_check():
    logging.info("Health check endpoint called")
    return {"status": "healthy", "message": "House Price Prediction API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Convert input to model format
        input_values = np.array([[
            input_data.area, input_data.bedrooms, input_data.bathrooms,
            input_data.stories, input_data.parking,
            input_data.mainroad_yes, input_data.basement_yes,
            input_data.airconditioning_yes, input_data.prefarea_yes,
            input_data.furnishingstatus_semi_furnished, input_data.furnishingstatus_unfurnished,
            input_data.guestroom_yes, input_data.hotwaterheating_yes
        ]])
        
        # Make prediction
        prediction = model.predict(input_values)[0]
        
        # Calculate confidence score (e.g., variance of the prediction from training data)
        confidence = np.std(model.predict(np.array([[
            input_data.area, input_data.bedrooms, input_data.bathrooms, 
            input_data.stories, input_data.parking, 
            input_data.mainroad_yes, input_data.basement_yes, 
            input_data.airconditioning_yes, input_data.prefarea_yes,
            input_data.furnishingstatus_semi_furnished, input_data.furnishingstatus_unfurnished,
            input_data.guestroom_yes, input_data.hotwaterheating_yes
        ]])))

        # Log the prediction request
        logging.info(f"Prediction made: {prediction} with confidence: {confidence}")
        
        # Return prediction and confidence
        return PredictionOutput(prediction=prediction, confidence=confidence)

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch-predict")
def batch_predict(input_data: List[PredictionInput]):
    try:
        # Convert the list of input data into a format that the model can process
        input_values = np.array([[
            data.area, data.bedrooms, data.bathrooms, data.stories, data.parking,
            data.mainroad_yes, data.basement_yes, data.airconditioning_yes, data.prefarea_yes,
            data.furnishingstatus_semi_furnished, data.furnishingstatus_unfurnished, data.guestroom_yes,
            data.hotwaterheating_yes
        ] for data in input_data])

        # Make predictions for the batch
        predictions = model.predict(input_values)
        
        # Calculate confidence for each prediction
        confidences = [np.std(model.predict(input_values[i:i+1])) for i in range(len(input_values))]

        # Return predictions and confidence scores
        return [{"prediction": pred, "confidence": conf} for pred, conf in zip(predictions, confidences)]
    
    except Exception as e:
        logging.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestRegressor",
        "problem_type": "regression",
        "features": [
            "area", "bedrooms", "bathrooms", "stories", "parking", "mainroad_yes", 
            "basement_yes", "airconditioning_yes", "prefarea_yes", 
            "furnishingstatus_semi_furnished", "furnishingstatus_unfurnished", 
            "guestroom_yes", "hotwaterheating_yes"
        ]
    }

# # Testing client (for example purposes)
# client = TestClient(app)
# response = client.get("/")
# assert response.status_code == 200
# logging.info("Health check test passed.")
