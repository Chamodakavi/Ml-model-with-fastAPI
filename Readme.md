# House Price Prediction API

## Problem Description:

This project provides an API for predicting house prices based on various features using a RandomForestRegressor machine learning model.

## Model Choice Justification:

The RandomForestRegressor was selected due to its ability to handle non-linear relationships and its robustness against overfitting in high-dimensional datasets.

## API Usage Examples:

1. **Health Check**

   - `GET /`
   - Returns the status of the API.

2. **Prediction Request**
   - `POST /predict`
   - Body:
     ```json
     {
       "area": 7420,
       "bedrooms": 4,
       "bathrooms": 2,
       "stories": 3,
       "parking": 2,
       "mainroad_yes": 1,
       "basement_yes": 0,
       "airconditioning_yes": 0,
       "prefarea_yes": 0,
       "furnishingstatus_semi_furnished": 0,
       "furnishingstatus_unfurnished": 0,
       "guestroom_yes": 1,
       "hotwaterheating_yes": 0
     }
     ```
   - Response:
     ```json
     {
       "prediction": 356000.0,
       "confidence": null
     }
     ```

## How to Run the Application:

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
