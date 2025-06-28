# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field, create_model # Ensure these are imported

# Create the FastAPI app instance
app = FastAPI()

# Load the trained PyCaret Pipeline model
# It's important that "my_api" matches the name of your saved model file (e.g., my_api.pkl)
try:
    model = load_model("my_api")
    print("Transformation Pipeline and Model Successfully Loaded")
except Exception as e:
    print(f"Error loading model: {e}")
    # You might want to raise the exception or handle it more gracefully
    # if the model loading is critical for your app to start.
    # For now, we'll let the app attempt to start and fail on prediction if model isn't loaded.
    model = None # Ensure model is None if loading fails

# Define the structure of your input data using pydantic's create_model.
# Each field is defined as a tuple: (type, default_value_or_ellipsis_for_required)
# The example values you provided (e.g., 36, 'male') are now replaced with their types.
input_model = create_model(
    "my_api_input",
    age=(int, ...),       # 'age' is an integer and required
    sex=(str, ...),       # 'sex' is a string and required
    bmi=(float, ...),     # 'bmi' is a float and required
    children=(int, ...),  # 'children' is an integer and required
    smoker=(str, ...),    # 'smoker' is a string and required
    region=(str, ...),    # 'region' is a string and required
)

# Define the structure of your output data using pydantic's create_model.
# 'prediction' is expected to be a float, as indicated by your example value.
# THIS IS THE LINE THAT WAS CHANGED: prediction is now a type, not a value.
output_model = create_model("my_api_output", prediction=(float, ...))


# Define the prediction endpoint
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    # Ensure the model was loaded successfully before attempting prediction
    if model is None:
        # Return an error response if the model failed to load
        # You might want a more specific error type or HTTP status code
        return {"prediction": None, "error": "Model not loaded"}

    # Convert the Pydantic input model to a dictionary, then to a Pandas DataFrame.
    # Using .model_dump() for Pydantic V2 compatibility (instead of .dict())
    input_df = pd.DataFrame([data.model_dump()])

    # Generate predictions using the loaded PyCaret model
    predictions = predict_model(model, data=input_df)

    # Extract the prediction label from the predictions DataFrame
    # Assuming 'prediction_label' is the column with your final prediction
    predicted_value = predictions["prediction_label"].iloc[0]

    # Return the prediction in the format defined by output_model
    return {"prediction": predicted_value}


# Standard boilerplate to run the FastAPI app using Uvicorn
# This block is for running the script directly (e.g., python my_api.py)
# When running inside Docker with the CMD ["uvicorn", "my_api:app", ...]
# this block will not be executed.
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

