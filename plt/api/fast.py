import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from taxifare.ml_logic.registry import load_model
from taxifare.ml_logic.preprocessor import preprocess_features
import uvicorn



app = FastAPI()
app.state.model = load_model()

# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(text)
# X_pred = 
# X_processed = ...(X_pred)
# y_pred = app.state.model.predict(X_processed)
# return {'Text Score' : y_pred}

@app.get("/")
def root():
    return {'Bonjour' : 'Salut toi !'}