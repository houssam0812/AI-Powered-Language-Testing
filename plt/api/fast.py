import pandas as pd
from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from plt.dl_logic.preprocessor import load_tokenizer,tokenize
from plt.dl_logic.model import load_weights,initialize_model,prediction
import uvicorn

model=initialize_model() #expecting compile to be embedded in initialize

app = FastAPI()
#1 app.state.model = load_weights(model)

# app.state.tokenizer = load_tokenizer() #no *arg expected in "load_tokenizer"

# Optional, good practice for dev purposes. Allow all middlewares
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

@app.get("/predict")
def score_text(text):
#1  model=app.state.model
#2  evaluation_score= prediction(model,text)
    return {' evaluation scores': text + " tata"}

@app.get("/test")
def score_text(text):
    return {' text_output': text + " toto"}

@app.get("/")
def root():
    return {'Bonjour' : 'Hello world !'}
