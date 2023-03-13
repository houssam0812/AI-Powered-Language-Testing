import os
import pandas as pd
import numpy as np
import keras
from keras import backend as K
from keras import Model, Input
from typing import Tuple
from transformers import  TFDebertaV2Model, DebertaV2TokenizerFast
from tensorflow.keras import callbacks
from tensorflow.keras import layers, Input, Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from plt.dl_logic.model import initialize_model, compile_model, load_weights, predict_model, root_mean_squared_error
from plt.dl_logic.preprocessor import tokenize, tokenizer, load_tokenizer


if __name__ == "__main__":
    model = initialize_model()
    root_mean_squared_error = root_mean_squared_error()
    model = compile_model(model)
    model = load_weights()
    X = {'text': ["Hi, I'm a text to predict", "Hi, I'm a second text to predict"]}
    predict(model, X)