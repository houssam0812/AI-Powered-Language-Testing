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
from plt.dl_logic.preprocess import tokenizer, tokenize
from plt.model import initialize_model, compile_model, load_weights, predict_model
from plt.preprocess import tokenize, tokenizer, load_tokenizer


def initialize_model() -> Model:
    
# Import the needed model with output_hidden_states=True
    transformer = TFDebertaV2Model.from_pretrained('microsoft/deberta-v2-xlarge', output_hidden_states=True, return_dict=True)
    transformer.trainable = False
    input_ids = Input(shape=((input_shape)),dtype='int32', name='input_ids')
    attention_mask = Input(shape=((input_shape)), dtype='int32', name='attention_mask')
    transformer = transformer(dict(input_ids=input_ids,attention_mask=attention_mask))    
    hidden_states = transformer[0] # get output_hidden_states
    # Add a layer maxpool 1D
    pooling_layer = layers.GlobalMaxPooling1D()(hidden_states)
    # Now we can use selected_hiddes_states as we want
    last_hidden_layer = layers.Dense(64, activation='relu')(pooling_layer)
    # Defining the regression layer
    cohesion_output=layers.Dense(1, activation="linear", name="cohesion")(last_hidden_layer)
    syntax_output=layers.Dense(1, activation="linear", name="syntax")(last_hidden_layer)
    vocabulary_output=layers.Dense(1, activation="linear", name="vocabulary")(last_hidden_layer)
    phraseology_output=layers.Dense(1, activation="linear", name="phraseology")(last_hidden_layer)
    grammar_output=layers.Dense(1, activation="linear", name="grammar")(last_hidden_layer)
    conventions_output=layers.Dense(1, activation="linear", name="conventions")(last_hidden_layer)
    # output in a list
    output= [cohesion_output, syntax_output, vocabulary_output, phraseology_output, grammar_output, conventions_output]
    #Assembling the model
    model = Model(inputs = [input_ids, attention_mask], outputs = output)
    print("✅ model initialized")
    return model 

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def compile_model(model: Model) -> Model :
    model.compile(loss='mse', optimizer='adam',loss_weights=[1/6 for i in range(6)], metrics= root_mean_squared_error)
    return model 

def load_weights() -> keras.Model:
    from google.cloud import storage
    client = storage.Client()
    try:
        latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, model.name)
        model = keras.models.load_weigths(latest_model_path_to_save)
        print("✅ Latest model downloaded from cloud storage")
        return model
    except:
        print(f"\n❌ No model found on GCS bucket {BUCKET_NAME}")
        return None

def load_tokenizer(tokenizer=DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v2-xlarge")):
    tokenizer= DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v2-xlarge")
    return tokenizer

def tokenize(data_to_predict : pd.DataFrame) -> dict:
    tokenizer=load_tokenizer()
    tokenized_texts_to_predict = tokenizer(data_to_predict, return_tensors='tf',truncation=True, padding=True)
    return {'input_ids':tokenized_texts_to_predict['input_ids'],
            'attention_mask':tokenized_texts_to_predict['attention_mask']}

def predict(model: Model,
            X: dict,) -> np.array:
    
    if model is None:
        print(f"\n❌ no model to predict")
        return None
    
    tokenized_test_texts = tokenize(X)
    test_predictions = model.predict(X)  
    predictions_list = []
    for i in range(len(test_predictions[0])):
        prediction_dict = {'cohesion': test_predictions[0][i],
                        'syntax': test_predictions[1][i],
                        'vocabulary': test_predictions[2][i],
                        'phraseology': test_predictions[3][i],
                        'grammar': test_predictions[4][i],
                        'conventions': test_predictions[5][i]}
    predictions_list.append(prediction_dict)
    predictions_df = pd.DataFrame(predictions_list)
                                
    return predictions_df

if __name__ == "__main__":
    model = initialize_model()
    root_mean_squared_error = root_mean_squared_error()
    model = compile_model(model)
    model = load_weights()
    X = {'text': ["Hi, I'm a text to predict", "Hi, I'm a second text to predict"]}
    predict(model, X)