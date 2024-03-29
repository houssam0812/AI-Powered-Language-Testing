import os
import pandas as pd
import numpy as np
# import keras
from keras import backend as K
# from keras import Model, Input
from typing import Tuple
from transformers import  TFDebertaV2Model
from transformers import DebertaV2TokenizerFast
from tensorflow.keras import callbacks
from tensorflow.keras import layers, Input, Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from plt.dl_logic.preprocessor import tokenize, load_tokenizer
from plt.params import *

# Define the model
# Here we use the output of the pretrained DeBerta model as an input of a dense intermediate layer,
# then we input the result in the linear regression parallele output layers, for each target.

# Load model
def load_weights(model: Model) -> Model:

    # from google.cloud import storage
    # client = storage.Client()
    # try:
    # #Define the path to save the model
    #     latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, 'model.h5')
    #     print(latest_model_path_to_save)
    # #Reference to the bucket
    #     bucket = client.get_bucket(BUCKET_NAME)
    #     blob = bucket.blob(latest_model_path_to_save)
    # #Download the model
    #     blob.download_to_filename(latest_model_path_to_save)
    # #Load the model


        path = os.path.dirname(__file__) #dl_logic
        path_parent = os.path.dirname(path) #plt
        path_parent_parent = os.path.dirname(path_parent) #AI-Powered-Language-Testing
        path_weights = os.path.join(path_parent_parent,'raw_data','training_outputs','my_best_model.epoch97-loss0.26.hdf5') #model weights

        
        model.load_weights(path_weights)
        print("✅ Latest model downloaded from load storage")
        return model





def initialize_model(input_shape= 512) -> Model:

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
    print('✅ Model initialized')
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    model.compile(loss='mse', optimizer='adam',loss_weights=[1/6 for i in range(6)], metrics= root_mean_squared_error)
    print('✅ Model compiled')
    return model



def train (model: Model,
           X: dict,
           y: pd.DataFrame,
           batch_size = 32,
           patience = 5,
           validation_data =None, #Overrides the valdiation_split)
           validation_split=0.3) -> Tuple[Model, dict]:

# Fit the model
    es = callbacks.EarlyStopping(monitor='val_loss',patience=10, restore_best_weights=True)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.0001,cooldown=2, mode=auto)

    #checkpoint = callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, save_freq='epoch', mode=min)

    callbacks = [es,reduce_lr]

    history = model.fit(x={'input_ids':tokenized_train_texts['input_ids'],
                        'attention_mask':tokenized_train_texts['attention_mask']},
                    y=train_targets,epochs=100,batch_size=4,validation_split=0.2, callbacks=callbacks,verbose=1)

    print(f"✅ model trained with with min val RMSE: {round(np.min(history.history['root_mean_squared_error']))}")

    return model, history

# Evaluate the model
def evaluate(model: Model,
             X: dict,
             y: pd.DataFrame,
             batch_size= 32) -> Tuple[Model, dict]:

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(x=X,
                            y =y)
    loss = metrics['loss']
    rmse = metrics['rmse']
    print(f"✅ model evaluated: rmse {round(rmse,1)}")
    return metrics


# Save the model weigths
def save_weights(model: Model, path: str):
    model.save_weights('/path/to/weights.h5') #path to be defined
    return f"✅ model weights saved to {path}"


# Predict the score of a new text
def prediction(model: Model,
            X: str,) -> dict:

    if model is None:
        print(f"\n❌ no model to predict")
        return None


    X_tokenized = tokenize(X)
    print(len(X_tokenized))
    test_predictions = model.predict(X_tokenized)
    def score_to_dict(score):
        metrics=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        score_dict={}
        i=0
        for m in metrics:  
            score_dict[m]=[]
            for j in range(len(score[0])):
                score_dict[m].append(float(score[i][j][0]))
            i+=1
        return score_dict
        
    return score_to_dict(test_predictions)

def select_one_text() -> str:
    test_path='/Users/mathieusavary/code/houssam0812/AI-Powered-Language-Testing/raw_data/test.csv'
    #test_path='~/code/houssam0812/AI-Powered-Language-Testing/raw_data/test.csv'
    
    path = os.path.dirname(__file__) #dl_logic  
    path_parent = os.path.dirname(path) #plt 
    path_parent_parent = os.path.dirname(path_parent) #AI-Powered-Language-Testing 
    test_path = os.path.join(path_parent_parent,'raw_data','test.csv') #Test csv

    df=pd.read_csv(test_path)
    X =df.iat[0,1]
    return X

if __name__ == "__main__":
    print(select_one_text())


