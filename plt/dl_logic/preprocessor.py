import numpy as np
import pandas as pd
from transformers import DebertaV2TokenizerFast

def load_train_test_data(file_path:str, y_index:int=-1, split_ratio:float=0.3,reshuffle:bool=True):
    """
    'file_path' = the path to the required file (typically 'train.csv' or 'test.csv' under a 'raw_data' folder)
    NB: it can be an absolute or a relative path !
    'y_index' expected values = [-1,0,1,2,3,4,5] indicated which evaluation(s) will be returned
    'split_ratio'is expected to be >0 and <1 ([0.01-0.99])
    data is reshuffled when 'reshuffle'= True!
    outputs= (X_train_raw,X_test_raw,y_train,y_test):
        X_train_raw, X_test_raw= 2x 2-column DataFrames ['text_id','full_text']
        y_train, y_test= 2x 6-column DataFrames (or  1-column DataFrames if y_index is used)
        nb of rows in (X_train_raw,y_train,X_val_raw,y_val) is controlled by the 'split_ratio' kwarg
    """
    df=pd.read_csv(file_path)
    if reshuffle:
        df=df.sample(frac=1)
    len_train=int(df.shape[0]*(1-split_ratio))
    target_names=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    X_train_raw=df[['text_id','full_text']].iloc[:len_train,:]
    y_train=df[target_names].iloc[:len_train,:]
    X_test_raw=df[['text_id','full_text']].iloc[len_train:,:]
    y_test=df[target_names].iloc[len_train:,:]
    if y_index >0 and y_index < len(y_train.columns):
        y_train=y_train.iloc[:,[y_index]]
        y_test=y_test.iloc[:,[y_index]]
    return list(X_train_raw), list(X_test_raw), y_train, y_test

def load_tokenizer():
    tokenizer= DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v2-xlarge")
    return tokenizer

def tokenize(data_to_predict : pd.DataFrame) -> dict:
    tokenizer=load_tokenizer()
    tokenized_texts_to_predict = tokenizer(data_to_predict, return_tensors='tf',truncation=True, padding=True)
    return {'input_ids':tokenized_texts_to_predict['input_ids'],
            'attention_mask':tokenized_texts_to_predict['attention_mask']}
