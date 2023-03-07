import pandas as pd


def load_data(file_path: str, size:int =-1, y_index:int=-1) -> tuple(pd.DataFrame,pd.DataFrame) :
    """
    'file_path' = the path to the required file (typically 'train.csv' or 'test.csv' under a 'raw_data' folder)
    NB: it can be an absolute or a relative path !
    if 'size'=-1 the entire data is returned, else only the 1st 'size' rows (might also suffle if this is better) are returned
    'y_index' expected values = [-1,0,1,2,3,4,5] indicated which evaluation(s) will be returned
    outputs= (X_raw,y):
        X_raw= 2-column DataFrame ['text_id','full_text']
        y= a 6-column DataFrame (or a 1-column DataFrame if y_index is used)
        nb of rows in X_raw and y is controlled by the 'size' kwarg
    """
    df=pd.read_csv(file_path)
    X_raw=df[['text_id','full_text']].iloc[0:size,:]
    y=df[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].iloc[0:size,:]
    if y_index >0 and y_index < len(y.columns):
        y=y.iloc[:,[y_index]]

    return X_raw,y


def load_train_val_data(file_path:str, y_index:int=-1, split_ratio:float=0.3,reshuffle:bool=True) -> tuple(pd.DataFrame, pd.DataFrame,pd.DataFrame, pd.DataFrame) :
    """
    'file_path' = the path to the required file (typically 'train.csv' or 'test.csv' under a 'raw_data' folder)
    NB: it can be an absolute or a relative path !
    'y_index' expected values = [-1,0,1,2,3,4,5] indicated which evaluation(s) will be returned
    'split_ratio'is expected to be >0 and <1 ([0.01-0.99])
    data is reshuffled when 'reshuffle'= True!
    outputs= (X_train_raw,y_train,X_val_raw,y_val):
        X_train_raw, X_test_raw= 2x 2-column DataFrames ['text_id','full_text']
        y_train, y_test= 2x 6-column DataFrames (or  1-column DataFrames if y_index is used)
        nb of rows in (X_train_raw,y_train,X_val_raw,y_val) is controlled by the 'split_ratio' kwarg
    """
    df=pd.read_csv(file_path)
    if reshuffle:
        df=df.sample(frac=1)
    len_train=int(df.shape[0]*(1-split_ratio))
    X_train_raw=df[['text_id','full_text']].iloc[:len_train,:]
    y_train=df[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].iloc[:len_train,:]
    X_val_raw=df[['text_id','full_text']].iloc[len_train:,:]
    y_val=df[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].iloc[len_train:,:]
    if y_index >0 and y_index < len(y_train.columns):
        y_train=y_train.iloc[:,[y_index]]
        y_val=y_val.iloc[:,[y_index]]
    return X_train_raw,X_val_raw,y_train,y_val
