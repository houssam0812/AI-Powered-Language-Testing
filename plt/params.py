import os


##################  VARIABLES  ##################

MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
input_shape=(512)
max_length = 512
################ CONSTANTS ################
LOCAL_DATA_PATH = os.path.join("/Users","mathieusavary","code", "houssam0812","AI-Powered-Language-Testing", "raw_data")
LOCAL_REGISTRY_PATH =  os.path.join("/Users","mathieusavary","code", "houssam0812","AI-Powered-Language-Testing", "raw_data", "training_outputs")
LOCAL_TEST_PATH = os.path.join("/Users","mathieusavary","code", "houssam0812","AI-Powered-Language-Testing", "raw_data", "test.csv")

#/Users/mathieusavary/code/houssam0812/AI-Powered-Language-Testing/raw_data/training_outputs/model.h5