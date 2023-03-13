import os


##################  VARIABLES  ##################

MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
input_shape=(512)

################ CONSTANTS ################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".houssam0812", "raw_data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".houssam0812", "raw_data", "training_outputs")
