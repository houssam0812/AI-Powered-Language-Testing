import os
import numpy as np

##################  VARIABLES  ##################
LOCAL_REGISTRY_PATH = os.environ.get("LOCAL_REGISTRY_PATH")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
input_shape=(512)

