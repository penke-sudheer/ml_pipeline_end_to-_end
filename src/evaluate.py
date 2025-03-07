import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse

os.environ["MLFLOW_TRACKING_URL"]="https://dagshub.com/penke-sudheer/ml_pipeline_end_to_end.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="penke-sudheer"
os.environ["MLFLOW_TRACKING_PASSWORD"]="9c6efee5f6304ba5c315b05cc10e770d46a2182d"


# load parameters from params.yaml
params=yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/penke-sudheer/ml_pipeline_end_to_end.mlflow")

    # load the model from the disk
    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)

    # log matrics to mlflow

    mlflow.log_metric("accuracy",accuracy)
    print("model accuracy:{accuracy}")
if __name__=="__main__":
    evaluate(params['data'],params['model'])