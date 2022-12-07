# Databricks notebook source
import os
import json
import shutil
import mlflow
import numpy as np
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

# COMMAND ----------

MODELS_DIR = "/dbfs/FileStore/Retail-Recommender/models/"
MLFLOW_MODELS_DIR = "/dbfs/FileStore/Retail-Recommender/mlflow_models/"
MLFLOW_MODELS_URI = "dbfs:/FileStore/Retail-Recommender/mlflow_models/"
os.makedirs(MLFLOW_MODELS_DIR, exist_ok=True)

# COMMAND ----------

class ModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to use Turicreate Models
    """

    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """
        import turicreate

        self.model_s = turicreate.load_model(context.artifacts["model_s_path"])
        self.model_r = turicreate.load_model(context.artifacts["model_r_path"])

    def predict(self, context, model_input):
        """This is an abstract function. We customized it into a method to fetch the FastText model.
        Args:
            context ([type]): MLflow context where the model artifact is stored.
            model_input ([type]): the input data to fit into the model.
        Returns:
            [type]: the loaded model artifact.
        """

        import turicreate

        data = model_input.to_dict('list')

        try:
            if "visitorid" in data:
                print("visitorid found")

                # If observation side data exists
                if ("period" or "month" or "weekday") in data:
                    print("Using Ranking Factorization Model")
                    users_query = turicreate.SFrame(
                        {"visitorid": data["visitorid"], 
                        "period": data["period"] if "period" in data else ["Night"] * len(data["visitorid"]),
                        "month": data["month"] if "month" in data else [8] * len(data["visitorid"]),
                        "weekday": data["weekday"] if "weekday" in data else [2] * len(data["visitorid"])
                        }
                    )
                    result = model_r.recommend(users=users_query, k=10).to_dataframe()

                else:
                    print("Using Similarity Model")
                    result = model_s.recommend(users=data["visitorid"], k=10).to_dataframe()

                if "itemid" in data:
                    return "Error: Request cannot have visitorid and itemid together. Please make separate requests for visitorid recommendations and for itemid basket similarity"

            elif "itemid" in data:
                    print("Using Similarity Model")
                    result = model_s.get_similar_items(data["itemid"], k=10).to_dataframe()

            else:
                return "Error: Payload must contain at least a field named visitorid or itemid"

            return result
            
        except Exception as e:
            return e


# COMMAND ----------


test_load1 = '''
    [
        {"visitorid":6000, "period":"Night",  "weekday":2},
        {"visitorid":567987987988, "period":"Night", "weekday":2}
    ]
'''

test_load2 = '''
    [
        {"visitorid":6000},
        {"visitorid":567987987988}
    ]
'''

test_load3 = '''
    [
        {"visitorid":6000,"period":"Night",  "weekday":2 , "month": 9},
        {"visitorid":567987987988,"period":"Night",  "weekday":2 , "month": 9}
    ]
'''

test_load4 = '''
    [
        {"itemid":152913},
        {"itemid":355908}
    ]
'''


# COMMAND ----------

# MLflow contains utilities to create a conda environment used to serve models.
# The necessary dependencies are added to a conda.yaml file which is logged along with the model.
conda_env =  _mlflow_conda_env(
    additional_conda_deps=None,
    additional_pip_deps=["turicreate"],
    additional_conda_channels=None,
)

# artifacts is where we give the local path to the models and any other data necessary to the container
artifacts = {
    "model_s_path": MODELS_DIR + "recommendation_s.model",
    "model_f_path": MODELS_DIR + "recommendation_r.model"
    }

# Path where to save the MLFLow model
mlflow_model_path = MLFLOW_MODELS_DIR + "recommendation.model"
if shutil.shutil.rmtree(mlflow_model_path)

# Use above defined Custom Model Wrapper
mlflow.pyfunc.save_model(
    path=mlflow_model_path,
    python_model=ModelWrapper(),
    artifacts=artifacts,
)

# COMMAND ----------

model_version = mlflow.register_model(
                    model_uri = MLFLOW_MODELS_URI + "recommendation.model",
                    name = "recommender"
                )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the model to Staging

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
        name="recommender",
        version=model_version.version,
        stage="Staging", # "Archived" or "Staging" or "Production"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model serving
# MAGIC You need a Databricks token to issue requests to your model endpoint. You can generate a token from the User Settings page (click Settings in the left sidebar). Copy the token into the next cell.

# COMMAND ----------

import os
os.environ["DATABRICKS_TOKEN"] = "YOUR_PERSONAL_TOKEN"

# COMMAND ----------

# MAGIC %md
# MAGIC Click Models in the left sidebar and navigate to the registered wine model. Click the serving tab, and then click ***Enable Serving***.
# MAGIC 
# MAGIC Then, under Call The Model, click the Python button to display a Python code snippet to issue requests. Copy the code into this notebook. It should look similar to the code in the next cell.
# MAGIC 
# MAGIC You can use the token to make these requests from outside Databricks notebooks as well.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring

# COMMAND ----------

import os
import requests

def score_model(data_json):
  url = 'https://<YOUR_DATABRICKS_WORKSPACE>.azuredatabricks.net/model/recommender/Staging/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()


# COMMAND ----------

score_model(test_load1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transition model to Production

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
        name="recommender",
        version=model_version.version,
        stage="Production", # "Archived" or "Staging" or "Production"
)

# COMMAND ----------


