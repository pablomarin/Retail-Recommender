
import os
import sys
import turicreate
from turicreate import SFrame
import pandas as pd
import json

def init():
    
    global model_s
    global model_r
    
    model_s_name = "similarity_model"
    model_s_version = "6"
    model_s_filename = "recommendation_s.model"
    
    model_r_name = "ranking_factorization_model"
    model_r_version = "1"  
    model_r_filename = "recommendation_r.model"
    
    model_s_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_s_name, model_s_version, model_s_filename)
    model_r_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_r_name, model_r_version, model_r_filename)
    
    print("model_s_path",model_s_path)
    print("model_r_path",model_r_path)
    
    model_s = turicreate.load_model(model_s_path)
    model_r = turicreate.load_model(model_r_path)
    
    
def run(raw_data):
    
    data = json.loads(raw_data)
    data = pd.DataFrame(data).to_dict(orient="list")

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
    
        return result.to_json(orient="records")
        
    except Exception as e:
        return e

