import joblib
import prometheus_client as prom
import numpy as np
import pandas as pd
import joblib
import gradio
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score 
from fastapi import FastAPI, Request, Response

# Function for prediction

def predict_death_event(age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, serum_creatinine, serum_sodium, smoking, time):

  sample = {
        "age": int(age),
        "anaemia": int(anaemia),
        "high_blood_pressure": int(high_blood_pressure),
        "creatinine_phosphokinase": int(creatinine_phosphokinase),
        "diabetes": int(diabetes),
        "ejection_fraction": int(ejection_fraction),
        "platelets": float(platelets),
        "sex": int(sex),
        "serum_creatinine": float(serum_creatinine),
        "serum_sodium": int(serum_sodium),
        "smoking": int(smoking),
        "time": int(time)
    }

  print(sample)
  sample_df=pd.DataFrame(sample,index=[0])
  #sample_df[numerical_cols]=scaler.transform(sample_df[numerical_cols])

  xgb_clf = joblib.load('xgboost-model.pkl')

  out = xgb_clf.predict(sample_df)
  print(out)

  if out[0]==1:
      return("The patient is dead")
  else:
      return("The patient is not dead")

## Prometheus 

#def predict_event(age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, serum_creatinine, serum_sodium, smoking, time):

#   sample = {
#         "age": int(age),
#         "anaemia": int(anaemia),
#         "high_blood_pressure": int(high_blood_pressure),
#         "creatinine_phosphokinase": int(creatinine_phosphokinase),
#         "diabetes": int(diabetes),
#         "ejection_fraction": int(ejection_fraction),
#         "platelets": float(platelets),
#         "sex": int(sex),
#         "serum_creatinine": float(serum_creatinine),
#         "serum_sodium": int(serum_sodium),
#         "smoking": int(smoking),
#         "time": int(time)
#     }

#   print(sample)
#   sample_df=pd.DataFrame(sample,index=[0])
#   #sample_df[numerical_cols]=scaler.transform(sample_df[numerical_cols])


#   out = xgb_clf.predict(sample_df)
#   print(out)
#   return out

test_data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
accuracy_metric = prom.Gauge('DEATH_EVENT_r2_score', 'R2 score for random 100 test samples')

xgb_clf = joblib.load('xgboost-model.pkl')

# Function for updating metrics
def update_metrics():
    test = test_data.sample(100)
    test_feat = test.drop('DEATH_EVENT', axis=1)
    test_cnt = test['DEATH_EVENT'].values
    test_pred = xgb_clf.predict(test_feat.iloc[:, :])#[0] #['predictions'] 
    accuracy = accuracy_score(test_cnt, test_pred)#.round(3)
    accuracy_metric.set(accuracy)


# FastAPI object
app = FastAPI()
@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain",content = prom.generate_latest())


demo = gradio.Interface(
    fn=predict_death_event,
    inputs=["text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text"],
    outputs=["text"],  
)
#
# Output response

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 

# demo.launch(share=True, server_name="0.0.0.0", server_port=8001)