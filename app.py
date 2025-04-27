import joblib
# import numpy as np
import pandas as pd
import joblib
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import gradio

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



demo = gradio.Interface(
    fn=predict_death_event,
    inputs=["text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text"],
    outputs=["text"],
)

# Output response
demo.launch(share=True)