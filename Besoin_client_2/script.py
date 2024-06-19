import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.metrics import r2_score

def estimate_age(data_json, pkl_dict):
    scaler_X = pkl_dict["scaler_X"]
    scaler_Y = pkl_dict["scaler_Y"]
    model = pkl_dict["RandomForest"]
    ordinal = pkl_dict["ordinal"]
    label = pkl_dict["label"]

    df = pd.DataFrame(json.loads(data_json))

    X = df[['haut_tronc','tronc_diam','fk_stadedev','clc_nbr_diag','fk_nomtech','haut_tot']]
    print(X)

    X['fk_stadedev'] = ordinal.fit_transform(X[['fk_stadedev']])
    X['fk_nomtech'] = label.fit_transform(X[['fk_nomtech']])
    Y_test = df['age_estim']

    X = scaler_X.fit_transform(X)

    print(X)

    print(pkl_dict["RandomForest"])

    pred = pkl_dict["RandomForest"].predict(X) 

    print(pred)
    
    pred = pred.reshape(-1, 1)
    
    pred = scaler_Y.inverse_transform(pred)

    r2_score_model = r2_score(Y_test, pred)
    print("r2_score = ", r2_score_model)
    
    age_estimated = pd.DataFrame(pred, columns=['age_estim'])
    
    #envoie age_estimated dans un fichier json
    age_estimated.to_json('age_estimated_results.json', orient='records')



with open("dict.pkl", "rb") as f:
    pkl_dict = pickle.load(f)
    print(pkl_dict.keys())

with open("Data_Arbre_test.json", "r") as f:
    data_json = f.read()


estimate_age(data_json, pkl_dict)