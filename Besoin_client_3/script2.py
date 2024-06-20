import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import requests
from datetime import datetime

def afficher_arbres_a_risque(data_json):
    # Charger le scaler et le modèle préalablement enregistrés
    scaler = joblib.load('scaler.pkl')
    best_clf = joblib.load('best_random_forest_model.pkl')

    # Charger les données de test à partir du JSON
    DataFrame = pd.read_json(data_json)

    # Préparation des données comme dans le script original
    DataFrame.loc[DataFrame['fk_arb_etat'] == 'Essouché', 'fk_arb_etat'] = 1
    DataFrame.loc[DataFrame['fk_arb_etat'] != 1, 'fk_arb_etat'] = 0
    DataFrame['fk_arb_etat'] = DataFrame['fk_arb_etat'].astype('int')

    colonnes_numeriques = ['longitude', 'latitude', 'haut_tot', 'haut_tronc', 'tronc_diam', 'age_estim', 'fk_prec_estim', 'clc_nbr_diag']
    colonnes_categorieles = ['clc_quartier', 'clc_secteur', 'fk_stadedev', 'fk_port', 'fk_pied', 'fk_situation', 'fk_revetement', 'fk_nomtech', 'feuillage', 'villeca']
    DataFrame.loc[DataFrame['remarquable'] == 'Oui', 'remarquable'] = 1
    DataFrame.loc[DataFrame['remarquable'] != 1, 'remarquable'] = 0
    DataFrame['remarquable'] = DataFrame['remarquable'].astype('int')

    # Encodage des variables catégorielles
    X = pd.concat([DataFrame[colonnes_numeriques], pd.get_dummies(DataFrame[colonnes_categorieles]), DataFrame['remarquable']], axis=1)


    # Aligner les colonnes de X avec celles utilisées pour le scaler et le modèle
    scaler_features = scaler.get_feature_names_out()
    missing_cols = set(scaler_features) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[scaler_features]

    # Standardisation des données
    X_scaled = scaler.transform(X)

    # Prédiction
    y_pred = best_clf.predict(X_scaled)

    # Ajout des prédictions au DataFrame original
    DataFrame['arbre_a_risque'] = y_pred


    # Fonction météo
    API_KEY = 'd1e8510be2ffab2edb288c8dee41c9a3'
    BASE_URL = 'http://api.openweathermap.org/data/2.5/weather'
    def meteo(latitude, longitude, api_key=API_KEY):
        url = f"{BASE_URL}?lat={latitude}&lon={longitude}&appid={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather_info = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'weather_description': data['weather'][0]['description'],
                'timestamp': datetime.utcfromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
            }
            return weather_info
        else:
            return None

    for i in range(10):
        latitude = DataFrame.iloc[i]['latitude']
        longitude = DataFrame.iloc[i]['longitude']
        weather = meteo(latitude, longitude)
        print(weather)

    # Aligner les colonnes de X avec celles utilisées pour le scaler et le modèle
    scaler_features = scaler.get_feature_names_out()
    missing_cols = set(scaler_features) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[scaler_features]


    # Filtrer les arbres essouchés à risque
    risky_trees = DataFrame[(DataFrame['fk_arb_etat'] == 1) & (DataFrame['arbre_a_risque'] == 1)]

    # Affichage des arbres à risque
    fig = px.scatter_mapbox(risky_trees, lat="latitude", lon="longitude", color="arbre_a_risque", zoom=12)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()

# Exemple d'utilisation
# Remplacer 'data_test.json' par le chemin du fichier JSON de test
afficher_arbres_a_risque('data_test.json')
