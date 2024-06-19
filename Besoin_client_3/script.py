import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import plotly.express as px
import matplotlib.pyplot as plt


def fonction_carte(file) :
    # Chargement des données
    DataFrame = pd.read_csv(file)

    # Préparation des données
    DataFrame.loc[DataFrame['fk_arb_etat'] == 'Essouché', 'fk_arb_etat'] = 1
    DataFrame.loc[DataFrame['fk_arb_etat'] != 1, 'fk_arb_etat'] = 0
    DataFrame['fk_arb_etat'] = DataFrame['fk_arb_etat'].astype('int')

    # Sélectionner les colonnes pertinentes
    colonnes_numeriques = ['longitude', 'latitude', 'haut_tot', 'haut_tronc', 'tronc_diam', 'age_estim', 'fk_prec_estim', 'clc_nbr_diag']
    colonnes_categorieles = ['clc_quartier', 'clc_secteur', 'fk_stadedev', 'fk_port', 'fk_pied', 'fk_situation', 'fk_revetement', 'fk_nomtech', 'feuillage', 'villeca']
    DataFrame.loc[DataFrame['remarquable'] == 'Oui', 'remarquable'] = 1
    DataFrame.loc[DataFrame['remarquable'] != 1, 'remarquable'] = 0
    DataFrame['remarquable'] = DataFrame['remarquable'].astype('int')

    # Encodage des variables catégorielles
    X = pd.concat([DataFrame[colonnes_numeriques], pd.get_dummies(DataFrame[colonnes_categorieles]), DataFrame['remarquable']], axis=1)
    y = DataFrame['fk_arb_etat']

    # Standardisation des données
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Appliquer SMOTE sur l'ensemble d'entraînement uniquement
    smote = SMOTE(random_state=42, sampling_strategy=0.2)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Entraînement du modèle RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_smote, y_train_smote)

    # Importance des caractéristiques
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    print("\nImportance des caractéristiques :")
    for f in range(X_train_smote.shape[1]):
        print(f"{X.columns[indices[f]]} : {importance[indices[f]]}")

    # Utiliser les caractéristiques les plus importantes pour entraîner un nouveau modèle
    X_train_selected = X_train_smote[:, indices[:20]]
    clf_selected = RandomForestClassifier(random_state=42)
    clf_selected.fit(X_train_selected, y_train_smote)
    X_test_selected = X_test[:, indices[:20]]

    # Prédiction
    y_pred = clf_selected.predict(X_test_selected)

    # Évaluation du modèle
    print("\nAccuracy : ", accuracy_score(y_test, y_pred))
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix :")
    print(confusion_matrix(y_test, y_pred, normalize='true'))
    print("\naccuracy Score :")
    print(accuracy_score(y_test, y_pred))

    # Ajouter les prédictions aux données de test
    test_indices = X_test_selected.shape[0]
    DataFrame.loc[y_test.index, 'arbre_a_risque'] = y_pred

    # affichage des arbres à risque
    print("\nArbres à risque :")
    print(DataFrame[DataFrame['arbre_a_risque'] == 1])

    only_arbre_a_risque = DataFrame[DataFrame['arbre_a_risque'] == 1]
    fig = px.scatter_mapbox(only_arbre_a_risque, lat="latitude", lon="longitude", color="arbre_a_risque", zoom=12)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()





