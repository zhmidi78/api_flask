# Import all packages and libraries
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

import pickle

app = Flask(__name__, template_folder='templates')

########################
# Lecture des fichiers #
########################
def lecture_X_test_original():
    X_test_original = pd.read_csv("X_test_original.csv")
    X_test_original = X_test_original.rename(columns=str.lower)
    return X_test_original

def lecture_X_test_clean():
    X_test_clean = pd.read_csv("X_test_clean.csv")
    return X_test_clean

#################################################
# Lecture du modèle de prédiction et des scores #
#################################################
model_LGBM = pickle.load(open("trained_model.pkl", "rb"))
y_pred_lgbm = model_LGBM.predict(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))    # Prédiction de la classe 0 ou 1
y_pred_lgbm_proba = model_LGBM.predict_proba(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1)) # Prédiction du % de risque

# Récupération du score du client
y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'],
                                lecture_X_test_clean()['sk_id_curr']], axis=1)

# Récupération de la décision
y_pred_lgbm_df = pd.DataFrame(y_pred_lgbm, columns=['prediction'])
y_pred_lgbm_df = pd.concat([y_pred_lgbm_df, lecture_X_test_clean()['sk_id_curr']], axis=1)
y_pred_lgbm_df['client'] = np.where(y_pred_lgbm_df.prediction == 1, "non solvable", "solvable")
y_pred_lgbm_df['decision'] = np.where(y_pred_lgbm_df.prediction == 1, "refuser", "accorder")


@app.route("/", methods = ['GET'])
def template():
    return render_template("index.html")

@app.route("/", methods = ['POST'])
def predict():
    all_id_client = list(lecture_X_test_original()['sk_id_curr'])

    ID = request.form['id_client']
    ID = int(ID)
    if ID not in all_id_client:
        number="L'identifiant que vous avez saisi n'est pas valide !"
        prediction="NA"
        solvabilite="NA"
        decision="NA"
    else :
        number=""
        score = y_pred_lgbm_proba_df[y_pred_lgbm_proba_df['sk_id_curr']==ID]
        prediction = round(score.proba_classe_1.iloc[0]*100, 1)
        solvabilite = y_pred_lgbm_df.loc[y_pred_lgbm_df['sk_id_curr']==ID, "client"].values
        solvabilite = solvabilite[0]
        decision = y_pred_lgbm_df.loc[y_pred_lgbm_df['sk_id_curr']==ID, "decision"].values
        decision = decision[0]

    return render_template('index.html', 
                           number=number,
                           identifiant_text=ID,
                           prediction_text=prediction,
                           solvabilite_text=solvabilite,
                           decision_text=decision)

if __name__ == "__main__":
     app.run(debug=True)