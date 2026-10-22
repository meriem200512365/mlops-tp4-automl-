"""Démontre pourquoi une fuite temporelle invalide un score de test."""
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("data/transactions.csv")
    rng = np.random.default_rng(0)
    df["num_dossier_litige"] = np.where(
        df["fraude"] == 1, rng.integers(10000, 99999, len(df)), 0
    )
    X, y = df.drop(columns="fraude"), df["fraude"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    automl = AutoML()
    automl.fit(
        Xtr,
        ytr,
        task="classification",
        metric="f1",
        time_budget=5,
        seed=42,
        verbose=0,
        estimator_list=["lgbm"],
        n_jobs=1,
    )
    print("famille retenue :", automl.best_estimator)
    modele = automl.model.estimator
    if hasattr(modele, "feature_importances_"):
        imp = sorted(zip(X.columns, modele.feature_importances_), key=lambda paire: -paire[1])
        print("importance des variables :", [(c, round(float(v), 3)) for c, v in imp])
    print("F1 en test, avec la colonne piegee :", round(f1_score(yte, automl.predict(Xte)), 4))
    Xprod = Xte.copy()
    Xprod["num_dossier_litige"] = 0
    print('F1 "en production" (colonne vide) :', round(f1_score(yte, automl.predict(Xprod), zero_division=0), 4))


if __name__ == "__main__":
    main()
