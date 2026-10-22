"""Recherche AutoML avec FLAML, selon le TP04."""
import sys
import time
from pathlib import Path

import mlflow
import pandas as pd
from flaml import AutoML
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    df = pd.read_csv("data/transactions.csv")
    X, y = df.drop(columns="fraude"), df["fraude"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    Path("reports").mkdir(exist_ok=True)
    journal = f"reports/flaml_{budget}s.log"
    automl = AutoML()
    t0 = time.time()
    automl.fit(
        Xtr,
        ytr,
        task="classification",
        metric="f1",
        time_budget=budget,
        seed=42,
        verbose=0,
        log_file_name=journal,
        log_type="all",
    )
    duree = round(time.time() - t0, 1)
    f1 = round(f1_score(yte, automl.predict(Xte)), 4)
    essayees = [nom for nom, config in automl.best_config_per_estimator.items() if config is not None]
    print(f"budget {budget} s | famille retenue : {automl.best_estimator} | F1 test = {f1} | {duree} s")
    print("familles essayees :", essayees)
    print("meilleure config :", automl.best_config)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("tp4-automl")
    with mlflow.start_run(run_name=f"flaml-{budget}s", nested=False):
        mlflow.log_params({"outil": "flaml", "budget_s": budget, "famille": automl.best_estimator})
        mlflow.log_metrics({"f1": f1, "f1_validation": round(1 - automl.best_loss, 4), "duree_s": duree})
        mlflow.log_artifact(journal)


if __name__ == "__main__":
    main()
