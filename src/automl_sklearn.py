"""Plan B AutoML du TP04 : recherche random reproductible sur quatre familles."""
import json
import random
import sys
import time
from pathlib import Path
import os

import mlflow
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    if len(sys.argv) > 2 and sys.argv[2] == "--replay":
        replay_run(budget)
        return
    rng = random.Random(42)
    espace = {
        "logistique": lambda: {"C": 10 ** rng.uniform(-3, 2)},
        "rf": lambda: {
            "n_estimators": rng.choice([100, 200, 400]),
            "max_depth": rng.choice([6, 10, 16, None]),
            "min_samples_leaf": rng.choice([1, 2, 5]),
        },
        "extra_tree": lambda: {
            "n_estimators": rng.choice([100, 200, 400]),
            "max_depth": rng.choice([6, 10, 16, None]),
        },
        "hist_gb": lambda: {
            "learning_rate": 10 ** rng.uniform(-2, -0.5),
            "max_leaf_nodes": rng.choice([15, 31, 63]),
            "max_iter": rng.choice([100, 200, 400]),
        },
    }
    classes = {
        "logistique": lambda c: LogisticRegression(max_iter=2000, **c),
        "rf": lambda c: RandomForestClassifier(random_state=42, n_jobs=1, **c),
        "extra_tree": lambda c: ExtraTreesClassifier(random_state=42, n_jobs=1, **c),
        "hist_gb": lambda c: HistGradientBoostingClassifier(random_state=42, **c),
    }
    df = pd.read_csv("data/transactions.csv")
    X, y = df.drop(columns="fraude"), df["fraude"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    Path("reports").mkdir(exist_ok=True)
    journal_path = Path(f"reports/maison_{budget}s.log")
    debut, n, meilleur = time.time(), 0, None
    familles = list(espace)
    with journal_path.open("w", encoding="utf-8") as journal:
        while (time.time() - debut < budget or n < len(familles)) and n < 10_000:
            famille = familles[n] if n < len(familles) else rng.choice(familles)
            config = espace[famille]()
            score = cross_val_score(
                classes[famille](config), Xtr, ytr, cv=3, scoring="f1", n_jobs=1
            ).mean()
            n += 1
            if n > len(familles) and time.time() - debut >= budget:
                break
            entree = {
                "record_id": n,
                "learner": famille,
                "config": config,
                "validation_loss": 1 - float(score),
                "wall_clock_time": round(time.time() - debut, 2),
            }
            journal.write(json.dumps(entree) + "\n")
            journal.flush()
            if meilleur is None or score > meilleur[0]:
                meilleur = (score, famille, config)
    duree = round(time.time() - debut, 1)
    score_val, famille, config = meilleur
    modele = classes[famille](config).fit(Xtr, ytr)
    f1 = round(f1_score(yte, modele.predict(Xte)), 4)
    print(f"budget {budget} s | {n} essais | famille retenue : {famille} | F1 test = {f1} | {duree} s")
    print("meilleure config :", config)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("tp4-automl")
    with mlflow.start_run(run_name=f"maison-{budget}s", nested=False):
        mlflow.log_params({"outil": "sklearn-maison", "budget_s": budget, "famille": famille})
        mlflow.log_metrics(
            {"f1": f1, "f1_validation": round(float(score_val), 4), "duree_s": duree, "nb_essais": n}
        )
        mlflow.log_artifact(str(journal_path))


def replay_run(budget):
    """Enregistre dans MLflow un budget déjà calculé et conservé dans reports/."""
    journal_path = Path(f"reports/maison_{budget}s.log")
    if not journal_path.exists():
        raise SystemExit(f"Journal absent : {journal_path}")
    essais = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not essais:
        raise SystemExit(f"Journal vide : {journal_path}")
    gagnant = min(essais, key=lambda entree: entree["validation_loss"])
    df = pd.read_csv("data/transactions.csv")
    X, y = df.drop(columns="fraude"), df["fraude"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    famille, config = gagnant["learner"], gagnant["config"]
    classes = {
        "logistique": lambda c: LogisticRegression(max_iter=2000, **c),
        "rf": lambda c: RandomForestClassifier(random_state=42, n_jobs=1, **c),
        "extra_tree": lambda c: ExtraTreesClassifier(random_state=42, n_jobs=1, **c),
        "hist_gb": lambda c: HistGradientBoostingClassifier(random_state=42, **c),
    }
    modele = classes[famille](config).fit(Xtr, ytr)
    f1 = round(f1_score(yte, modele.predict(Xte)), 4)
    duree = max(float(e.get("wall_clock_time", 0)) for e in essais)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("tp4-automl")
    with mlflow.start_run(run_name=f"maison-{budget}s", nested=False):
        mlflow.log_params({"outil": "sklearn-maison", "budget_s": budget, "famille": famille})
        mlflow.log_metrics({"f1": f1, "f1_validation": round(1 - gagnant["validation_loss"], 4), "duree_s": duree, "nb_essais": len(essais)})
        mlflow.log_artifact(str(journal_path))
    print(f"reprise budget {budget} s | {len(essais)} essais | famille retenue : {famille} | F1 test = {f1}")


if __name__ == "__main__":
    main()
