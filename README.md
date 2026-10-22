# TP04 - AutoML local et plateformes low-code

Projet de détection de fraude réalisé avec le jeu déterministe du TP2. L'environnement disponible utilise le plan B prévu dans le sujet : recherche de modèles scikit-learn au lieu de FLAML.

## Installation

Dans PowerShell, avec Python 3.10 à 3.13 :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data, src, reports | Out-Null
python donnees.py
```

## Reproduire les résultats

Les scripts utilisent un backend MLflow SQLite (`mlflow.db`) pour conserver les résultats localement :

```powershell
python src/references.py
python src/automl_sklearn.py 10
python src/automl_sklearn.py 30
python src/automl_sklearn.py 60
python src/automl_sklearn.py 120
python src/inspecter.py reports/maison_60s.log
python src/rendement.py
python src/fuite.py
```

Valeurs obtenues pendant la préparation : références F1 = 0.7407 (logistique) et 0.8669 (forêt TP2) ; recherche maison 60 s F1 test = 0.8582. Les durées et les scores peuvent varier selon la machine.

## Contenu

- `src/references.py` : références logistique et forêt manuelle, journalisées dans MLflow.
- `src/automl_sklearn.py` : recherche random sur quatre familles avec journal JSON Lines.
- `src/inspecter.py` : résumé des essais et meilleurs scores de validation.
- `src/rendement.py` : graphique de rendement des budgets dans `reports/rendement.png`.
- `src/fuite.py` : démonstration de la fuite temporelle.
- `reports/comparatif.md` : tableau comparatif, réponses aux questions et sources cloud officielles.

Ne versionnez pas `data/`, `mlflow.db`, `mlartifacts/` ni `.venv/`.
