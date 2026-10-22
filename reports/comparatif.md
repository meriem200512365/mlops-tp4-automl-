# Comparatif des approches - projet fraude (TP04)

Données et références reproduites avec `donnees.py` (5 000 transactions, 568 fraudes), découpage `test_size=0.25, random_state=42`. Les temps sont mesurés sur le poste d'exécution. Les coûts locaux excluent le temps humain et l'électricité ; le calcul cloud dépend de la région, du type d'instance et de la durée réelle.

| Approche | F1 test | Temps humain | Temps machine | Coût estimé (source, consultée le 22/10/2026) | Explicabilité | Export / réversibilité |
|---|---:|---|---|---|---|---|
| Logistique (réf.) | 0.7407 | Faible | À relever sur votre poste (`python src/references.py`) | 0 € de facturation cloud (poste local) | Maximale : coefficients inspectables | Totale : modèle scikit-learn local |
| Manuel (TP2) | 0.8669 | Moyen/élevé : essais et réglages manuels | À relever sur votre poste (`python src/references.py`) | 0 € de facturation cloud (poste local) | Bonne : importances et paramètres accessibles | Totale : modèle scikit-learn local |
| FLAML (budget 60 s) | 0.8582 | Faible après préparation des données | 60 s demandées, 60.6 s mesurées | 0 € de facturation cloud (poste local) | Moyenne : dépend de la famille gagnante et de ses outils | Totale : modèle Python local, dépendances à conserver |
| Vertex AI AutoML tabulaire | non mesuré | Faible à moyen | Budget et ressources configurés dans la tâche | Facturation à l'heure de nœud ; tarif tabulaire selon la configuration, consulter le calculateur / devis Google. Le déploiement d'un endpoint est facturé tant que le modèle reste déployé. [Tarifs Vertex AI](https://cloud.google.com/products/gemini-enterprise-agent-platform/pricing) | Bonne : importance des variables et évaluations disponibles | Partielle : export tabulaire possible vers Cloud Storage, mais contraintes d'exécution et explications Vertex indisponibles hors service |
| SageMaker Autopilot | non mesuré | Faible à moyen | Ressources de calcul et stockage effectivement consommés | Facturation des ressources de calcul et stockage sous-jacentes ; endpoint facturé pendant son utilisation. [Tarification SageMaker](https://aws.amazon.com/sagemaker/ai/pricing/) | Bonne : rapports de performance, importance des variables et notebooks générés | Moyenne : artefacts et notebooks exportables, intégration à l'écosystème AWS |
| Azure Automated ML | non mesuré | Faible à moyen | Calcul facturé par VM/cluster pendant son exécution | VM/stockage/réseau facturés selon région et durée ; supprimer les ressources inutilisées. [Tarifs Azure ML](https://azure.microsoft.com/en-us/pricing/details/machine-learning/) | Bonne : métriques, graphiques, explications/Responsible AI selon configuration | Moyenne : script de scoring et environnement générés, format dépend du modèle |

## Commentaire (5 lignes)

- La régression logistique convient à une première référence rapide et à un besoin d'explication simple.
- Le réglage manuel convient quand le nombre de familles est limité et que l'équipe veut garder la maîtrise des essais.
- La recherche locale convient aux expérimentations reproductibles, aux données sensibles et aux budgets d'infrastructure serrés.
- Une plateforme managée convient lorsqu'il faut intégrer entraînement, suivi, registre et déploiement à un environnement cloud déjà utilisé.
- Le choix cloud doit intégrer la portabilité, la gouvernance des données, les coûts des endpoints actifs et la dépendance au fournisseur.

## Questions d'évaluation

### 1. Réglage manuel ou AutoML ?

La référence manuelle du TP2 obtient F1 = 0.8669, contre 0.8582 pour FLAML à 60 s : l'écart est faible (0.0087) et le réglage manuel reste légèrement meilleur. L'intérêt d'AutoML tient au temps humain économisé, à l'exploration systématique de plusieurs familles et à la traçabilité des essais ; il ne garantit pas un meilleur score.

### 2. Le coût réel

Sur le poste local, une recherche de 60 secondes ne génère pas de facture cloud ; le coût direct se limite à l'électricité et au temps machine, hors temps humain. Dans le cloud, 60 secondes ne suffisent pas pour donner un prix universel : le coût dépend des types d'instances, de la région, du stockage, des étapes de préparation, et éventuellement du temps de vie de l'endpoint. Il faut multiplier les ressources facturées par leur durée au tarif de la région choisie, puis vérifier la facture réelle.

### 3. Ce qui reste à votre charge

L'équipe doit vérifier la qualité et la pertinence des données, éviter les fuites temporelles ou de cible, choisir une métrique et un seuil adaptés au coût métier, valider le modèle sur un protocole réaliste, et surveiller le modèle après son déploiement. Automatiser les essais n'automatise pas le jugement métier ou la responsabilité.

### 4. Managé ou local ?

Une plateforme managée est préférable si l'équipe a besoin d'un service intégré de calcul, de stockage, de suivi, de registre et de déploiement à l'échelle, et accepte la facturation et les contraintes du fournisseur. Une bibliothèque locale est préférable pour maîtriser les coûts et l'environnement, garder les données sur site, personnaliser le pipeline et faciliter la portabilité.

### 5. Secteur régulé

Il faut pouvoir tracer les versions des données, du code, des modèles et des paramètres ; retrouver les essais et les métriques ; expliquer les prédictions et les limites ; vérifier les biais par groupes ; reproduire l'entraînement ; contrôler les accès, la localisation et la conservation des données ; et documenter les validations humaines et le suivi en production.

### 6. La fuite de données

Le numéro de litige est créé après la contestation, mais il encode ici directement la cible. Le jeu de test conserve cette colonne, donc F1 vaut 1.0. En production, la colonne est nulle : le signal disparaît et F1 tombe à 0.0. On détecte cela en définissant les variables disponibles au moment précis de la décision, en auditant la temporalité et la provenance de chaque feature, en testant sur une fenêtre chronologique, puis en simulant le schéma réel de production. L'AutoML optimise les données qui lui sont données ; il ne connaît pas leur chronologie métier.

## Limites de ce rendu

FLAML 2.7.0 a été installé sous Python 3.13.2 et utilisé pour les quatre budgets. Les comparaisons cloud sont documentaires, sans compte ni entraînement facturé. Ce rendu contient les journaux locaux et le graphique ; les captures d'écran de l'interface MLflow restent à prendre dans votre session.

## Sources officielles consultées le 22/10/2026

- Google Cloud : [Entraîner un modèle AutoML tabulaire](https://docs.cloud.google.com/vertex-ai/docs/tabular-data/tabular-workflows/e2e-automl-train), [tarification](https://cloud.google.com/products/gemini-enterprise-agent-platform/pricing), [export de modèle tabulaire](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/export/export-model-tabular).
- AWS : [SageMaker Autopilot](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development.html), [tarification SageMaker](https://aws.amazon.com/sagemaker/ai/pricing/), [lineage des essais](https://docs.aws.amazon.com/sagemaker/latest/dg/lineage-tracking.html).
- Microsoft : [Configurer Automated ML tabulaire](https://learn.microsoft.com/azure/machine-learning/how-to-use-automated-ml-for-ml-models?view=azureml-api-2), [évaluer les expériences AutoML](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml), [planifier et gérer les coûts](https://learn.microsoft.com/en-us/azure/machine-learning/concept-plan-manage-cost?view=azureml-api-2).
