"""Trace le F1 obtenu en fonction du budget de recherche."""
import mlflow
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(experiment_names=["tp4-automl"])
    auto = runs[runs["params.outil"] == "flaml"].copy()
    if auto.empty:
        raise SystemExit("Aucun run AutoML trouvé : lancez les budgets 10, 30, 60 et 120 s d'abord.")
    auto["budget"] = auto["params.budget_s"].astype(int)
    auto = auto.drop_duplicates(["params.outil", "budget"]).sort_values("budget")
    refs = runs[runs["params.outil"] == "scikit-learn"].drop_duplicates("tags.mlflow.runName")
    plt.figure(figsize=(7, 4))
    for outil, groupe in auto.groupby("params.outil"):
        plt.plot(groupe["budget"], groupe["metrics.f1"], "o-", label=f"AutoML ({outil})")
        for budget, f1 in zip(groupe["budget"], groupe["metrics.f1"]):
            print(f"{outil:15s} budget {budget:4d} s -> F1 {f1:.4f}")
    for _, run in refs.iterrows():
        nom, f1 = run["tags.mlflow.runName"], run["metrics.f1"]
        plt.axhline(f1, linestyle="--", color="gray", linewidth=1)
        plt.text(auto["budget"].min(), f1 + 0.003, f"{nom} ({f1:.4f})", fontsize=8, color="gray")
        print(f"{nom:22s} F1 {f1:.4f}")
    plt.xscale("log")
    budgets = sorted(auto["budget"].unique())
    plt.xticks(budgets, [str(b) for b in budgets])
    plt.minorticks_off()
    plt.xlabel("budget de calcul (secondes, echelle log)")
    plt.ylabel("F1 sur le jeu de test")
    plt.title("Rendement du budget de calcul")
    plt.legend(loc="lower right")
    plt.savefig("reports/rendement.png", dpi=120, bbox_inches="tight")
    print("graphique ecrit : reports/rendement.png")


if __name__ == "__main__":
    main()
