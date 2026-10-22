"""Résume les essais conservés dans un journal JSON Lines."""
import json
import sys
from collections import Counter


def main():
    fichier = sys.argv[1] if len(sys.argv) > 1 else "reports/maison_60s.log"
    essais = []
    with open(fichier, encoding="utf-8") as source:
        for ligne in source:
            ligne = ligne.strip()
            if ligne.startswith("{"):
                entree = json.loads(ligne)
                if "learner" in entree and entree.get("validation_loss") is not None:
                    essais.append(entree)
    if not essais:
        raise SystemExit(f"Aucun essai exploitable dans {fichier}")
    print(len(essais), "essais journalises dans", fichier)
    print("essais par famille :", dict(Counter(e["learner"] for e in essais)))
    meilleurs = {}
    for entree in essais:
        famille = entree["learner"]
        if famille not in meilleurs or entree["validation_loss"] < meilleurs[famille]["validation_loss"]:
            meilleurs[famille] = entree
    print("\nMeilleur F1 de validation par famille (F1 = 1 - perte) :")
    for famille, entree in sorted(meilleurs.items(), key=lambda paire: paire[1]["validation_loss"]):
        print(f"  {famille:15s} F1 val = {1-entree['validation_loss']:.4f} (trouve apres {entree.get('wall_clock_time', 0):.1f} s)")
    gagnant = min(essais, key=lambda entree: entree["validation_loss"])
    print("\nEssai gagnant :", gagnant["learner"], gagnant.get("config"))


if __name__ == "__main__":
    main()
