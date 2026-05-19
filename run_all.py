"""
Script principal - Exécute Phase 1 + Phase 2
Projet NLP : Analyse Automatique des Sentiments
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("╔══════════════════════════════════════════════════════════════════╗")
print("║  PROJET NLP - Analyse Automatique des Sentiments               ║")
print("║  Phase 1 : Prétraitement + Phase 2 : ML Classique              ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print()

# Phase 1
print(">>> Lancement Phase 1...")
result = subprocess.run([sys.executable, "phase1_preprocessing.py"], cwd=os.path.dirname(os.path.abspath(__file__)))
if result.returncode != 0:
    print("ERREUR Phase 1!")
    sys.exit(1)

print("\n\n")

# Phase 2
print(">>> Lancement Phase 2...")
result = subprocess.run([sys.executable, "phase2_ml_classique.py"], cwd=os.path.dirname(os.path.abspath(__file__)))
if result.returncode != 0:
    print("ERREUR Phase 2!")
    sys.exit(1)

print("\n\n")
print("╔══════════════════════════════════════════════════════════════════╗")
print("║  TOUTES LES PHASES TERMINÉES AVEC SUCCÈS                       ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print()
print("Fichiers générés :")
print("  data/raw_reviews.csv          - Données brutes")
print("  data/preprocessed_reviews.csv - Données prétraitées")
print("  data/tfidf_matrix.npz         - Matrice TF-IDF")
print("  output/phase1_eda_visualizations.png")
print("  output/phase1_wordclouds.png")
print("  output/phase2_ml_results.png")
print("  output/phase2_confusion_matrices.png")
print("  output/phase2_results_summary.csv")
print("  output/phase2_report.txt")
