import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("+" * 70)
print("  PROJET NLP - Analyse Automatique des Sentiments")
print("  Phase 1 + Phase 2 + Phase 3")
print("+" * 70)
print()

scripts = [
    ("Phase 1 : Prétraitement", "phase1_preprocessing.py"),
    ("Phase 2 : ML Classique", "phase2_ml_classique.py"),
    ("Phase 3 : Modèles Avancés", "phase3_advanced_models.py"),
]

for label, script in scripts:
    print(f">>> {label}...")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"ERREUR {label}!")
        sys.exit(1)
    print("\n")

print("+" * 70)
print("  TOUTES LES PHASES TERMINÉES")
print("+" * 70)
print()
print("Pour lancer l'application web :")
print("  streamlit run app.py")
print()
print("Pour générer la présentation :")
print("  cd presentation && node generate_pptx.js")
