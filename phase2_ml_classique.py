import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs("output", exist_ok=True)

print("=" * 70)
print("PHASE 2 : ALGORITHMES DE MACHINE LEARNING CLASSIQUE")
print("=" * 70)

print("\n[STEP 1] Chargement des données prétraitées (Phase 1)...")

df = pd.read_csv('data/preprocessed_reviews.csv')
X_tfidf = scipy.sparse.load_npz('data/tfidf_matrix.npz')

print(f"   Dataset : {len(df)} échantillons")
print(f"   Features TF-IDF : {X_tfidf.shape[1]}")
print(f"   Classes : {df['sentiment'].unique()}")

le = LabelEncoder()
y = le.fit_transform(df['sentiment'])
class_names = le.classes_
print(f"   Encoding : {dict(zip(class_names, le.transform(class_names)))}")

print("\n[STEP 2] Split Train/Test (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train : {X_train.shape[0]} échantillons")
print(f"   Test  : {X_test.shape[0]} échantillons")

print("\n[STEP 3] Entraînement des modèles ML classiques...")

models = {
    'SVM (Linear)': LinearSVC(max_iter=5000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=20),
    'Naive Bayes': MultinomialNB(alpha=1.0),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
}

results = {}

for name, model in models.items():
    print(f"\n   Training: {name}...")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cv_scores = cross_val_score(model, X_tfidf, y, cv=5, scoring='accuracy', n_jobs=-1)
    
    results[name] = {
        'model': model,
        'accuracy': acc,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'report': classification_report(y_test, y_pred, target_names=class_names)
    }
    
    print(f"   ✓ Accuracy: {acc:.4f} | F1: {f1:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n\n[STEP 4] Comparaison des modèles...")
print("\n" + "-" * 70)
print(f"{'Modèle':<25} {'Accuracy':<12} {'F1-Score':<12} {'CV Mean':<12} {'CV Std':<10}")
print("-" * 70)

for name, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"{name:<25} {res['accuracy']:<12.4f} {res['f1_score']:<12.4f} {res['cv_mean']:<12.4f} {res['cv_std']:<10.4f}")
print("-" * 70)

best_model_name = max(results, key=lambda x: results[x]['f1_score'])
print(f"\n   🏆 Meilleur modèle : {best_model_name} (F1={results[best_model_name]['f1_score']:.4f})")

print(f"\n[STEP 5] Rapport détaillé - {best_model_name}...")
print("\n" + results[best_model_name]['report'])

print("[STEP 6] Génération des visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
f1_scores = [results[m]['f1_score'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35
axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db')
axes[0, 0].bar(x + width/2, f1_scores, width, label='F1-Score', color='#2ecc71')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
axes[0, 0].set_title('Comparaison des Modèles', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Score')
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 1.1)

cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=class_names, yticklabels=class_names)
axes[0, 1].set_title(f'Matrice de Confusion - {best_model_name}', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Prédit')
axes[0, 1].set_ylabel('Réel')

cv_means = [results[m]['cv_mean'] for m in model_names]
cv_stds = [results[m]['cv_std'] for m in model_names]
axes[1, 0].barh(model_names, cv_means, xerr=cv_stds, color='#9b59b6', alpha=0.8)
axes[1, 0].set_title('Cross-Validation (5-Fold)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Accuracy Moyenne')

all_accuracies = pd.DataFrame({
    'Modèle': model_names,
    'Accuracy': accuracies,
    'F1-Score': f1_scores,
    'CV Mean': cv_means
}).set_index('Modèle')
sns.heatmap(all_accuracies, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 1])
axes[1, 1].set_title('Résumé des Performances', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('output/phase2_ml_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Visualisations sauvegardées : output/phase2_ml_results.png")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, (name, res) in enumerate(results.items()):
    row, col = idx // 3, idx % 3
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                xticklabels=class_names, yticklabels=class_names)
    axes[row, col].set_title(f'{name}\nAcc={res["accuracy"]:.3f}', fontsize=10)
    axes[row, col].set_xlabel('Prédit')
    axes[row, col].set_ylabel('Réel')

plt.suptitle('Matrices de Confusion - Tous les Modèles', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/phase2_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Matrices de confusion : output/phase2_confusion_matrices.png")

print(f"\n[STEP 7] Hyperparameter Tuning - SVM...")

param_grid = {'C': [0.1, 1, 10], 'max_iter': [3000, 5000]}
grid_search = GridSearchCV(LinearSVC(random_state=42), param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"   Meilleurs paramètres : {grid_search.best_params_}")
print(f"   Meilleur F1 (CV) : {grid_search.best_score_:.4f}")

y_pred_tuned = grid_search.predict(X_test)
acc_tuned = accuracy_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted')
print(f"   Après tuning - Accuracy: {acc_tuned:.4f} | F1: {f1_tuned:.4f}")

print("\n[STEP 8] Sauvegarde des résultats...")

results_df = pd.DataFrame({
    'Modèle': model_names,
    'Accuracy': accuracies,
    'F1-Score': f1_scores,
    'CV Mean': cv_means,
    'CV Std': cv_stds
}).sort_values('F1-Score', ascending=False)

results_df.to_csv('output/phase2_results_summary.csv', index=False)
print("   Résultats : output/phase2_results_summary.csv")

with open('output/phase2_report.txt', 'w', encoding='utf-8') as f:
    f.write("PHASE 2 - RAPPORT ML CLASSIQUE\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Dataset : {len(df)} échantillons\n")
    f.write(f"Features : {X_tfidf.shape[1]} (TF-IDF)\n")
    f.write(f"Split : 80% train / 20% test\n\n")
    f.write("RÉSULTATS PAR MODÈLE\n")
    f.write("-" * 70 + "\n\n")
    for name, res in sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True):
        f.write(f"\n{'='*50}\n{name}\n{'='*50}\n")
        f.write(res['report'])
        f.write(f"\nCV Score: {res['cv_mean']:.4f} ± {res['cv_std']:.4f}\n")
    f.write(f"\n\nMEILLEUR MODÈLE : {best_model_name}\n")
    f.write(f"Accuracy : {results[best_model_name]['accuracy']:.4f}\n")
    f.write(f"F1-Score : {results[best_model_name]['f1_score']:.4f}\n")

print("   Rapport : output/phase2_report.txt")

print("\n" + "=" * 70)
print("PHASE 2 TERMINÉE AVEC SUCCÈS")
print("=" * 70)
print(f"\nRésumé :")
print(f"  - 6 modèles ML entraînés et évalués")
print(f"  - Meilleur : {best_model_name} (F1={results[best_model_name]['f1_score']:.4f})")
print(f"  - Hyperparameter tuning effectué")
print(f"  - Visualisations et rapports dans output/")
