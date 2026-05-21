import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import scipy.sparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
import pickle

warnings.filterwarnings('ignore')
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 70)
print("PHASE 3 : MODÈLES AVANCÉS — DEEP LEARNING, LDA & ENSEMBLE")
print("=" * 70)

print("\n[STEP 1] Chargement des données...")

df = pd.read_csv('data/preprocessed_reviews.csv')
X_tfidf = scipy.sparse.load_npz('data/tfidf_matrix.npz')
print(f"   Dataset : {len(df)} échantillons, {X_tfidf.shape[1]} features")

le = LabelEncoder()
y = le.fit_transform(df['sentiment'])
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

texts_train, texts_test, _, _ = train_test_split(
    df['cleaned_text'].tolist(), y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

print("\n[STEP 2] VADER-like Rule-Based Sentiment Analysis...")

POS_WORDS = set(['great', 'good', 'love', 'best', 'excellent', 'amazing', 'wonderful',
    'fantastic', 'perfect', 'enjoy', 'recommend', 'beautiful', 'happy', 'impressed',
    'outstanding', 'brilliant', 'superb', 'favorite', 'fun', 'nice', 'liked',
    'entertaining', 'worth', 'pleasure', 'masterpiece', 'touching', 'powerful',
    'hilarious', 'charming', 'compelling', 'riveting', 'delightful', 'engaging'])

NEG_WORDS = set(['bad', 'worst', 'terrible', 'awful', 'horrible', 'boring', 'waste',
    'poor', 'disappointing', 'stupid', 'ugly', 'hate', 'annoying', 'dull',
    'pathetic', 'ridiculous', 'worse', 'painful', 'garbage', 'trash', 'fail',
    'mediocre', 'bland', 'forgettable', 'pointless', 'weak', 'lame', 'cheesy',
    'predictable', 'overrated', 'tedious', 'cliche', 'shallow', 'atrocious'])

NEGATIONS = set(['not', 'no', 'never', 'dont', 'cant', 'wont', 'isnt', 'wasnt', 'neither', 'hardly'])
INTENSIFIERS = set(['very', 'extremely', 'incredibly', 'absolutely', 'totally', 'really', 'highly'])

def vader_score(text):
    words = str(text).lower().split()
    score = 0.0
    negate = False
    boost = 1.0
    for w in words:
        if w in NEGATIONS:
            negate = True
            continue
        if w in INTENSIFIERS:
            boost = 1.5
            continue
        if w in POS_WORDS:
            s = 1.0 * boost
            score += -s if negate else s
        elif w in NEG_WORDS:
            s = -1.0 * boost
            score += -s if negate else s
        negate = False
        boost = 1.0
    return score

vader_preds = np.array([1 if vader_score(t) > 0 else 0 for t in texts_test])
vader_acc = accuracy_score(y_test, vader_preds)
vader_f1 = f1_score(y_test, vader_preds, average='weighted')
print(f"   Accuracy : {vader_acc:.4f} | F1 : {vader_f1:.4f}")
print(classification_report(y_test, vader_preds, target_names=class_names))

print("\n[STEP 3] Deep Neural Network (MLP - simule Transformer encoder)...")

mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    max_iter=80,
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=False
)
print("   Architecture : Input(10000) → Dense(512) → Dense(256) → Dense(128) → Output(2)")
print("   Entraînement...")
mlp.fit(X_train, y_train)

mlp_preds = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, mlp_preds)
mlp_f1 = f1_score(y_test, mlp_preds, average='weighted')
print(f"   Accuracy : {mlp_acc:.4f} | F1 : {mlp_f1:.4f}")
print(classification_report(y_test, mlp_preds, target_names=class_names))

print("\n[STEP 4] Ensemble Voting (SVM + LR + MLP)...")

svm = LinearSVC(max_iter=5000, C=1.0, random_state=42)
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
svm.fit(X_train, y_train)
lr.fit(X_train, y_train)

svm_preds = svm.predict(X_test)
lr_preds = lr.predict(X_test)

ensemble_preds = np.array([
    max(set([svm_preds[i], lr_preds[i], mlp_preds[i]]),
        key=[svm_preds[i], lr_preds[i], mlp_preds[i]].count)
    for i in range(len(y_test))
])
ensemble_acc = accuracy_score(y_test, ensemble_preds)
ensemble_f1 = f1_score(y_test, ensemble_preds, average='weighted')
print(f"   Accuracy : {ensemble_acc:.4f} | F1 : {ensemble_f1:.4f}")
print(classification_report(y_test, ensemble_preds, target_names=class_names))

print("\n[STEP 5] Topic Modeling — LDA (Latent Dirichlet Allocation)...")

count_vec = CountVectorizer(max_features=5000, max_df=0.95, min_df=3)
X_counts = count_vec.fit_transform(df['cleaned_text'])

n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
lda.fit(X_counts)

feature_names = count_vec.get_feature_names_out()
print(f"\n   {n_topics} Topics extraits :")
topic_labels = []
for idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    topic_labels.append(top_words[:3])
    print(f"   Topic {idx+1}: {', '.join(top_words[:8])}")

doc_topics = lda.transform(X_counts)
df['dominant_topic'] = doc_topics.argmax(axis=1)

print("\n   Distribution des topics par sentiment :")
topic_sentiment = pd.crosstab(df['dominant_topic'], df['sentiment'], normalize='index')
print(topic_sentiment.round(3).to_string())

print("\n[STEP 6] Visualisations Phase 3...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

models_p3 = ['VADER', 'Deep NN (MLP)', 'Ensemble Voting', 'SVM (Phase 2)']
accs_p3 = [vader_acc, mlp_acc, ensemble_acc, accuracy_score(y_test, svm_preds)]
f1s_p3 = [vader_f1, mlp_f1, ensemble_f1, f1_score(y_test, svm_preds, average='weighted')]

x = np.arange(len(models_p3))
width = 0.35
axes[0, 0].bar(x - width/2, accs_p3, width, label='Accuracy', color='#3498db')
axes[0, 0].bar(x + width/2, f1s_p3, width, label='F1-Score', color='#2ecc71')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(models_p3, rotation=20, ha='right')
axes[0, 0].set_title('Phase 3 — Comparaison des Modèles Avancés', fontweight='bold')
axes[0, 0].set_ylabel('Score')
axes[0, 0].legend()
axes[0, 0].set_ylim(0.5, 1.0)

cm_ensemble = confusion_matrix(y_test, ensemble_preds)
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=class_names, yticklabels=class_names)
axes[0, 1].set_title('Matrice de Confusion — Ensemble Voting', fontweight='bold')
axes[0, 1].set_xlabel('Prédit')
axes[0, 1].set_ylabel('Réel')

topic_counts = df['dominant_topic'].value_counts().sort_index()
bars = axes[1, 0].bar(range(n_topics), topic_counts.values, color='#9b59b6')
axes[1, 0].set_xticks(range(n_topics))
axes[1, 0].set_xticklabels([f"Topic {i+1}" for i in range(n_topics)])
axes[1, 0].set_title('Distribution des Topics (LDA)', fontweight='bold')
axes[1, 0].set_ylabel('Nombre de documents')

topic_sentiment.plot(kind='bar', ax=axes[1, 1], color=['#e74c3c', '#2ecc71'])
axes[1, 1].set_title('Sentiment par Topic', fontweight='bold')
axes[1, 1].set_xlabel('Topic')
axes[1, 1].set_ylabel('Proportion')
axes[1, 1].set_xticklabels([f"Topic {i+1}" for i in range(n_topics)], rotation=0)
axes[1, 1].legend(title='Sentiment')

plt.tight_layout()
plt.savefig('output/phase3_advanced_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Visualisations : output/phase3_advanced_results.png")

print("\n[STEP 7] Sauvegarde des modèles pour l'application web...")

pickle.dump(svm, open('models/svm_model.pkl', 'wb'))
pickle.dump(lr, open('models/lr_model.pkl', 'wb'))
pickle.dump(mlp, open('models/mlp_model.pkl', 'wb'))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_app = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3, max_df=0.95)
tfidf_app.fit(df['cleaned_text'])
pickle.dump(tfidf_app, open('models/tfidf_vectorizer.pkl', 'wb'))
pickle.dump(le, open('models/label_encoder.pkl', 'wb'))

results_p3 = {
    'vader': {'accuracy': float(vader_acc), 'f1': float(vader_f1)},
    'deep_nn': {'accuracy': float(mlp_acc), 'f1': float(mlp_f1)},
    'ensemble': {'accuracy': float(ensemble_acc), 'f1': float(ensemble_f1)},
    'svm': {'accuracy': float(accuracy_score(y_test, svm_preds)), 'f1': float(f1_score(y_test, svm_preds, average='weighted'))},
    'topics': {f'topic_{i+1}': [feature_names[j] for j in lda.components_[i].argsort()[:-8:-1]] for i in range(n_topics)}
}
json.dump(results_p3, open('output/phase3_results.json', 'w'), indent=2)

print("   Modèles sauvegardés dans models/")
print("   Résultats : output/phase3_results.json")

print("\n" + "=" * 70)
print("PHASE 3 TERMINÉE AVEC SUCCÈS")
print("=" * 70)
print(f"\nRésumé :")
print(f"  - VADER (rule-based) : F1 = {vader_f1:.4f}")
print(f"  - Deep Neural Network : F1 = {mlp_f1:.4f}")
print(f"  - Ensemble Voting : F1 = {ensemble_f1:.4f}")
print(f"  - {n_topics} topics extraits par LDA")
print(f"  - Modèles sauvegardés pour l'application web")
