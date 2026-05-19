import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os

ENGLISH_STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
    'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', 'also', 'would', 'could', 'one', 'two', 'get', 'got', 'much',
    'even', 'like', 'make', 'made', 'well', 'back', 'still', 'way', 'take', 'see',
    'come', 'know', 'think', 'go', 'going', 'film', 'movie', 'show', 'time', 'first'
])

print("=" * 70)
print("PHASE 1 : EXTRACTION, PRÉTRAITEMENT ET EXPLORATION DES DONNÉES")
print("=" * 70)

print("\n[STEP 1] Chargement du dataset IMDb Reviews...")

os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)

df = pd.read_csv('data/IMDB-Dataset.csv')
print(f"   Dataset complet : {len(df)} reviews")
print(f"   Colonnes : {list(df.columns)}")

df = df.sample(n=5000, random_state=42).reset_index(drop=True)
print(f"   Échantillon utilisé : {len(df)} reviews")

df.columns = ['review_text', 'sentiment']
df['review_id'] = range(1, len(df) + 1)

print(f"   Classes : {df['sentiment'].unique()}")

df.to_csv('data/raw_reviews.csv', index=False)
print(f"   Sauvegardé dans : data/raw_reviews.csv")

print("\n[STEP 2] Exploration des données (EDA)...")

print(f"\n   Dimensions du dataset : {df.shape}")
print(f"\n   Distribution des sentiments :")
print(df['sentiment'].value_counts().to_string())

df['text_length'] = df['review_text'].apply(len)
df['word_count'] = df['review_text'].apply(lambda x: len(str(x).split()))
print(f"\n   Statistiques texte :")
print(f"   - Longueur moyenne : {df['text_length'].mean():.0f} caractères")
print(f"   - Nombre moyen de mots : {df['word_count'].mean():.1f}")
print(f"   - Min mots : {df['word_count'].min()}")
print(f"   - Max mots : {df['word_count'].max()}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sentiment_counts = df['sentiment'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, color=colors)
axes[0, 0].set_title('Distribution des Sentiments', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Nombre de reviews')

axes[0, 1].hist(df['word_count'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Distribution du Nombre de Mots', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Nombre de mots')
axes[0, 1].set_ylabel('Fréquence')

for sent, color in zip(['positive', 'negative'], ['#2ecc71', '#e74c3c']):
    subset = df[df['sentiment'] == sent]['word_count']
    axes[1, 0].hist(subset, alpha=0.6, label=sent, bins=40, color=color)
axes[1, 0].set_title('Longueur par Sentiment', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Nombre de mots')
axes[1, 0].legend()

length_bins = pd.cut(df['word_count'], bins=[0, 100, 200, 500, 1000, 5000], labels=['<100', '100-200', '200-500', '500-1000', '>1000'])
ct = pd.crosstab(length_bins, df['sentiment'])
ct.plot(kind='bar', ax=axes[1, 1], color=colors)
axes[1, 1].set_title('Sentiment par Longueur de Review', fontsize=12, fontweight='bold')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
axes[1, 1].legend(title='Sentiment')

plt.tight_layout()
plt.savefig('output/phase1_eda_visualizations.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Visualisations sauvegardées : output/phase1_eda_visualizations.png")

print("\n[STEP 3] Prétraitement des données...")

stop_words = ENGLISH_STOPWORDS

def simple_lemmatize(word):
    if word.endswith('ing') and len(word) > 5:
        return word[:-3]
    if word.endswith('tion') and len(word) > 5:
        return word
    if word.endswith('ly') and len(word) > 4:
        return word[:-2]
    if word.endswith('ed') and len(word) > 4:
        return word[:-2]
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        return word[:-1]
    return word

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [simple_lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

df['cleaned_text'] = df['review_text'].apply(preprocess_text)

print(f"   Exemple avant  : {df['review_text'].iloc[0][:100]}...")
print(f"   Exemple après  : {df['cleaned_text'].iloc[0][:100]}...")
print(f"   Textes vides après nettoyage : {(df['cleaned_text'] == '').sum()}")

df = df[df['cleaned_text'] != ''].reset_index(drop=True)

print("\n[STEP 4] Vectorisation TF-IDF...")

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3, max_df=0.95)
X_tfidf = tfidf.fit_transform(df['cleaned_text'])

print(f"   Matrice TF-IDF : {X_tfidf.shape}")
print(f"   Vocabulaire : {len(tfidf.vocabulary_)} termes")

print("\n   Top termes par sentiment :")
for sentiment in ['positive', 'negative']:
    mask = (df['sentiment'] == sentiment).values
    if mask.sum() > 0:
        mean_tfidf = X_tfidf[mask].mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[-10:][::-1]
        feature_names = tfidf.get_feature_names_out()
        top_terms = [feature_names[i] for i in top_indices]
        print(f"   {sentiment:>10} : {', '.join(top_terms[:8])}")

print("\n[STEP 5] Génération des WordClouds...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, sentiment in enumerate(['positive', 'negative']):
    text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
    if text.strip():
        wc = WordCloud(width=800, height=500, background_color='white',
                      colormap='Greens' if sentiment == 'positive' else 'Reds',
                      max_words=100)
        wc.generate(text)
        axes[idx].imshow(wc, interpolation='bilinear')
    axes[idx].set_title(f'WordCloud - {sentiment.capitalize()}', fontsize=14, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('output/phase1_wordclouds.png', dpi=150, bbox_inches='tight')
plt.close()
print("   WordClouds sauvegardées : output/phase1_wordclouds.png")

print("\n[STEP 6] Sauvegarde des données prétraitées...")

df.to_csv('data/preprocessed_reviews.csv', index=False)

import scipy.sparse
scipy.sparse.save_npz('data/tfidf_matrix.npz', X_tfidf)

vocab_df = pd.DataFrame({'term': tfidf.get_feature_names_out()})
vocab_df.to_csv('data/tfidf_vocabulary.csv', index=False)

print(f"   Données prétraitées : data/preprocessed_reviews.csv")
print(f"   Matrice TF-IDF : data/tfidf_matrix.npz")
print(f"   Vocabulaire : data/tfidf_vocabulary.csv")

print("\n" + "=" * 70)
print("PHASE 1 TERMINÉE AVEC SUCCÈS")
print("=" * 70)
print(f"\nRésumé :")
print(f"  - {len(df)} reviews IMDb chargées et nettoyées")
print(f"  - {X_tfidf.shape[1]} features TF-IDF extraites")
print(f"  - Visualisations générées dans output/")
print(f"  - Données prêtes pour Phase 2 (ML classique)")
