"""
Fashion Style Analysis using TF-IDF and Non-Negative Matrix Factorization (NMF)
Analyzes fashion product descriptions to discover common style themes and patterns.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =======================================
# DATA LOADING AND EXPLORATION
# =======================================

# Load the cleaned dataset
fashion_data = pd.read_csv('fashion_dataset_cleaned.csv')

# Initial data exploration
print(f"Dataset shape: {fashion_data.shape}")
print(f"Columns: {fashion_data.columns.tolist()}")

print("Let's peek at the first 10 products:")
print("\n1. Description examples:")
print(fashion_data['Description'].head(10))
print("\n2. Product Name examples:")  
print(fashion_data['ProductName'].head(10))

# Check data diversity
print(f"Total products: {len(fashion_data)}")
print(f"Unique ProductNames: {fashion_data['ProductName'].nunique()}")
print(f"Unique Descriptions: {fashion_data['Description'].nunique()}")

# =======================================
# TF-IDF (Term Frequency-Inverse Document Frequency) ANALYSIS
# =======================================
"""
TF-IDF Analysis extracts the most important words from product descriptions:
- TF (Term Frequency): How often a word appears in a document
- IDF (Inverse Document Frequency): How rare/common a word is across all documents
- TF-IDF Score = TF × IDF (higher score = more important/distinctive word)
"""

print("\n" + "="*50)
print("🎯 STARTING TF-IDF ANALYSIS")
print("="*50)

# Prepare text data for analysis
fashion_data['style_text'] = fashion_data['Description'].fillna('')

# Configure TF-IDF vectorizer with optimized parameters
tfidf = TfidfVectorizer(
    max_features=5000,              # Keep top 5000 most important words
    stop_words='english',           # Remove common English words like 'the', 'and'
    min_df=10,                     # Word must appear in at least 10 products
    max_df=0.6,                    # Ignore words in more than 60% of products
    ngram_range=(1, 2),            # Include both single words and word pairs
    token_pattern=r'\b[a-zA-Z]{3,}\b',  # Only alphabetic words, 3+ characters
    lowercase=True                  # Convert all text to lowercase
)

print("Applying TF-IDF transformation...")
tfidf_matrix = tfidf.fit_transform(fashion_data['style_text'])

# Extract and rank feature importance
feature_names = tfidf.get_feature_names_out()
word_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
word_importance = list(zip(feature_names, word_scores))
word_importance.sort(key=lambda x: x[1], reverse=True)

# Remove duplicate root words to get unique fashion terms
seen_roots = set()
unique_words = []

for word, score in word_importance:
    root = word.split()[0]  # Get base word for phrases like 'slim fit'
    
    if root not in seen_roots:
        unique_words.append((word, score))
        seen_roots.add(root)
    
    if len(unique_words) >= 20:
        break

print("Top 20 most important fashion terms:")
for i, (word, score) in enumerate(unique_words):
    print(f"{i+1:2d}. {word:20s} (score: {score:.1f})")

# =======================================
# NMF (Non-negative Matrix Factorization) THEME DISCOVERY
# =======================================
"""
NMF discovers hidden style themes by decomposing the TF-IDF matrix:
- Input: Product descriptions as numerical features
- Output: Style themes and how much each product uses each theme
- Advantage: Results are interpretable and non-negative (realistic for word frequencies)
"""

print("\n" + "="*50)
print("🎨 STARTING NMF - STYLE THEME DISCOVERY")
print("="*50)

# Configure NMF model
n_themes = 8  # Number of style themes to discover

nmf_model = NMF(
    n_components=n_themes,    # Number of themes to extract
    random_state=42,          # For reproducible results
    max_iter=200             # Maximum training iterations
)

print(f"Setting up NMF to discover {n_themes} style themes...")
print("Training the model...")

# Train the NMF model
W = nmf_model.fit_transform(tfidf_matrix)  # Product-to-theme weights
H = nmf_model.components_                  # Theme-to-feature weights

print("✅ NMF training completed!")
print(f"📊 Matrix dimensions:")
print(f"   Original TF-IDF: {tfidf_matrix.shape}")
print(f"   Product-Theme Matrix (W): {W.shape}")
print(f"   Theme-Feature Matrix (H): {H.shape}")

# =======================================
# THEME ANALYSIS AND INTERPRETATION
# =======================================

print("\n" + "="*50)
print("🔍 DISCOVERED STYLE THEMES")
print("="*50)

# Analyze each discovered theme
for theme_idx in range(n_themes):
    print(f"\n🎨 THEME {theme_idx + 1}:")
    print("-" * 30)
    
    # Get the most important words for this theme
    top_words_idx = H[theme_idx].argsort()[-10:][::-1]  # Top 10 words
    top_words = [feature_names[idx] for idx in top_words_idx]
    top_scores = H[theme_idx][top_words_idx]
    
    print("Key characteristics:")
    for word, score in zip(top_words, top_scores):
        print(f"  {word}: {score:.3f}")
    
    # Count products strongly associated with this theme
    strong_products = np.where(W[:, theme_idx] > np.percentile(W[:, theme_idx], 90))[0]
    print(f"Products strongly using this theme: {len(strong_products):,}")

# =======================================
# PRODUCT THEME COMPOSITION ANALYSIS
# =======================================

print("\n" + "="*50)
print("📊 SAMPLE PRODUCT ANALYSIS")
print("="*50)

# Analyze theme composition for sample products
sample_indices = np.random.choice(len(fashion_data), size=10, replace=False)

for i, idx in enumerate(sample_indices):
    product_name = fashion_data.iloc[idx]['ProductName']
    description = fashion_data.iloc[idx]['Description']
    
    print(f"\n--- Sample Product {i+1} ---")
    print(f"Name: {product_name[:70]}...")
    print(f"Description: {description[:80]}...")
    
    # Get theme composition for this product
    theme_scores = W[idx]
    top_themes = theme_scores.argsort()[-3:][::-1]  # Top 3 themes
    
    print("Style composition:")
    for rank, theme_idx in enumerate(top_themes):
        percentage = (theme_scores[theme_idx] / theme_scores.sum()) * 100
        print(f"  Theme {theme_idx+1}: {percentage:.1f}%")

# =======================================
# DATA QUALITY ASSESSMENT
# =======================================

print("\n" + "="*50)
print("📈 DATASET QUALITY ANALYSIS")
print("="*50)

print("Dataset composition:")
print(f"Total products: {len(fashion_data):,}")
print(f"Unique product names: {fashion_data['ProductName'].nunique():,}")
print(f"Unique descriptions: {fashion_data['Description'].nunique():,}")

# Identify most frequent product names
print("\nMost common products:")
top_products = fashion_data['ProductName'].value_counts().head(10)
for i, (product, count) in enumerate(top_products.items()):
    percentage = (count / len(fashion_data)) * 100
    print(f"{i+1:2d}. {product[:60]}... ({count:,} items, {percentage:.1f}%)")

# Check for data diversity issues
if top_products.iloc[0] / len(fashion_data) > 0.5:
    print(f"\n⚠️  Data Quality Note: {top_products.iloc[0]:,} products ({top_products.iloc[0]/len(fashion_data)*100:.1f}%) have identical names")
    print("💡 This suggests product variants or data normalization issues")
    print("📊 However, NMF successfully identified themes using description diversity")

print("\n" + "="*50)
print("✨ ANALYSIS COMPLETE!")
print("="*50)