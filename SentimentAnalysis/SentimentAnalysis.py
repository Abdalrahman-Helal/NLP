# ============================================================================
# TEXT PREPROCESSING AND MACHINE LEARNING PIPELINE
# ============================================================================

# ALL IMPORTS
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from gensim.models import Word2Vec

# DOWNLOAD REQUIRED NLTK DATA
nltk.download('movie_reviews', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


# ============================================================================
# STEP 1: DATA SELECTION
# ============================================================================
print("------- 1. DATA SELECTION -------")
# data (500 reviews each)
pos_files = movie_reviews.fileids('pos')[:500]
neg_files = movie_reviews.fileids('neg')[:500]

sample_files = pos_files + neg_files
sample_text = movie_reviews.raw(sample_files)

print(sample_text[:500] + "...\n")

# ============================================================================
# STEP 2: SEGMENTATION
# ============================================================================
print("------- 2. SEGMENTATION -------")
sentences = nltk.sent_tokenize(sample_text)
first_sentence = sentences[0]
print(first_sentence + "\n")


# ============================================================================
# STEP 3: TOKENIZATION
# ============================================================================
print("------- 3. TOKENIZATION -------")
tokens = nltk.word_tokenize(sample_text)
print(tokens[:20], "\n")


# ============================================================================
# STEP 4: REMOVE STOPWORDS, PUNCTUATION & NUMBERS
# ============================================================================
print("------- 4. REMOVE STOPWORDS, PUNCTUATION & NUMBERS -------")
stopwords_list = stopwords.words('english')

filtered_text = []
for token in sample_text.lower().split():
    if token not in stopwords_list:
        # Remove punctuation and numbers
        word = re.sub(r'[!"#\$%&\'\(\)\*\+,-\./:;<=>\?@[\\\]\^_`\{|\}~]+', '', token)
        word = re.sub(r'\d+', '', word)
        if word:
            filtered_text.append(word)

print(filtered_text[:20], "\n")


# ============================================================================
# STEP 5: LEMMATIZATION
# ============================================================================
print("------- 5. LEMMATIZATION -------")
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in filtered_text]

print(lemmatized_words[:20])

processed_text = ' '.join(lemmatized_words)
print("\n------- FINAL PROCESSED TEXT -------")
print(processed_text[:200] + "...\n\n")


# ============================================================================
# STEP 6: N-GRAM PROBABILITY 
# ============================================================================
print("\n\n======= 6. BIGRAM PROBABILITY =======\n")

# N-gram function 
def extract_gram(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# Build bigram and unigram counts from training data
bigrams = extract_gram(lemmatized_words, 2)
bigram_counts = Counter(bigrams)
unigram_counts = Counter(lemmatized_words)

print(f"Total words in dataset: {len(lemmatized_words)}")
print(f"Total bigrams: {len(bigrams)}\n")

print("------- ANALYZING 10 SENTENCES -------\n")

for i, sentence in enumerate(sentences[:10], 1):
    # Process sentence
    sent_tokens = []
    for token in sentence.lower().split():
        if token not in stopwords_list:
            word = re.sub(r'[^\w]', '', token)
            if word:
                sent_tokens.append(lemmatizer.lemmatize(word))
    
    # Take first 5 words only
    first_5_words = sent_tokens[:5]
    
    if len(first_5_words) < 1:
        print(f"Sentence {i}: (no words after processing)\n")
        continue
    
    print(f"Sentence {i}: {sentence}")
    print(f"First 5 words after processing: {first_5_words}\n")
    
    # Calculate unigram probability for each word
    print("Unigram Probabilities:")
    total_words = len(lemmatized_words)
    for word in first_5_words:
        word_count = unigram_counts[word]
        unigram_prob = word_count / total_words
        print(f"  P({word}) = {word_count}/{total_words} = {unigram_prob:.6f}")
    
    # Find top 5 bigrams for these words
    print("\nTop 5 Bigrams for each word:")
    for word in first_5_words:
        # Get all bigrams starting with this word
        word_bigrams = [(bg, count) for bg, count in bigram_counts.items() if bg[0] == word]
        
        if not word_bigrams:
            print(f"  '{word}' -> No bigrams found")
            continue
        
        # Sort by count and get top 5
        top_5_bigrams = sorted(word_bigrams, key=lambda x: x[1], reverse=True)[:5]
        
        print(f"  '{word}' ->")
        for bg, count in top_5_bigrams:
            word1, word2 = bg[0], bg[1]
            unigram_count = unigram_counts[word1]
            bg_prob = count / unigram_count
            print(f"    {bg}: P({word2}|{word1}) = {count}/{unigram_count} = {bg_prob:.6f}")
    
    print("=" * 80 + "\n")

# ============================================================================
# STEP 7: FEATURE EXTRACTION
# ============================================================================
print("======= 7. FEATURE EXTRACTION =======")

# Treat the whole processed text as one document
# documents = [processed_text]

# Split processed text into 3 documents (by dividing into thirds)
words = processed_text.split()
chunk_size = len(words) // 3

doc1 = ' '.join(words[:chunk_size])
doc2 = ' '.join(words[chunk_size:2*chunk_size])
doc3 = ' '.join(words[2*chunk_size:])

documents = [doc1, doc2, doc3]
print(f"Split into {len(documents)} documents\n")

# 1) BINARY ENCODING
print("\n------- 1. BINARY ENCODING -------")
binary_vectorizer = CountVectorizer(binary=True)
binary_features = binary_vectorizer.fit_transform(documents)

print("Binary Feature Names (first 20):")
print(binary_vectorizer.get_feature_names_out()[:20])
print("\nBinary Feature Vector (first 20 values):")
print(binary_features.toarray()[0][:20])

# 2) COUNT VECTOR (BAG OF WORDS)
print("\n------- 2. COUNT ENCODING (BAG OF WORDS) -------")
count_vectorizer = CountVectorizer()
count_features = count_vectorizer.fit_transform(documents)

print("Count Feature Names (first 20):")
print(count_vectorizer.get_feature_names_out()[:20])
print("\nCount Feature Vector (first 20 values):")
print(count_features.toarray()[0][:20])

# 3) TF-IDF
print("\n------- 3. TF-IDF -------")
tfidf_vectorizer = TfidfVectorizer()
tfidf_features = tfidf_vectorizer.fit_transform(documents)

print("TF-IDF Feature Names (first 20):")
print(tfidf_vectorizer.get_feature_names_out()[:20])
print("\nTF-IDF Feature Vector (first 20 values):")
print(tfidf_features.toarray()[0][:20])

# Summary
print("\n------- FEATURE VECTOR SHAPES -------")
print(f"Binary Encoding: {binary_features.shape}")
print(f"Count Encoding : {count_features.shape}")
print(f"TF-IDF         : {tfidf_features.shape}\n\n")


# ============================================================================
# STEP 8: MACHINE LEARNING (SENTIMENT CLASSIFICATION)
# ============================================================================
print("------- 8. MACHINE LEARNING TASK (NAIVE BAYES) -------")

# Prepare data with labels
X_data = []
y_data = []

for f in pos_files:
    X_data.append(movie_reviews.raw(f))
    y_data.append(1)  # Positive

for f in neg_files:
    X_data.append(movie_reviews.raw(f))
    y_data.append(0)  # Negative

# Vectorize using Count Vectorizer
vectorizer = CountVectorizer()
X_vectors = vectorizer.fit_transform(X_data)

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectors, y_data, test_size=0.2, random_state=42
)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"Model Accuracy: {accuracy:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

