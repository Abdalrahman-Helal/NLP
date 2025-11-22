import nltk
import re
from collections import Counter
from nltk.util import ngrams

# Download data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer

# STEP 1: DATA SELECTION
print("------- 1. DATA SELECTION -------")
pos_files = movie_reviews.fileids('pos')[:50]
neg_files = movie_reviews.fileids('neg')[:50]

sample_files = pos_files + neg_files
sample_text = movie_reviews.raw(sample_files)

print(sample_text[:500] + "...")

# STEP 2: SEGMENTATION
print("\n------- 2. SEGMENTATION -------")
sentences = nltk.sent_tokenize(sample_text)
first_sentence = sentences[0]
print(first_sentence)

# STEP 3: TOKENIZATION
print("\n------- 3. TOKENIZATION -------")
tokens = nltk.word_tokenize(sample_text)
print(tokens[:20])

# STEP 4: REMOVE STOPWORDS, PUNCTUATION, NUMBERS
print("\n------- 4. REMOVE STOPWORDS, PUNCTUATION & NUMBERS -------")
stopwords_list = stopwords.words('english')
txt = sample_text.lower()

filtered_text = []
for token in txt.split():
    if token not in stopwords_list:
        word = re.sub(r'[!"#\$%&\'\(\)\*\+,-\./:;<=>\?@[\\\]\^_`\{|\}~]+','',token)
        word = re.sub(r'\d+', '', word)
        if word != '':
            filtered_text.append(word)
print(filtered_text[:20])

# STEP 5: LEMMATIZATION
print("\n------- 5. LEMMATIZATION -------")
lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for word in filtered_text:
    lemma = lemmatizer.lemmatize(word, pos='v')
    lemmatized_words.append(lemma)
print(lemmatized_words[:20])

processed_text = ' '.join(lemmatized_words)
print("\n------- FINAL PROCESSED TEXT -------")
print(processed_text[:200] + "...")

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