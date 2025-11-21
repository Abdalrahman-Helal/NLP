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

#N-gram function 
def extract_gram(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


bigrams = extract_gram(lemmatized_words, 2)
bigram_counts = Counter(bigrams)
unigram_counts = Counter(lemmatized_words)

print(f"Total words in dataset: {len(lemmatized_words)}")
print(f"Total bigrams: {len(bigrams)}")
print(f"Most common bigrams: {bigram_counts.most_common(5)}\n")

# Select 10 sentences from the original dataset
print("------- CALCULATING PROBABILITY FOR 10 SENTENCES -------\n")

selected_sentences = sentences[:10]

for i, sentence in enumerate(selected_sentences, 1):
    
    sent_tokens = []
    for token in sentence.lower().split():
        if token not in stopwords_list:
            word = re.sub(r'[^\w]', '', token)
            if word:
                sent_tokens.append(lemmatizer.lemmatize(word))
    
    if len(sent_tokens) < 2:
        print(f"Sentence {i}: (too short after processing)\n")
        continue
    
    print(f"Sentence {i}: {sentence}")
    print(f"Processed: {' '.join(sent_tokens)}")
    
    # Extract bigrams from sentence
    sent_bigrams = extract_gram(sent_tokens, 2)
    print(f"Bigrams: {sent_bigrams}")
    
    # Calculate probability by multiplying all bigram probabilities
    probability = 1.0
    print("Calculations:")
    
    for bg in sent_bigrams:
        word1, word2 = bg[0], bg[1]
        bigram_count = bigram_counts[bg]
        unigram_count = unigram_counts[word1]
        
        if unigram_count == 0:
            probability = 0
            print(f"  P({word2}|{word1}) = 0 (word '{word1}' not in training)")
            break
        
        bg_prob = bigram_count / unigram_count
        probability *= bg_prob
        print(f"  P({word2}|{word1}) = Count({word1},{word2})/Count({word1}) = {bigram_count}/{unigram_count} = {bg_prob:.6f}")
    
    print(f"\nFinal Probability = {probability:.15e}")
    print("=" * 80 + "\n") 