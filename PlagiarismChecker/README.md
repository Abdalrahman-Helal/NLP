# Plagiarism Checker

An NLP college project that detects plagiarism between text pairs using a feed-forward neural network.

## What I built

A full ML pipeline that takes a CSV of text pairs and classifies each as plagiarized or clean:

- **Preprocessing** — lowercasing, cleaning, tokenization, and lemmatization using NLTK
- **Feature extraction** — TF-IDF word n-grams (1–2) and char n-grams (3–5) combined with handcrafted similarity features: cosine similarity, Jaccard, overlap coefficient, and length ratio
- **Model** — scikit-learn `MLPClassifier` with two hidden layers (32, 16), ReLU activation, L2 regularization, and early stopping
- **Evaluation** — accuracy, precision, recall, F1, confusion matrix, and a validation-tuned decision threshold to maximize F1 instead of using a fixed 0.5 cutoff

## Stack

Python · scikit-learn · NLTK · pandas · scipy
