"""
Plagiarism Checker — NLP College Project
=========================================
Pipeline:
  1. Data loading
  2. Preprocessing
  3. Feature extraction
  4. Modeling (MLP neural network)
  5. Evaluation

Run:
    python plagiarism_checker.py

Inference demo (uses saved models if they exist):
    Edit the texts at the bottom of this file and re-run.
"""

import re
import sys
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# ─────────────────────────────────────────────
# CONFIG  (edit these if needed)
# ─────────────────────────────────────────────
DATASET_PATH   = "data/pairs.csv"
MODELS_DIR     = Path("models")
RANDOM_STATE   = 42
TEST_SIZE      = 0.15
VAL_SIZE       = 0.15

# ─────────────────────────────────────────────
# STEP 1: DATA LOADING
# ─────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        sys.exit(f"[ERROR] Dataset not found: {csv_path}")
    df = pd.read_csv(path)
    missing = {"text1", "text2", "label"} - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns: {sorted(missing)}")
    df = df[["text1", "text2", "label"]].copy()
    df["text1"] = df["text1"].fillna("").astype(str)
    df["text2"] = df["text2"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df[df["label"].isin([0, 1])].reset_index(drop=True)


def split_dataset(df):
    train_val, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"]
    )
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    train_df, val_df = train_test_split(
        train_val, test_size=val_ratio, random_state=RANDOM_STATE, stratify=train_val["label"]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _label_stats(title: str, labels) -> None:
    labels = np.asarray(labels)
    n = len(labels)
    n1 = int((labels == 1).sum())
    n0 = int((labels == 0).sum())
    print(f"  {title}: total={n} | plagiarized={n1} ({n1/n*100:.1f}%) | clean={n0} ({n0/n*100:.1f}%)")


# ─────────────────────────────────────────────
# STEP 2: PREPROCESSING
# ─────────────────────────────────────────────

def _ensure_nltk() -> None:
    for res in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
        nltk.download(res, quiet=True)


_nltk_ready = False


def preprocess_text(text: str) -> str:
    global _nltk_ready
    if not _nltk_ready:
        _ensure_nltk()
        _nltk_ready = True

    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s.,;:!?'\"]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Cap length — keeps char TF-IDF fast on any laptop
    text = text[:4000]

    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(tok) for tok in tokens)


def preprocess_split(df):
    return df["text1"].map(preprocess_text), df["text2"].map(preprocess_text)


# ─────────────────────────────────────────────
# STEP 3: FEATURE EXTRACTION
# ─────────────────────────────────────────────

class PairFeatureBuilder:
    """TF-IDF (word + char n-grams) + cosine/Jaccard/overlap/length-ratio features."""

    def __init__(self):
        # Word n-grams: capture lexical overlap
        self.word_vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_df=0.95,
            max_features=3000, dtype=np.float32,
        )
        # Char n-grams: robust to small edits / obfuscation
        self.char_vec = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=3, max_df=0.95,
            max_features=2000, dtype=np.float32,
        )

    def fit_transform(self, t1: pd.Series, t2: pd.Series) -> csr_matrix:
        combined = pd.concat([t1, t2], ignore_index=True)
        self.word_vec.fit(combined)
        self.char_vec.fit(combined)
        return self._build(t1, t2)

    def transform(self, t1: pd.Series, t2: pd.Series) -> csr_matrix:
        return self._build(t1, t2)

    def _build(self, t1: pd.Series, t2: pd.Series) -> csr_matrix:
        w1 = self.word_vec.transform(t1)
        w2 = self.word_vec.transform(t2)
        c1 = self.char_vec.transform(t1)
        c2 = self.char_vec.transform(t2)
        sim = self._similarity(t1, t2, w1, w2)
        return hstack([w1, w2, c1, c2, sim], format="csr")

    @staticmethod
    def _similarity(t1, t2, w1, w2) -> csr_matrix:
        cos     = cosine_similarity(w1, w2).diagonal()
        jaccard = np.array([_jaccard(a, b) for a, b in zip(t1, t2)])
        overlap = np.array([_overlap(a, b)  for a, b in zip(t1, t2)])
        lenrat  = np.array([_len_ratio(a, b) for a, b in zip(t1, t2)])
        return csr_matrix(np.vstack([cos, jaccard, overlap, lenrat]).T)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def _overlap(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, min(len(sa), len(sb)))


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(1, max(la, lb))

# ─────────────────────────────────────────────
# STEP 4 & 5: MODELING + EVALUATION helpers
# ─────────────────────────────────────────────

def pick_best_threshold(y_true, y_prob) -> float:
    """Choose the threshold on the validation set that maximises F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.20, 0.81, 0.02):
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t


def print_eval(title: str, y_true, y_pred) -> dict:
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("  confusion_matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("  classification_report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    return metrics


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    # ── 1. Data loading ──────────────────────
    print("\n[1/5] Data loading")
    df = load_dataset(DATASET_PATH)
    train_df, val_df, test_df = split_dataset(df)
    print(f"  Loaded rows: {len(df)}")
    print(f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print("  Split method: stratified (keeps class balance across splits)")
    _label_stats("Overall", df["label"].values)
    _label_stats("Train",   train_df["label"].values)
    _label_stats("Val",     val_df["label"].values)
    _label_stats("Test",    test_df["label"].values)

    # ── 2. Preprocessing ─────────────────────
    print("\n[2/5] Preprocessing")
    print("  Steps: lowercase -> clean -> tokenize -> lemmatize (same for text1 & text2)")
    train_t1, train_t2 = preprocess_split(train_df)
    val_t1,   val_t2   = preprocess_split(val_df)
    test_t1,  test_t2  = preprocess_split(test_df)
    print("  Done.")

    # ── 3. Feature extraction ─────────────────
    print("\n[3/5] Feature extraction")
    print("  Features: TF-IDF word n-grams + char n-grams + cosine/Jaccard/overlap/length-ratio")
    fb = PairFeatureBuilder()
    x_train = fb.fit_transform(train_t1, train_t2)
    x_val   = fb.transform(val_t1,   val_t2)
    x_test  = fb.transform(test_t1,  test_t2)
    print(f"  Feature matrix shape: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    # ── 4. Modeling ───────────────────────────
    print("\n[4/5] Modeling (MLP training)")
    print("  Model: feed-forward neural network (MLP) — sklearn MLPClassifier")
    print("  Architecture: 2 hidden layers (32, 16 units) | relu | L2 reg | early stopping")
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=1e-3,
        batch_size=64,
        learning_rate_init=5e-4,
        max_iter=120,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    print("  Training done.")

    # ── 5. Evaluation ─────────────────────────
    print("\n[5/5] Evaluation")
    print("  Threshold tuning: searching validation set for best F1 threshold...")
    val_prob   = model.predict_proba(x_val)[:, 1]
    threshold  = pick_best_threshold(y_val, val_prob)
    print(f"  Best threshold: {threshold:.2f}")

    train_pred = (model.predict_proba(x_train)[:, 1] >= threshold).astype(int)
    val_pred   = (val_prob >= threshold).astype(int)
    test_pred  = (model.predict_proba(x_test)[:, 1] >= threshold).astype(int)

    train_m = print_eval("Train",      y_train, train_pred)
    val_m   = print_eval("Validation", y_val,   val_pred)
    _       = print_eval("Test",       y_test,  test_pred)

    gap = train_m["f1"] - val_m["f1"]
    if gap > 0.08:
        print(f"\n  [Warning] Possible overfitting — train F1 vs val F1 gap = {gap:.4f}")
    else:
        print(f"\n  Overfitting check: no clear signal (train/val F1 gap = {gap:.4f})")

    # ── Save artifacts ────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "mlp_model.joblib")
    joblib.dump(fb,    MODELS_DIR / "feature_builder.joblib")
    (MODELS_DIR / "decision_threshold.txt").write_text(f"{threshold:.4f}", encoding="utf-8")
    print(f"\n  Saved: {MODELS_DIR}/mlp_model.joblib")
    print(f"  Saved: {MODELS_DIR}/feature_builder.joblib")
    print(f"  Saved: {MODELS_DIR}/decision_threshold.txt")


# ─────────────────────────────────────────────
# INFERENCE DEMO  (edit texts below and re-run)
# ─────────────────────────────────────────────

def predict(text1: str, text2: str) -> dict:
    """Load saved model and predict whether a text pair is plagiarized."""
    model_path     = MODELS_DIR / "mlp_model.joblib"
    fb_path        = MODELS_DIR / "feature_builder.joblib"
    threshold_path = MODELS_DIR / "decision_threshold.txt"

    if not model_path.exists():
        return {"error": "No saved model found. Run main() first."}

    mdl = joblib.load(model_path)
    fb  = joblib.load(fb_path)
    thr = float(threshold_path.read_text(encoding="utf-8").strip()) if threshold_path.exists() else 0.5

    t1 = preprocess_text(text1)
    t2 = preprocess_text(text2)
    x  = fb.transform(pd.Series([t1]), pd.Series([t2]))
    prob = float(mdl.predict_proba(x)[0][1])
    label = "plagiarized" if prob >= thr else "not_plagiarized"
    return {"label": label, "probability": round(prob, 4), "threshold": round(thr, 4)}


# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()

    # ── Quick inference demo after training ──
    print("\n" + "=" * 50)
    print("INFERENCE DEMO")
    print("=" * 50)

    sample1 = "The quick brown fox jumps over the lazy dog near the river bank."
    sample2 = "A quick brown fox leaped over a lazy dog beside the river."

    result = predict(sample1, sample2)
    print(f"  text1   : {sample1}")
    print(f"  text2   : {sample2}")
    print(f"  verdict : {result['label']}")
    print(f"  prob    : {result['probability']:.4f}  (threshold={result['threshold']})")
