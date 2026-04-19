"""
Multinomial Naïve Bayes spam classifier built from scratch (no scikit-learn).
Structured to align with mathematical formulation and academic reporting requirements.
"""

from __future__ import annotations

import html
import math
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ===========================================================================
# CODE BLOCK 1: Preprocessing & Bag-of-Words Representation
# ===========================================================================
# Purpose: To transform raw, unstructured email text into a standardized 
#          computational structure by removing noise (HTML, special characters, 
#          stop words) and tokenizing the remaining text.
#
# Mathematical Operation: Represents a document as a sparse multiset (vector) 
#          x = {x_1, x_2, ..., x_k} where x_i represents the frequency count 
#          of the word w_i occurring in the specific document.
#
# Expected Result: A sparse dictionary mapping unique string tokens (keys) to 
#          their integer frequency counts (values) for a given email.
# ===========================================================================

DEFAULT_STOP_WORDS: frozenset[str] = frozenset(
    {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", 
     "as", "by", "with", "from", "is", "are", "was", "were", "be", "been", 
     "being", "have", "has", "had", "do", "does", "did", "will", "would", 
     "could", "should", "may", "might", "must", "can", "this", "that", "these", 
     "those", "it", "its", "i", "you", "he", "she", "we", "they", "me", "him", 
     "her", "us", "them", "my", "your", "his", "our", "their", "what", "which", 
     "who", "whom", "when", "where", "why", "how", "all", "each", "every", 
     "both", "few", "more", "most", "other", "some", "such", "no", "nor", 
     "not", "only", "own", "same", "so", "than", "too", "very", "just", "also", 
     "about", "into", "through", "during", "before", "after", "above", "below", 
     "between", "under", "again", "once", "here", "there", "any", "if", 
     "because", "until", "while", "against", "out", "off", "over", "then", "now"}
)

class TextPreprocessor:
    _tag_pattern = re.compile(r"<[^>]+>")
    _non_word_pattern = re.compile(r"[^a-z0-9\s]+")

    def __init__(self, stop_words: Optional[Iterable[str]] = None) -> None:
        self._stop_words = (
            frozenset(w.lower() for w in stop_words)
            if stop_words is not None
            else DEFAULT_STOP_WORDS
        )

    def preprocess(self, raw_text: str) -> List[str]:
        text = html.unescape(raw_text)
        text = self._tag_pattern.sub(" ", text)
        text = text.lower()
        text = self._non_word_pattern.sub(" ", text)
        tokens = [t for t in text.split() if t and t not in self._stop_words]
        return tokens

def bag_of_words(tokens: Sequence[str]) -> Dict[str, int]:
    bow: Dict[str, int] = {}
    for token in tokens:
        bow[token] = bow.get(token, 0) + 1
    return bow


# ===========================================================================
# CODE BLOCK 2: Model Training and Parameter Estimation
# ===========================================================================
# Purpose: To calculate the prior probabilities for each class and the 
#          conditional probabilities of words given a class using the 
#          training dataset. Laplacian smoothing is applied to prevent 
#          zero-probability errors for unseen words.
#
# Mathematical Operation: 
#          1. Prior: P(C) = N_C / N_total
#          2. Smoothed Likelihood: P(w_i|C) = (N_{w_i|C} + 1) / (N_C + |V|)
#          3. Transformation: Logarithm is applied to both to prevent underflow.
#
# Expected Result: A 'TrainedNaiveBayesModel' data structure storing the 
#          calculated log-priors and smoothed log-likelihoods for the entire 
#          training vocabulary.
# ===========================================================================

@dataclass
class TrainedNaiveBayesModel:
    log_prior: Dict[str, float]
    log_conditional: Dict[str, Dict[str, float]]
    vocabulary_size: int
    class_token_totals: Dict[str, int]
    classes: Tuple[str, ...]

class MultinomialNaiveBayesClassifier:
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None) -> None:
        self._preprocessor = preprocessor or TextPreprocessor()
        self._model: Optional[TrainedNaiveBayesModel] = None

    @property
    def model(self) -> TrainedNaiveBayesModel:
        if self._model is None:
            raise RuntimeError("Classifier is not trained; call fit() first.")
        return self._model

    def fit(self, documents: Sequence[str], labels: Sequence[str]) -> None:
        if len(documents) != len(labels):
            raise ValueError("documents and labels must have the same length.")

        class_docs: Dict[str, int] = {}
        word_counts_per_class: Dict[str, Dict[str, int]] = {}
        vocabulary: set[str] = set()

        # Aggregate counts
        for raw, label in zip(documents, labels):
            label_norm = label.lower().strip()
            class_docs[label_norm] = class_docs.get(label_norm, 0) + 1
            tokens = self._preprocessor.preprocess(raw)
            bow = bag_of_words(tokens)
            
            if label_norm not in word_counts_per_class:
                word_counts_per_class[label_norm] = {}
            wc = word_counts_per_class[label_norm]
            for word, count in bow.items():
                vocabulary.add(word)
                wc[word] = wc.get(word, 0) + count

        n_docs = len(documents)
        v_size = len(vocabulary)
        classes = tuple(sorted(class_docs.keys()))

        # Calculate Log Priors
        log_prior: Dict[str, float] = {}
        for c in classes:
            log_prior[c] = math.log(class_docs[c] / n_docs)

        class_token_totals: Dict[str, int] = {
            c: sum(word_counts_per_class.get(c, {}).values()) for c in classes
        }

        # Calculate Smoothed Log Likelihoods
        log_conditional: Dict[str, Dict[str, float]] = {c: {} for c in classes}
        for c in classes:
            n_c = class_token_totals[c]
            denom_log = math.log(n_c + v_size) 
            for word, cnt in word_counts_per_class.get(c, {}).items():
                log_conditional[c][word] = math.log(cnt + 1) - denom_log

        self._model = TrainedNaiveBayesModel(
            log_prior=log_prior,
            log_conditional=log_conditional,
            vocabulary_size=v_size,
            class_token_totals=class_token_totals,
            classes=classes,
        )


# ===========================================================================
# CODE BLOCK 3: Prediction and Decision Rule
# ===========================================================================
# Purpose: To classify a new, unseen email by computing the posterior 
#          probability for both the "Spam" and "Ham" classes, returning the 
#          label that yields the highest score.
#
# Mathematical Operation: 
#          C_hat = argmax_{C} [ log P(C) + sum( log P(w_i | C) ) ]
#
# Expected Result: A discrete string label ("Spam" or "Ham") assigning the 
#          email to a class based on the maximum unnormalized log-posterior.
# ===========================================================================

    def _log_prob_word_given_class(self, word: str, cls: str) -> float:
        m = self.model
        n_c = m.class_token_totals[cls]
        v = m.vocabulary_size
        inner = m.log_conditional.get(cls, {})
        
        if word in inner:
            return inner[word]
        # Smooth unseen words
        return math.log(1.0) - math.log(n_c + v)

    def predict_log_scores(self, raw_text: str) -> Dict[str, float]:
        m = self.model
        bow = bag_of_words(self._preprocessor.preprocess(raw_text))
        scores: Dict[str, float] = {}
        
        for cls in m.classes:
            total = m.log_prior[cls]
            for word, freq in bow.items():
                total += freq * self._log_prob_word_given_class(word, cls)
            scores[cls] = total
        return scores

    def predict(self, raw_text: str) -> str:
        scores = self.predict_log_scores(raw_text)
        return max(scores.items(), key=lambda kv: kv[1])[0]


# ===========================================================================
# CODE BLOCK 4: Testing Plan and Evaluation Metrics
# ===========================================================================
# Purpose: To rigorously evaluate the classifier's real-world viability by 
#          testing it against unseen data (20% hold-out), generating a 
#          confusion matrix, and calculating standardized performance metrics.
#
# Mathematical Operation: 
#          Accuracy = (TP + TN) / (TP + TN + FP + FN)
#          Precision = TP / (TP + FP)
#          Recall = TP / (TP + FN)
#          F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
#
# Expected Result: A summary dictionary and printout containing the confusion 
#          matrix counts alongside the calculated Accuracy, Precision, Recall, 
#          and F1 Score of the model.
# ===========================================================================

def train_test_split(
    documents: Sequence[str], labels: Sequence[str], test_fraction: float = 0.2, seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    paired = list(zip(documents, labels))
    rng = random.Random(seed)
    rng.shuffle(paired)
    
    n_test = max(1, int(round(len(paired) * test_fraction)))
    test_pairs = paired[:n_test]
    train_pairs = paired[n_test:]
    
    train_docs, train_labels = zip(*train_pairs) if train_pairs else ([], [])
    test_docs, test_labels = zip(*test_pairs) if test_pairs else ([], [])
    
    return list(train_docs), list(train_labels), list(test_docs), list(test_labels)

@dataclass
class ConfusionMatrix:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    def update(self, y_true: str, y_pred: str, positive_label: str = "spam") -> None:
        y_true_n, y_pred_n, pos = y_true.lower().strip(), y_pred.lower().strip(), positive_label.lower().strip()
        is_actual_pos = (y_true_n == pos)
        is_pred_pos = (y_pred_n == pos)

        if is_pred_pos and is_actual_pos: self.true_positive += 1
        elif is_pred_pos and not is_actual_pos: self.false_positive += 1
        elif not is_pred_pos and not is_actual_pos: self.true_negative += 1
        else: self.false_negative += 1

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: ConfusionMatrix

    def summary(self) -> str:
        c = self.confusion
        return (f"Accuracy:  {self.accuracy:.4f}\nPrecision: {self.precision:.4f}\n"
                f"Recall:    {self.recall:.4f}\nF1 Score:  {self.f1:.4f}\n"
                f"Confusion: TP={c.true_positive} FP={c.false_positive} TN={c.true_negative} FN={c.false_negative}")

def evaluate_classifier(clf: MultinomialNaiveBayesClassifier, test_docs: Sequence[str], test_labels: Sequence[str], positive_label: str = "spam") -> ClassificationMetrics:
    cm = ConfusionMatrix()
    correct = 0
    
    for doc, true_label in zip(test_docs, test_labels):
        pred = clf.predict(doc)
        if pred.lower().strip() == true_label.lower().strip(): correct += 1
        cm.update(true_label, pred, positive_label=positive_label)

    total = cm.true_positive + cm.true_negative + cm.false_positive + cm.false_negative
    accuracy = correct / total if total else 0.0

    denom_p = cm.true_positive + cm.false_positive
    precision = cm.true_positive / denom_p if denom_p else 0.0

    denom_r = cm.true_positive + cm.false_negative
    recall = (cm.true_positive / denom_r) if denom_r != 0 else (1.0 if cm.false_positive == 0 else 0.0)

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) != 0 else 0.0

    return ClassificationMetrics(accuracy=accuracy, precision=precision, recall=recall, f1=f1, confusion=cm)

# ===========================================================================
# Execution / Testing Wrapper
# ===========================================================================
def load_spamassassin() -> Tuple[List[str], List[str]]:
    from datasets import load_dataset
    ds = load_dataset("talby/spamassassin", "text", split="train")
    return [row["text"] for row in ds], ["spam" if row["label"] == 0 else "ham" for row in ds]

def main() -> None:
    documents, labels = load_spamassassin()
    train_docs, train_labels, test_docs, test_labels = train_test_split(documents, labels, test_fraction=0.2, seed=7)

    clf = MultinomialNaiveBayesClassifier()
    clf.fit(train_docs, train_labels)

    metrics = evaluate_classifier(clf, test_docs, test_labels)
    print("=== Naïve Bayes Spam Classifier (from scratch) ===\n")
    print(metrics.summary())

if __name__ == "__main__":
    main()