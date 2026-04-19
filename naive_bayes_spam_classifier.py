"""
Multinomial Naïve Bayes spam classifier built from scratch (no scikit-learn).

Mathematical backbone (Bayes' theorem for document classification):
    P(Spam | document) = P(document | Spam) * P(Spam) / P(document)

We use the Naïve Bayes *conditional independence* assumption: given the class,
word occurrences are independent, so

    P(document | class) ≈ ∏_w P(w | class)^{count(w, document)}

Taking logarithms turns the product into a sum (avoids floating-point underflow):

    log P(class | document) ∝ log P(class) + Σ_w count(w, doc) * log P(w | class)

We predict the class with the **argmax** of that score (denominator P(document)
is the same for both classes, so it cancels in the comparison).

Laplace (add-one) smoothing for conditional word probabilities:

    P(w | c) = (count(w, c) + 1) / (N_c + |V|)

where N_c is the total token count in class c and |V| is the global vocabulary
size (unique words across the corpus). The +1 in the numerator and +|V| in the
denominator ensure no estimated probability is exactly zero.
"""

from __future__ import annotations

import html
import math
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 1. Preprocessing & bag-of-words
# ---------------------------------------------------------------------------


DEFAULT_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "as",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "once",
        "here",
        "there",
        "any",
        "if",
        "because",
        "until",
        "while",
        "against",
        "out",
        "off",
        "over",
        "then",
        "now",
    }
)


class TextPreprocessor:
    """
    Turn raw email strings into token sequences suitable for a bag-of-words model.

    Steps align with typical text-mining pipelines: decode entities, strip HTML,
    normalize case, remove non-alphanumeric boundaries, drop stop words.
    """

    _tag_pattern = re.compile(r"<[^>]+>")
    _non_word_pattern = re.compile(r"[^a-z0-9\s]+")

    def __init__(self, stop_words: Optional[Iterable[str]] = None) -> None:
        self._stop_words = (
            frozenset(w.lower() for w in stop_words)
            if stop_words is not None
            else DEFAULT_STOP_WORDS
        )

    def preprocess(self, raw_text: str) -> List[str]:
        """
        Preprocess raw text and return a list of tokens (features).

        The multiset of these tokens is the bag-of-words representation; we store
        it sparsely as a dict word -> count via ``bag_of_words``.
        """
        # Unescape HTML entities (e.g. &amp;) before stripping tags
        text = html.unescape(raw_text)
        text = self._tag_pattern.sub(" ", text)
        text = text.lower()
        text = self._non_word_pattern.sub(" ", text)
        tokens = [t for t in text.split() if t and t not in self._stop_words]
        return tokens


def bag_of_words(tokens: Sequence[str]) -> Dict[str, int]:
    """
    Build a **sparse** bag-of-words: keys are unique words, values are counts.

    This is a multiset / frequency map; word order and grammar are discarded,
    which matches the Naïve Bayes generative story for text.
    """
    bow: Dict[str, int] = {}
    for token in tokens:
        bow[token] = bow.get(token, 0) + 1
    return bow


# ---------------------------------------------------------------------------
# 2. Multinomial Naïve Bayes (training & inference)
# ---------------------------------------------------------------------------


@dataclass
class TrainedNaiveBayesModel:
    """
    Estimated parameters after training.

    * ``log_prior``: log P(class) — class priors from training label frequencies.
    * ``log_conditional``: log P(word | class) with Laplace smoothing.
    * ``vocabulary_size``: |V|, shared denominator term in smoothing.
    * ``class_token_totals``: N_c, total tokens observed per class.
    """

    log_prior: Dict[str, float]
    log_conditional: Dict[str, Dict[str, float]]
    vocabulary_size: int
    class_token_totals: Dict[str, int]
    classes: Tuple[str, ...]


class MultinomialNaiveBayesClassifier:
    """
    Multinomial Naïve Bayes for discrete count features (word frequencies).

    Inference uses log-scores proportional to the posterior:

        score(c) = log P(c) + Σ_w f(w) * log P(w | c)

    where f(w) is the count of w in the document. The predicted class is
    ``argmax_c score(c)`` — maximum *unnormalized* log-posterior (equivalent to
    comparing true posteriors since the evidence log P(doc) is class-constant).
    """

    def __init__(self, preprocessor: Optional[TextPreprocessor] = None) -> None:
        self._preprocessor = preprocessor or TextPreprocessor()
        self._model: Optional[TrainedNaiveBayesModel] = None

    @property
    def model(self) -> TrainedNaiveBayesModel:
        if self._model is None:
            raise RuntimeError("Classifier is not trained; call fit() first.")
        return self._model

    def fit(self, documents: Sequence[str], labels: Sequence[str]) -> None:
        """
        Estimate priors P(c) and smoothed likelihoods P(w|c) from labeled email bodies.

        Priors:
            P(c) = (# documents of class c) / (# documents)

        Word counts per class feed Laplace-smoothed multinomial estimates:

            P(w|c) = (count(w, c) + 1) / (N_c + |V|)

        We store log P(c) and log P(w|c) for numerical stability.
        """
        if len(documents) != len(labels):
            raise ValueError("documents and labels must have the same length.")

        class_docs: Dict[str, int] = {}
        word_counts_per_class: Dict[str, Dict[str, int]] = {}
        vocabulary: set[str] = set()

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

        log_prior: Dict[str, float] = {}
        for c in classes:
            # P(c) from empirical document frequencies (MLE prior)
            log_prior[c] = math.log(class_docs[c] / n_docs)

        class_token_totals: Dict[str, int] = {
            c: sum(word_counts_per_class.get(c, {}).values()) for c in classes
        }

        # log P(w|c) for every (c, w) that appeared; unseen (c,w) use smoothed default
        log_conditional: Dict[str, Dict[str, float]] = {c: {} for c in classes}
        for c in classes:
            n_c = class_token_totals[c]
            denom_log = math.log(n_c + v_size)  # log(N_c + |V|)
            for word, cnt in word_counts_per_class.get(c, {}).items():
                # Laplace: (count + 1) / (N_c + |V|)
                log_conditional[c][word] = math.log(cnt + 1) - denom_log

        self._model = TrainedNaiveBayesModel(
            log_prior=log_prior,
            log_conditional=log_conditional,
            vocabulary_size=v_size,
            class_token_totals=class_token_totals,
            classes=classes,
        )

    def _log_prob_word_given_class(self, word: str, cls: str) -> float:
        """
        Return log P(word | cls) under Laplace smoothing, including *unseen* words.

        If the word never appeared in class cls during training, count(w, cls)=0:

            P(w|cls) = (0 + 1) / (N_cls + |V|)

        which is the same closed form as seen words with zero count.
        """
        m = self.model
        n_c = m.class_token_totals[cls]
        v = m.vocabulary_size
        cnt = 0
        inner = m.log_conditional.get(cls, {})
        if word in inner:
            # Reconstruct smoothed count from stored log prob: exp(log)= (cnt+1)/(N_c+V)
            return inner[word]
        return math.log(1.0) - math.log(n_c + v)

    def predict_log_scores(self, raw_text: str) -> Dict[str, float]:
        """
        Compute unnormalized log-scores for each class (proportional to log-posterior).

        log P(cls | doc) ∝ log P(cls) + Σ_w f(w) * log P(w | cls)
        """
        m = self.model
        bow = bag_of_words(self._preprocessor.preprocess(raw_text))
        scores: Dict[str, float] = {}
        for cls in m.classes:
            # Prior contributes additively in log-space (log of product term P(c))
            total = m.log_prior[cls]
            for word, freq in bow.items():
                # Σ count(w) * log P(w|c) — multinomial sufficient statistics
                total += freq * self._log_prob_word_given_class(word, cls)
            scores[cls] = total
        return scores

    def predict(self, raw_text: str) -> str:
        """Argmax decision rule over log-scores."""
        scores = self.predict_log_scores(raw_text)
        return max(scores.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# 3. Train/test split, confusion matrix, metrics
# ---------------------------------------------------------------------------


def train_test_split(
    documents: Sequence[str],
    labels: Sequence[str],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Random 80/20 (by default) split to mimic hold-out evaluation.

    Stratification is not required by the brief; simple shuffle split is enough
    for the mock framework.
    """
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
    return (
        list(train_docs),
        list(train_labels),
        list(test_docs),
        list(test_labels),
    )


@dataclass
class ConfusionMatrix:
    """
    Binary confusion counts with **Spam as the positive** class.

    * TP: predicted spam, true spam
    * FP: predicted spam, true ham (false alarm)
    * TN: predicted ham, true ham
    * FN: predicted ham, true spam (missed spam)
    """

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    def update(self, y_true: str, y_pred: str, positive_label: str = "spam") -> None:
        y_true_n = y_true.lower().strip()
        y_pred_n = y_pred.lower().strip()
        pos = positive_label.lower().strip()
        is_actual_positive = y_true_n == pos
        is_predicted_positive = y_pred_n == pos

        if is_predicted_positive and is_actual_positive:
            self.true_positive += 1
        elif is_predicted_positive and not is_actual_positive:
            self.false_positive += 1
        elif not is_predicted_positive and not is_actual_positive:
            self.true_negative += 1
        else:
            self.false_negative += 1

    def as_dict(self) -> Dict[str, int]:
        return {
            "TP": self.true_positive,
            "FP": self.false_positive,
            "TN": self.true_negative,
            "FN": self.false_negative,
        }


@dataclass
class ClassificationMetrics:
    """Standard binary metrics (positive class = spam)."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: ConfusionMatrix

    def summary(self) -> str:
        c = self.confusion
        return (
            f"Accuracy:  {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall:    {self.recall:.4f}\n"
            f"F1 Score:  {self.f1:.4f}\n"
            f"Confusion: TP={c.true_positive} FP={c.false_positive} "
            f"TN={c.true_negative} FN={c.false_negative}"
        )


def evaluate_classifier(
    classifier: MultinomialNaiveBayesClassifier,
    test_documents: Sequence[str],
    test_labels: Sequence[str],
    positive_label: str = "spam",
) -> ClassificationMetrics:
    """
    Populate the confusion matrix and compute accuracy, precision, recall, F1.

    Definitions (spam = positive):
        Accuracy  = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)   — if no spam predicted, define 0.0
        Recall    = TP / (TP + FN)   — if no actual spam, define 1.0 if FP=0 else 0.0
        F1        = 2 * P * R / (P + R)
    """
    cm = ConfusionMatrix()
    correct = 0
    for doc, true_label in zip(test_documents, test_labels):
        pred = classifier.predict(doc)
        if pred.lower().strip() == true_label.lower().strip():
            correct += 1
        cm.update(true_label, pred, positive_label=positive_label)

    total = cm.true_positive + cm.true_negative + cm.false_positive + cm.false_negative
    accuracy = correct / total if total else 0.0

    denom_p = cm.true_positive + cm.false_positive
    precision = cm.true_positive / denom_p if denom_p else 0.0

    denom_r = cm.true_positive + cm.false_negative
    if denom_r == 0:
        recall = 1.0 if cm.false_positive == 0 else 0.0
    else:
        recall = cm.true_positive / denom_r

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion=cm,
    )


# ---------------------------------------------------------------------------
# Sample dataset & demo
# ---------------------------------------------------------------------------

SAMPLE_RAW_EMAILS: List[str] = [
    # Ham — personal / work tone
    "<p>Hi Alice, meeting moved to <b>3pm</b> tomorrow. Thanks!</p>",
    "Please review the attached Q3 report and send feedback by Friday.",
    "Lunch at the cafe near the office? Let me know if you're free.",
    "Your invoice #1042 has been paid. Thank you for your business.",
    "Reminder: dentist appointment next Tuesday at 10am.",
    # Spam — promotional / scam patterns
    "<html><body>Congratulations!!! You WON $1,000,000!!! Click HERE now!!!</body></html>",
    "FREE!!! Viagra/Cialis cheap meds!!! 100% guarantee!!! Act NOW!!!",
    "You have been selected for a cash prize. Wire transfer fee required today.",
    "URGENT: Your account will be suspended. Verify password immediately!!!",
    "Lose weight fast with this one weird trick — guaranteed results!!!",
    "Hot singles in your area want to chat — click this link now FREE!!!",
]

SAMPLE_LABELS: List[str] = [
    "ham",
    "ham",
    "ham",
    "ham",
    "ham",
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
]


def load_spamassassin() -> Tuple[List[str], List[str]]:
    from datasets import load_dataset
    ds = load_dataset("talby/spamassassin", "text", split="train")
    documents = [row["text"] for row in ds]
    
    labels = ["spam" if row["label"] == 0 else "ham" for row in ds]
    
    return documents, labels

def main() -> None:
    """Run 80/20 evaluation on the SpamAssassin corpus and print metrics."""
    documents, labels = load_spamassassin()
    train_docs, train_labels, test_docs, test_labels = train_test_split(
        documents,
        labels,
        test_fraction=0.2,
        seed=7,
    )

    clf = MultinomialNaiveBayesClassifier()
    clf.fit(train_docs, train_labels)

    metrics = evaluate_classifier(clf, test_docs, test_labels)
    print("=== Naïve Bayes Spam Classifier (from scratch) — hold-out evaluation ===\n")
    print(f"Train size: {len(train_docs)}, Test size: {len(test_docs)}\n")
    print(metrics.summary())
    print()

    # Optional: show log-scores for one test example (Bayes decision trace)
    if test_docs:
        example = test_docs[0]
        scores = clf.predict_log_scores(example)
        print("Example log-scores (proportional to log-posterior):")
        for cls, s in sorted(scores.items()):
            print(f"  {cls}: {s:.4f}")
        print(f"Predicted: {clf.predict(example)} | True: {test_labels[0]}")


if __name__ == "__main__":
    main()
