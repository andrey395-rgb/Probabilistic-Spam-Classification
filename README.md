# Probabilistic Spam Classification
### A Naïve Bayes Approach to Email Spam Detection

A spam email classifier built **from scratch** in Python — no scikit-learn or ML libraries. Implements the full Multinomial Naïve Bayes pipeline using Bayes' theorem, Laplacian smoothing, and log-likelihood scoring, trained on the SpamAssassin public corpus.

---

## Features

- Multinomial Naïve Bayes classifier implemented from scratch
- Full text preprocessing pipeline (HTML stripping, lowercasing, stop word removal, tokenization)
- Sparse bag-of-words representation
- Laplacian (add-one) smoothing to handle unseen words
- Log-space computation to prevent floating-point underflow
- Argmax decision rule over log-posteriors
- 80/20 train/test evaluation with confusion matrix, accuracy, precision, recall, and F1 score
- Desktop GUI (Tkinter) for pasting and classifying emails in real time
- Trained on the [SpamAssassin public corpus](https://huggingface.co/datasets/talby/spamassassin) (~10,749 emails)

---

## Project Structure

```
Probabilistic-Spam-Classification/
├── naive_bayes_spam_classifier.py   # Core classifier, preprocessing, metrics, dataset loader
└── spam_classifier_ui.py            # Tkinter desktop UI
```

---

## Requirements

- Python 3.11+
- `datasets==2.14.7` (specific version required for SpamAssassin corpus compatibility)
- `matplotlib`
- `seaborn`
- `numpy`

Install dependencies:

```bash
pip install datasets==2.14.7 matplotlib seaborn numpy

> **Note:** The SpamAssassin dataset on Hugging Face (`talby/spamassassin`) still uses an older loading script format. Versions of `datasets` newer than 2.14.7 have dropped support for this format. Downgrading to `2.14.7` is required until the dataset author migrates to Parquet.

---

## Usage

### Run the Desktop UI

```bash
python spam_classifier_ui.py
```

On first launch, the app will download and cache the SpamAssassin corpus (~10MB), then train the model automatically. This takes around 30–60 seconds. Subsequent launches load from cache and are faster.

Once open:
1. Paste any email body (plain text or HTML) into the input box
2. Click **Classify** or press `Ctrl+Enter`
3. The result shows **SPAM** (red) or **HAM** (green) along with the log-posterior scores for both classes

### Run the CLI Evaluation

```bash
python naive_bayes_spam_classifier.py
```

This trains on the SpamAssassin corpus, evaluates on an 80/20 hold-out split, and prints the confusion matrix and metrics to the terminal. CLI will pop up visual graphs.

---

## How It Works

### 1. Preprocessing

Raw email text is passed through `TextPreprocessor`, which:
- Decodes HTML entities
- Strips HTML tags
- Lowercases all text
- Removes non-alphanumeric characters
- Removes common stop words (e.g., "the", "is", "and")

The result is tokenized into a **sparse bag-of-words** dictionary (`word → count`), discarding word order and grammar.

### 2. Training

Given labeled emails, the classifier estimates:

**Prior probability** of each class:

$$P(C) = \frac{\text{number of emails in class } C}{\text{total emails}}$$

**Smoothed word likelihood** using Laplacian smoothing:

$$\hat{P}(w_i | C) = \frac{N_{w_i|C} + 1}{N_C + |V|}$$

Where $N_{w_i|C}$ is the word's count in class $C$, $N_C$ is the total word count in class $C$, and $|V|$ is the global vocabulary size. This ensures no word is ever assigned zero probability.

All probabilities are stored in **log-space** to prevent floating-point underflow when multiplying thousands of small probabilities.

### 3. Classification

For a new email, the classifier computes an unnormalized log-posterior score for each class:

$$\hat{C} = \underset{C \in \{\text{Spam, Ham}\}}{\text{argmax}} \left( \log P(C) + \sum_{i=1}^{k} \text{count}(w_i) \cdot \log \hat{P}(w_i | C) \right)$$

The denominator $P(\text{message})$ is identical for both classes and cancels out, so the classifier simply picks the class with the higher score.

### 4. Evaluation

The dataset is split 80/20 (train/test). Performance is measured using:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall percentage of correct predictions |
| **Precision** | Of emails flagged as spam, how many actually were |
| **Recall** | Of actual spam emails, how many were caught |
| **F1 Score** | Harmonic mean of precision and recall |

---

## Dataset

**SpamAssassin Public Mail Corpus** via Hugging Face (`talby/spamassassin`, `"text"` config)

- ~10,749 labeled emails
- Binary labels: spam / ham
- Source: [https://huggingface.co/datasets/talby/spamassassin](https://huggingface.co/datasets/talby/spamassassin)

> **Label encoding in this dataset:** `label = 0` → spam, `label = 1` → ham

---

## References

- Forsyth, D. (2018). *Probability and statistics for computer science*. Springer. https://doi.org/10.1007/978-3-319-64410-3
- Han, M. (2023). Spam filter based on Naive Bayes algorithm. *Applied and Computational Engineering, 15*, 247–252. https://doi.org/10.54254/2755-2721/15/20230844
- Noto, A. P., & Saputro, D. R. S. (2022). Classification data mining with Laplacian smoothing on Naïve Bayes method. *AIP Conference Proceedings, 2566*(1). https://doi.org/10.1063/5.0116519
- Stone, R. [talby]. (2023). *Spamassassin* [Data set]. Hugging Face. https://huggingface.co/datasets/talby/spamassassin
- Su, S. (2025). Research on spam filters based on NB algorithm. *ITM Web of Conferences, 70*, 01016. https://doi.org/10.1051/itmconf/20257001016
- Tsun, A. (2020). *Probability & statistics with applications to computing*. University of Washington. https://www.alextsun.com/book