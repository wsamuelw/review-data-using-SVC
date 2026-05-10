# Sentiment Prediction from Restaurant Reviews

Classifying restaurant reviews as positive or negative using Support Vector Classification (SVC) — with text vectorisation via CountVectorizer.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wsamuelw/review-data-using-SVC/blob/main/2%2C%20review_data_using_SVC.ipynb)

## Problem

Given 1,000 restaurant reviews with binary labels (liked / not liked), build a model that predicts sentiment from raw text. The challenge: converting free-form text into numerical features that an SVM can learn from.

## Approach

1. **Vectorise** text using `CountVectorizer` — converts words into a bag-of-words matrix with English stop words removed
2. **Split** data 75/25 with stratified sampling
3. **Train** an SVC (RBF kernel, default parameters)
4. **Evaluate** accuracy on held-out test set
5. **Test** on unseen custom text

## What's Inside

| Notebook | What It Does |
|----------|-------------|
| `1, CountVectorizer_demo.ipynb` | Standalone demo of unigram and bigram vectorisation — how text becomes numbers |
| `2, review_data_using_SVC.ipynb` | Full pipeline: load data → vectorise → train SVC → evaluate → predict on unseen text |

## Results

The model achieves strong accuracy on the test set. Class distribution is balanced (~50/50 positive/negative), so accuracy is a reliable metric here.

Key output: the model can predict sentiment on completely unseen text:

```python
unseen_text = vect.transform(["Good customer service! The food was nice"])
model.predict(unseen_text)  # => [1] (positive)
```

## Setup

### Google Colab

Click the badge above — no setup required.

### Local

```bash
pip install scikit-learn pandas matplotlib seaborn
git clone https://github.com/wsamuelw/review-data-using-SVC.git
cd review-data-using-SVC
jupyter notebook "2, review_data_using_SVC.ipynb"
```

## Data

**Restaurant Reviews** — 1,000 reviews scraped from restaurant listings. Tab-separated with two columns:

| Column | Type | Description |
|--------|------|------------|
| `Review` | string | Free-text review |
| `Liked` | int (0/1) | Binary sentiment label |

Class distribution: ~50% positive, ~50% negative.

## How CountVectorizer Works

Raw text → tokenise → remove stop words → build vocabulary → create word-count matrix:

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['Great food', 'Terrible service']
vect = CountVectorizer(stop_words='english')
X = vect.fit_transform(corpus)

# Vocabulary: ['food', 'great', 'service', 'terrible']
# Matrix: [[1, 1, 0, 0],
#           [0, 0, 1, 1]]
```

Each row is a document, each column is a word in the vocabulary. The value is how many times that word appears.

**Bigrams** (2-word phrases) capture context that single words miss:

```python
vect = CountVectorizer(ngram_range=(2, 2))
# 'not good' → single feature, captures negation
```

## Why SVC for Text?

- **Works well in high dimensions** — text vectors have thousands of features (one per word), SVMs handle this naturally
- **Kernel trick** — RBF kernel captures non-linear relationships without explicit feature engineering
- **Robust to overfitting** — margin maximisation generalises well on small-to-medium datasets

## Tech Stack

- **scikit-learn** — CountVectorizer, SVC, train_test_split, accuracy_score
- **pandas** — data loading and manipulation
- **matplotlib / seaborn** — class distribution visualisation

## References

- [SVM introduction](https://monkeylearn.com/blog/introduction-to-support-vector-machines-svm/)
- [CountVectorizer docs](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [Restaurant Reviews analysis](https://www.analyticsvidhya.com/blog/2022/02/restaurant-reviews-analysis-model-based-on-ml-algorithms/)

## License

MIT
