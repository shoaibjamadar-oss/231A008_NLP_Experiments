# Fake Review Detection for E-Commerce Platforms using NLP

An NLP-based text classification system that detects whether an e-commerce product review is **genuine (OR)** or **fake / computer-generated (CG)**, built using TF-IDF feature extraction and a Naive Bayes classifier.

> NLP Course Mini Project — CSDOL7011

---

## Problem Statement

With an increasing number of people shopping online, product reviews serve as an essential source of information to them, but a significant number of these reviews are fake, computer-generated, or simply intended to affect the reputation of a certain product undeservedly. This project proposes an NLP-based solution to detect whether a certain review is genuine or fake by applying a text classification algorithm, namely, the Naive Bayes classifier.

### Problem Description
Fake reviews, either negative or positive, present a problem for being misleading to potential buyers and skewing the actual reputation of a product. The main issue leads to the fact that it is computationally inexpensive to generate a large quantity of fake reviews and, at scale, it becomes impossible to moderate each review manually.

### Objectives
To construct a text classifier that would be able to distinguish between real and fake reviews with a reasonable accuracy that would be achieved by utilizing the difference in word patterns between the two classes.

### Scope
The proposed solution would operate solely on the data of review text; behavioral patterns, images, and scraping data from other sources are out of the scope of this project.

### Target Audience
- Online shoppers that seek to receive fair and honest feedback about products they intend to purchase
- E-commerce websites that wish to improve their reputation by reducing the presence of fake reviews
- Honest reviewers that seek to prevent their reviews from being canceled or penalized

### Data Requirements
The labeled dataset of Amazon reviews with their classes of authenticity (available on Kaggle) that would be preprocessed to extract the text data and would undergo stemming or lemmatization.

### Challenges and Assumptions
One might argue that the fake reviews crafted by humans would not be significantly different from the ones written by humans. However, it is assumed that the fake reviews would have a clearly distinguishable pattern from genuine ones that would allow the model to achieve a sufficient accuracy. Another challenge could be presented by the volume of data, but the solution would rely on a relatively small amount of data for proof of concept.

### Evaluation Metrics
Classification accuracy, precision, recall, F1-score, and a confusion matrix.

### Tools and Techniques
Python programming language, NLTK library, TF-IDF, Naive Bayes algorithm (scikit-learn).

### Impact and Significance
The solution would increase the level of trust online buyers have in online reviews and provide e-commerce sites with a tool to abate the presence of fake reviews on their platforms.

### Deliverables
The trained Naive Bayes classifier, a Python script/notebook with comments, and an example/demo of the working model that would classify a user-submitted review.

---

## Methodology

```
Load Dataset → Preprocess Text → TF-IDF Vectorize → Train Naive Bayes → Evaluate → Predict Demo
```

1. **Load Dataset** — Amazon reviews dataset (Kaggle) with review text and authenticity labels.
2. **Preprocess Text** — lowercase, remove URLs/digits/punctuation, tokenize, remove stopwords, lemmatize.
3. **Feature Extraction** — convert cleaned text into numeric vectors using TF-IDF.
4. **Train/Test Split** — 80/20 split, stratified on label.
5. **Model Training** — Multinomial Naive Bayes.
6. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix.
7. **Prediction Demo** — classify a new, user-submitted review as real or fake.

## Dataset

- **Source:** [Kaggle — Fake Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)
- **Size:** 40,432 reviews (20,216 genuine / 20,216 computer-generated)
- **Columns:** `category`, `rating`, `label` (`OR`/`CG`), `text_`

## Results

| Metric | Score |
|---|---|
| Accuracy | 84.75% |
| Precision | 83.19% |
| Recall | 87.12% |
| F1-score | 85.11% |

## Tech Stack

- Python
- [NLTK](https://www.nltk.org/) — text preprocessing (stopwords, lemmatization)
- [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorization, Naive Bayes, evaluation metrics
- pandas / NumPy — data handling
- matplotlib / seaborn — confusion matrix visualization

## Project Structure

```
├── fake_review_detection.ipynb   # Main notebook (Colab-ready)
├── README.md
└── requirements.txt
```

## Setup & Usage

1. Clone the repository
   ```bash
   git clone <repo-url>
   cd fake-review-detection-nlp
   ```

2. Install dependencies
   ```bash
   pip install pandas numpy nltk scikit-learn matplotlib seaborn kagglehub
   ```

3. Download the dataset (via `kagglehub`)
   ```python
   import kagglehub
   path = kagglehub.dataset_download("mexwell/fake-reviews-dataset")
   ```

4. Run the notebook — `fake_review_detection.ipynb` (or open in Google Colab)

5. Try the live demo cell to classify any review text of your choice:
   ```python
   predict_review("Amazing amazing amazing best product ever buy now highly recommend")
   # → Fake (Computer-Generated)
   ```

## Limitations

- Well-written fake reviews can closely mimic the style of genuine ones.
- Trained on a single dataset/product domain — generalization to other domains is untested.
- Uses review text only; does not incorporate reviewer behavior, images, or metadata signals.

## Author

Shoaib — BE, CSE (AI & ML)
