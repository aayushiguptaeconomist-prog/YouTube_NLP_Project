# 🎬 Project Overview

This project performs end-to-end Natural Language Processing (NLP) analysis on real YouTube comments, using the YouTube Data API. The goal is to understand audience sentiment, dominant discussion themes, and emotional signals around a high-engagement video (Top Gun: Maverick trailer).

The project demonstrates a full real-world NLP pipeline:

Step 1. API-based data collection

Step 2. Text cleaning and exploratory data analysis

Step 3. Multiple sentiment and keyword analysis approaches

Step 4. Topic modeling

Step 5. Classical machine learning and transformer-based models

Step 6. Model comparison and interpretation

# 🎯 Objectives
* Collect real-world unstructured text data using an external API
* Quantify sentiment distribution in YouTube comments
* Identify dominant topics of discussion
* Compare rule-based, classical ML, and transformer-based NLP models
* Demonstrate interpretability and modeling tradeoffs


# 📊 Dataset

#### Source: YouTube Data API v3
#### Video: Top Gun: Maverick – Official Trailer
#### Data Collected:
* Comment text
* Author
* Like count
* Timestamp
* Size: ~2,000 top-level comments


## 🧠 Models Included

The notebook implements several analysis and modeling approaches to demonstrate a range of NLP techniques:

- **Model 1 — VADER (Rule-based Sentiment):** Fast, lexicon-based sentiment scoring for short social-media style text using the `vaderSentiment` package. Produces a compound score and mapped labels (positive/neutral/negative).
- **Model 2 — BERT (Hugging Face pipeline):** Transformer-based sentiment predictions via the Hugging Face `transformers` `pipeline('sentiment-analysis')`. Used to compare against VADER and downstream model labels.
- **Model 3 — Topic Modeling (LDA):** Uses `CountVectorizer` + `LatentDirichletAllocation` from scikit-learn to identify dominant discussion topics across comments.
- **Model 4 — TF-IDF + Logistic Regression:** A supervised classifier that uses `TfidfVectorizer` to create features and `LogisticRegression` to predict sentiment labels (trained/tested in the notebook).

Additional analyses in the notebook:

- Emotion and keyword-based scores (simple lexicon counts).
- Visualization: count plots, histograms, and word clouds for positive/negative comments.
- Model comparison: agreement rates and crosstabs between VADER and BERT labels.

## ⚙️ Requirements / Dependencies

Install the Python packages used by the notebook (tested on macOS with Python 3.8+):

```bash
pip install pandas seaborn matplotlib scikit-learn wordcloud google-api-python-client vaderSentiment transformers torch
```

Note: `transformers` requires `torch` or `tensorflow`; this notebook assumes `torch` is available.

## ▶️ How to run

1. Set your YouTube API key in the environment variable `YOUTUBE_API_KEY`:

```bash
export YOUTUBE_API_KEY="YOUR_KEY_HERE"
```

2. Open the notebook `youtube_nlp.ipynb` in Jupyter / VS Code and run the cells in order. The notebook will:

- Collect comments using the YouTube Data API (if `get_comments` is executed).
- Clean and preprocess text, then compute sentiment scores with VADER and BERT.
- Fit LDA topics and train/evaluate the TF-IDF + Logistic Regression classifier (if labels and splits are executed).

3. Inspect the visualizations (count plots, word clouds, histograms) and the model comparison tables.

## 📝 Notes & Tips

- For faster experimentation, save and load the CSV at `csv_files/youtube_comments.csv` instead of re-querying the API.
- If you run into transformer model download issues, ensure network access and enough disk space for model caches.
- Adjust `min_df`, number of LDA components, or classifier hyperparameters for different datasets.

