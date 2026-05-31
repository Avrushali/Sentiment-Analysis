# Sentiment Analysis & Business Intelligence System

A full continuous Data Science and Business Intelligence engineering project capable of extracting tweets, processing NLP, evaluating ML models, and visualizing executive business insights through an interactive dashboard.

## Overview
This platform marries modern data manipulation (Pandas, Scikit-Learn) with web technologies (Flask, Chart.js) to bring model pipelines straight to Business Users.

Features:
- **Instant NLP Sentiment Prediction**: Test raw text and view live probabilistic confidence scores using the polished web interactive layer.
- **Model Comparison Engine**: Automatically compares the mathematical performance of `Logistic Regression` & `Multinomial Naive Bayes` classifiers on your 1.6 Million Tweet dataset (Sentiment140).
- **Business Intelligence (BI) Dashboard**:
  - Trailing 12-month Trend visualization.
  - Sentiment Distribution mapping.
  - Granular Model comparison chart (Accuracy, Precision, Recall, F1).
  - Side-by-side Confusion Matrix analytical views.
- **Microservices API Layer**: Interact with models systemically via HTTP POST. 

## Installation

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Train Models
# (Note: This will read from Sentiment140.csv and output metrics.json + model pickles)
python train_model.py

# 4. Start the Application
python app.py
```

## API Documentation

### 1. Sentiment Predictor: `POST /api/predict`
Predicts the sentiment and returns confidence scores.
**Input (JSON):**
```json
{
  "text": "I absolutely love the new user interface!",
  "model": "lr" 
}
```
*Note: `model` can be `"lr"` (Logistic Regression) or `"nb"` (Naive Bayes).*

**Output Response (JSON):**
```json
{
  "text": "I absolutely love the new user interface!",
  "prediction": "positive",
  "confidence": 98.45,
  "model": "Logistic Regression"
}
```

### 2. Business Intelligence Aggregation: `GET /api/dashboard_data`
Provides raw metric pipelines driving the Analytics dashboard, including the `metrics_comparison` tree detailing Precision/Recall heuristics and Confusion Matrices.

## Technical Architecture
- **NLP Preprocessing Engine:** Uses NLTK for tokenization, stop-word reduction, and Porter Stemming, mapped to a memory-efficient `TfidfVectorizer(max_features=10000)`.
- **Stylistic Layer:** Ground-up modern Dark Mode CSS implemented with backdrop glassmorphism to recreate the feel of Enterprise SaaS analytic platforms (Vercel/Linear).

## Actionable Insights Generated
With the included metrics endpoints, organizations can map sentiment trends exactly against product release cycles, capturing instant ROI measurements on feature changes based on real user text-data confidence intervals.
