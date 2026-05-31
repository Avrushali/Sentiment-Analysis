import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# --- NLTK Data Download ---
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# --- Configuration ---
MODEL_DIR = 'model'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_NB_PATH = os.path.join(MODEL_DIR, 'sentiment_nb_model.pkl')
MODEL_LR_PATH = os.path.join(MODEL_DIR, 'sentiment_lr_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
DATA_PATH = 'Sentiment140.csv' # Using the new dataset

# --- Data Preprocessing Functions ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Preprocesses the input text by:
    1. Converting to lowercase.
    2. Removing punctuation.
    3. Removing stopwords.
    4. Applying stemming.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# --- Main Training Function ---
def train_and_save_model():
    print("Starting model training with Sentiment140 dataset...")
    
    # 1. Data Handling: Load 'Sentiment140.csv'
    try:
        df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1', header=None)
        df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
        print(f"Successfully loaded {len(df)} rows from {DATA_PATH}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Map sentiment labels: 0=negative, 2=neutral, 4=positive
    sentiment_map = {0: 'negative', 2: 'neutral', 4: 'positive'}
    df['sentiment'] = df['target'].map(sentiment_map)
    
    # Filter out neutral sentiments if desired, or keep all three
    df = df[['text', 'sentiment']].copy()
    
    print("Sentiment distribution in Sentiment140 dataset:")
    print(df['sentiment'].value_counts())

    # 2. Preprocess the text
    print("Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split data into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['sentiment'], test_size=0.2, random_state=42
    )
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # TF-IDF Vectorization
    print("Training TF-IDF Vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(max_features=10000) # Increased features for larger dataset
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print("TF-IDF Vectorizer trained.")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

    # Train a Multinomial Naive Bayes classifier
    print("Training Multinomial Naive Bayes classifier...")
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)
    print("NB Classifier trained.")

    # Train a Logistic Regression classifier
    print("Training Logistic Regression classifier...")
    lr_classifier = LogisticRegression(max_iter=1000)
    lr_classifier.fit(X_train_tfidf, y_train)
    print("LR Classifier trained.")

    # 3. Model Evaluation
    print("\n--- Model Evaluation ---")
    
    def evaluate_model(model, name):
        y_pred = model.predict(X_test_tfidf)
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=['negative', 'neutral', 'positive']).tolist()
        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': cm
        }

    metrics = {
        'Naive Bayes': evaluate_model(nb_classifier, 'Naive Bayes'),
        'Logistic Regression': evaluate_model(lr_classifier, 'Logistic Regression')
    }

    # Save metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save the trained models
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(MODEL_NB_PATH, 'wb') as f:
        pickle.dump(nb_classifier, f)
    with open(MODEL_LR_PATH, 'wb') as f:
        pickle.dump(lr_classifier, f)
    print(f"Models, vectorizer, and metrics saved to {MODEL_DIR}/")

if __name__ == '__main__':
    train_and_save_model()
