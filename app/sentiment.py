import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import os
from bs4 import BeautifulSoup
import string
from textblob import TextBlob


# Download required NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

class SentimentAnalyzer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = CountVectorizer()  # Menggunakan CountVectorizer
        self.classifier = SVC(kernel='linear')

    def preprocess_text(self, text):
        """Preprocesses input text for sentiment analysis."""
        if isinstance(text, str):
            # 1. URL removal
            text = re.sub(r'http(s)?://\S+|www\.\S+', '', text)
            # 2. Email removal
            text = re.sub(r'\S+@\S+', '', text)
            # 3. HTML Tags removal
            text = BeautifulSoup(text, 'html.parser').get_text()
            # 4. Punctuation removal
            text = text.translate(str.maketrans('', '', string.punctuation))
            # 5. Extra whitespaces removal
            text = ' '.join(text.split())
            # 6. Convert to lowercase
            text = text.lower()
            # 7. Tokenization and stemming
            tokens = word_tokenize(text)
            tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
            return ' '.join(tokens)
        else:
            print(f"Warning: Expected string input but got {type(text)}.")
            return ''  # Return empty string for non-string input

    def get_sentiment_label(self, text):
        """Get sentiment label from text using TextBlob."""
        if isinstance(text, str):  # Pastikan input adalah string
            analysis = TextBlob(text)
            score = analysis.sentiment.polarity
            if score > 0:
                return 'positive'
            elif score < 0:
                return 'negative'
            else:
                return 'neutral'
        else:
            return 'neutral'  # Kembalikan label 'neutral' untuk non-string


    def prepare_data(self, df):
        """Prepares data for training by processing reviews."""
        df['processed_review'] = df['review'].apply(self.preprocess_text)
        # Use TextBlob to create sentiment labels
        df['label'] = df['review'].apply(self.get_sentiment_label)  # Menambahkan kolom label
        return df

    def train(self, X_train, y_train):
        """Trains the sentiment analysis model."""
        X_train_vectors = self.vectorizer.fit_transform(X_train)  # Fit CountVectorizer
        self.classifier.fit(X_train_vectors, y_train)

    def train_and_save(self, df):
        """Trains and saves the model using provided data and labels."""
        df = self.prepare_data(df)  # Prepare the data
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_review'], df['label'], test_size=0.2, random_state=42
        )
        self.train(X_train, y_train)  # Train the model with processed reviews
        self.save_model()  # Save the trained model for future use

    def predict(self, texts):
        """Predicts the sentiment for the given texts."""
        processed_texts = [self.preprocess_text(text) for text in texts]
        vectors = self.vectorizer.transform(processed_texts)  # Use CountVectorizer
        return self.classifier.predict(vectors)

    def save_model(self, vectorizer_path='vectorizer.pkl', classifier_path='classifier.pkl'):
        """Saves the trained model."""
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(classifier_path, 'wb') as f:
            pickle.dump(self.classifier, f)

    def load_model(self, vectorizer_path='vectorizer.pkl', classifier_path='classifier.pkl'):
        """Loads a pretrained model if it exists."""
        if os.path.exists(vectorizer_path) and os.path.exists(classifier_path):
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
        else:
            raise FileNotFoundError("Model files not found. Please train the model first.")
