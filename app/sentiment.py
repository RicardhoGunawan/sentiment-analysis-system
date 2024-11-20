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

# Download required NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

class SentimentAnalyzer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        # Buat daftar stopwords bahasa Indonesia
        self.stop_words = set([
            'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau',
            'ini', 'itu', 'juga', 'sudah', 'saya', 'anda', 'dia', 'mereka', 'kita', 'akan',
            'bisa', 'ada', 'tidak', 'saat', 'oleh', 'setelah', 'tentang', 'seperti', 'ketika',
            'bagi', 'sampai', 'karena', 'jika', 'namun', 'serta', 'lain', 'sebuah', 'para'
        ])
        self.vectorizer = CountVectorizer()
        self.classifier = SVC(kernel='linear')
        
        # Kamus kata positif dan negatif bahasa Indonesia
        self.positive_words = set([
            'bagus', 'baik', 'suka', 'senang', 'puas', 'mantap', 'keren', 'enak',
            'recommended', 'rekomen', 'memuaskan', 'ramah', 'cepat', 'bersih',
            'nyaman', 'top', 'oke', 'ok', 'recommended', 'worth', 'mantul'
        ])
        
        self.negative_words = set([
            'buruk', 'jelek', 'kecewa', 'mahal', 'lambat', 'kotor', 'kasar',
            'tidak', 'jangan', 'hancur', 'rusak', 'busuk', 'mahal', 'kurang',
            'cacad', 'rugi', 'bau', 'pahit', 'basi', 'lecet'
        ])

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
            return ''

    def get_sentiment_label(self, text):
        """Get sentiment label from text using custom Indonesian sentiment dictionary."""
        if not isinstance(text, str):
            return 'neutral'
            
        # Preprocess the text
        text = text.lower()
        words = word_tokenize(text)
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        # Calculate sentiment based on word counts
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

    def prepare_data(self, df):
        """Prepares data for training by processing reviews and removing duplicates."""
        # Remove duplicate reviews
        df_no_duplicates = df.drop_duplicates(subset=['review'])
        print(f"Removed {len(df) - len(df_no_duplicates)} duplicate reviews")
        
        # Process the reviews
        df_no_duplicates['processed_review'] = df_no_duplicates['review'].apply(self.preprocess_text)
        # Create sentiment labels using custom Indonesian method
        df_no_duplicates['label'] = df_no_duplicates['review'].apply(self.get_sentiment_label)
        
        return df_no_duplicates

    def train(self, X_train, y_train):
        """Trains the sentiment analysis model."""
        X_train_vectors = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vectors, y_train)

    def train_and_save(self, df):
        """Trains and saves the model using provided data and labels."""
        df = self.prepare_data(df)  # Prepare the data and remove duplicates
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_review'], df['label'], test_size=0.2, random_state=42
        )
        self.train(X_train, y_train)
        self.save_model()

    def predict(self, texts):
        """Predicts the sentiment for the given texts."""
        processed_texts = [self.preprocess_text(text) for text in texts]
        vectors = self.vectorizer.transform(processed_texts)
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