from flask import app
import numpy as np
import pandas as pd
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
import joblib
import os
import string
from bs4 import BeautifulSoup
import logging
from app import create_app
from sklearn.metrics import accuracy_score, confusion_matrix

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Download required NLTK data if not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class SentimentAnalyzer:
    def __init__(self, positive_words_path='app/static/dictionary_words/positive_words.txt', 
                 negative_words_path='app/static/dictionary_words/negative_words.txt'):
        # Gunakan stemmer dari Sastrawi
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        # Gunakan stopwords dari Sastrawi
        stopword_factory = StopWordRemoverFactory()
        self.stop_words = set(stopword_factory.get_stop_words())

        self.vectorizer = TfidfVectorizer()  # Ubah ke TfidfVectorizer
        self.classifier = SVC(kernel='linear')  # Gantilah Naive Bayes dengan SVC

        # Path ke file kamus kata positif dan negatif
        self.positive_words_path = positive_words_path
        self.negative_words_path = negative_words_path

        # Memuat kata-kata positif dan negatif
        self.positive_words = self.load_words_from_file(positive_words_path)
        self.negative_words = self.load_words_from_file(negative_words_path)

        # Logging informasi
        logging.info(f"Positive words loaded: {len(self.positive_words)}")
        logging.info(f"Negative words loaded: {len(self.negative_words)}")

    def load_words_from_file(self, file_path):
        """Memuat kata-kata dari file teks ke dalam set."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                words = file.read().splitlines()
                words = [word.strip().lower() for word in words if word.strip()]
                return set(words)  # Ubah ke set untuk pencarian cepat
        except FileNotFoundError:
            logging.error(f"File {file_path} tidak ditemukan.")
            return set()  # Kembalikan set kosong jika file tidak ditemukan

    def preprocess_text(self, text):
        """Preprocesses input text dengan penanganan NaN."""
        # Pastikan input adalah string
        if pd.isna(text) or text is None:
            return ''
        
        if not isinstance(text, str):
            text = str(text)
        
        # 1. Case folding
        text = text.lower()

        # 2. URL removal
        text = re.sub(r'http(s)?://\S+|www\.\S+', '', text)

        # 3. Email removal
        text = re.sub(r'\S+@\S+', '', text)

        # 4. Date removal
        text = re.sub(r'\d{1,2}(st|nd|rd|th)?[-./]\d{1,2}[-./]\d{2,4}', '', text)

        # 5. HTML tags removal
        text = BeautifulSoup(text, 'html.parser').get_text()

        # 6. Emojis removal
        text = re.sub(r"[" 
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      "]+", '', text)

        # 7. Hashtags and mentions removal
        text = re.sub(r'(@\S+|#\S+)', '', text)

        # 8. Punctuation removal
        text = text.translate(str.maketrans('', '', string.punctuation))

        # 9. Number removal
        text = re.sub(r'\d+', '', text)

        # 10. Extra whitespaces removal
        text = ' '.join(text.split())

        # Tokenizing
        tokens = word_tokenize(text)

        # Normalization
        normalization_dict = {
            'gak': 'tidak', 'nggak': 'tidak', 'ga': 'tidak', 'aja': 'saja',
            'kok': 'tidak', 'dong': '', 'lah': 'sudah', 'memakan': 'makan',
            'kmarin': 'kemarin', 'cocok': 'bagus', 'muas': 'puas',
            'sarap': 'sarapan', 'alternativ': 'alternatif','tibatiba' :'tiba'
        }
        tokens = [normalization_dict.get(token, token) for token in tokens]

        # Stopwords removal
        tokens = [token for token in tokens if token not in self.stop_words]

        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens]

        return ' '.join(tokens)
    
    def get_sentiment_label(self, text):
        """Determine sentiment label using custom Indonesian sentiment dictionary."""
        if not isinstance(text, str):
            logging.warning(f"Non-string input received: {text} of type {type(text)}")
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
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty or None")
        
        required_columns = ['review']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

        df_no_duplicates = df.drop_duplicates(subset=['review'])
        logging.info(f"Removed {len(df) - len(df_no_duplicates)} duplicate reviews")
        
        if 'sentiment_label' not in df_no_duplicates.columns:
            df_no_duplicates['processed_review'] = df_no_duplicates['review'].apply(self.preprocess_text)
            df_no_duplicates['sentiment_label'] = df_no_duplicates['review'].apply(self.get_sentiment_label)

        df_filtered = df_no_duplicates.dropna(subset=['sentiment_label'])
        logging.info(f"Label distribution after filtering:\n{df_filtered['sentiment_label'].value_counts()}")
        logging.info(f"Total reviews after preprocessing: {len(df_filtered)}")
        
        if len(df_filtered) == 0:
            logging.warning("No valid data found. Using original data with neutral labels.")
            df_filtered = df_no_duplicates.copy()
            df_filtered['sentiment_label'] = 'neutral' 
        
        return df_filtered

    def train(self, X_train, y_train):
        # Pastikan X_train dan y_train memiliki panjang yang sama
        if len(X_train) != len(y_train):
            raise ValueError(f"Mismatch in data lengths. X_train: {len(X_train)}, y_train: {len(y_train)}")
        
        # Preprocessing dan vectorisasi
        X_train_processed = X_train.apply(self.preprocess_text)
        X_train_tfidf = self.vectorizer.fit_transform(X_train_processed)

        logging.info("Training sentiment classifier using SVM...")
        self.classifier.fit(X_train_tfidf, y_train)  # Melatih model dengan data
        logging.info("SVM classifier training completed.")

        
    def train_and_save(self, train_data):
        train_data = self.prepare_data(train_data)
        
        X_train = train_data['review']
        y_train = train_data['sentiment_label']

        # Preprocessing dan vectorisasi
        X_train_processed = X_train.apply(self.preprocess_text)
        X_train_tfidf = self.vectorizer.fit_transform(X_train_processed)

        # Gunakan self.classifier yang sudah diinisialisasi di __init__
        self.classifier.fit(X_train_tfidf, y_train)

        # Simpan model dan vectorizer
        joblib.dump(self.classifier, 'sentiment_model_svm.pkl')
        joblib.dump(self.vectorizer, 'vectorizer.pkl')

    def predict(self, texts):
        """Predicts the sentiment for the given texts."""
        processed_texts = [self.preprocess_text(text) for text in texts]
        vectors = self.vectorizer.transform(processed_texts)
        return self.classifier.predict(vectors)
    
    def create_wordcloud_base64(self, reviews):
        # Gabungkan semua review menjadi satu string
        text = ' '.join(reviews)
        
        # Membuat wordcloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        # Menyimpan gambar wordcloud ke dalam buffer
        img = BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        
        # Mengubah gambar menjadi base64
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        
        return img_base64

    def evaluate(self, test_data):
            test_data = test_data.fillna('')

            X_test = test_data['review']
            y_test = test_data['sentiment_label']

            X_test_tfidf = self.vectorizer.transform(X_test)
            y_pred = self.classifier.predict(X_test_tfidf)

            accuracy = accuracy_score(y_test, y_pred)

            # Hitung confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Buat plot confusion matrix dengan seaborn
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

            # Simpan confusion matrix sebagai gambar di folder uploads
            # Ubah path untuk menyimpan gambar confusion matrix
            app = create_app() 
            cm_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'confusion_matrix.png')
            plt.savefig(cm_image_path, format='png')
            plt.close()

            return accuracy, cm_image_path

    def save_model(self, vectorizer_path='vectorizer.pkl', classifier_path='classifier.pkl'):
        try:
            joblib.dump(self.vectorizer, vectorizer_path)
            joblib.dump(self.classifier, classifier_path)
            logging.info(f"Model saved to {vectorizer_path} and {classifier_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self, vectorizer_path='vectorizer.pkl', classifier_path='classifier.pkl'):
        try:
            if os.path.exists(vectorizer_path) and os.path.exists(classifier_path):
                self.vectorizer = joblib.load(vectorizer_path)
                self.classifier = joblib.load(classifier_path)
                logging.info("Model successfully loaded.")
            else:
                raise FileNotFoundError("Model files not found. Please train the model first.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
