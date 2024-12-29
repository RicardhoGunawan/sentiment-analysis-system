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
from sklearn.naive_bayes import MultinomialNB
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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter
from googletrans import Translator
from textblob import TextBlob
import threading



# Konfigurasi logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Download required NLTK data if not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class SentimentAnalyzer(metaclass=SingletonMeta):
    # Inisialisasi _instance sebagai class variable
    _instance = None
    _lock = threading.Lock()
    def __init__(self, normalization_file_path='app/static/dictionary_words/normalization.txt'):
        if not hasattr(self, '_initialized'):
            # Initialize stemmer
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
            
            # Initialize stopwords
            stopword_factory = StopWordRemoverFactory()
            self.stop_words = set(stopword_factory.get_stop_words())

            # Initialize classifiers and vectorizer
            self.vectorizer = TfidfVectorizer()
            self.svm_classifier = SVC(kernel='linear')
            self.nb_classifier = MultinomialNB()

            # Load normalization dictionary
            self.normalization_dict = self._load_normalization_dict(normalization_file_path)
            
            # Initialize translator
            self.translator = Translator()
            
            # Mark as initialized
            self._initialized = True
            logging.info("SentimentAnalyzer initialized successfully")


    # def load_words_from_file(self, file_path):
    #     """Memuat kata-kata dari file teks ke dalam set."""
    #     try:
    #         with open(file_path, 'r', encoding='utf-8') as file:
    #             words = file.read().splitlines()
    #             words = [word.strip().lower() for word in words if word.strip()]
    #             return set(words)  # Ubah ke set untuk pencarian cepat
    #     except FileNotFoundError:
    #         logging.error(f"File {file_path} tidak ditemukan.")
    #         return set()  #  Kembalikan set kosong jika file tidak ditemukan

    def _load_normalization_dict(self, file_path):
        """Load normalization dictionary with caching."""
        if hasattr(self, 'normalization_dict'):
            return self.normalization_dict
            
        normalization_dict = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) == 2:
                            normalization_dict[parts[0].strip()] = parts[1].strip()
            logging.info(f"Normalization dictionary loaded: {len(normalization_dict)}")
        except FileNotFoundError:
            logging.error(f"File {file_path} tidak ditemukan.")
        return normalization_dict
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
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
        tokens = [self.normalization_dict.get(token, token) for token in tokens]

        # Stopwords removal
        tokens = [token for token in tokens if token not in self.stop_words]

        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens]

        return ' '.join(tokens)

    
    def translate_to_english(self, text):
        """Menerjemahkan teks bahasa Indonesia ke bahasa Inggris."""
        try:
            if pd.isna(text) or text == '':
                return ''
            translation = self.translator.translate(text, src='id', dest='en')
            return translation.text
        except Exception as e:
            logging.error(f"Error translasi: {e}")
            return text

    def save_translated_data(self, df, output_path='translated_data.csv'):
        """Menyimpan hasil terjemahan review ke file CSV."""
        try:
            if 'review' not in df.columns:
                raise ValueError("Kolom 'review' tidak ditemukan dalam DataFrame")

            translated_df = df.copy()
            # Preprocess dulu
            translated_df['processed_review'] = translated_df['review'].apply(self.preprocess_text)
            # Lalu translate
            translated_df['translated_review'] = translated_df['processed_review'].apply(self.translate_to_english)
            translated_df.to_csv(output_path, index=False)
            logging.info(f"Data terjemahan berhasil disimpan ke {output_path}")
            return translated_df
        except Exception as e:
            logging.error(f"Error menyimpan data terjemahan: {e}")
            raise

    def get_textblob_sentiment(self, text):
        """Mendapatkan sentiment menggunakan TextBlob."""
        try:
            analysis = TextBlob(text)
            # Konversi polaritas TextBlob ke label sentiment
            if analysis.sentiment.polarity > 0:
                return 'positive'
            elif analysis.sentiment.polarity < 0:
                return 'negative'
            else:
                return 'neutral'
        except Exception as e:
            logging.error(f"Error analisis TextBlob: {e}")
            return 'neutral'

    def prepare_data(self, df):
        """Menyiapkan data untuk training dengan translasi dan analisis sentiment."""
        if df is None or len(df) == 0:
            raise ValueError("DataFrame input kosong atau None")
        
        if 'review' not in df.columns:
            raise ValueError("Kolom 'review' tidak ditemukan dalam DataFrame")
        
        # Hapus duplikat
        df_no_duplicates = df.drop_duplicates(subset=['review'])
        logging.info(f"Menghapus {len(df) - len(df_no_duplicates)} review duplikat")
        
        try:
            # Preprocess dan translate
            df_no_duplicates['processed_review'] = df_no_duplicates['review'].apply(self.preprocess_text)
            df_no_duplicates['translated_review'] = df_no_duplicates['processed_review'].apply(self.translate_to_english)
            
            # Analisis sentiment menggunakan TextBlob
            df_no_duplicates['sentiment_label'] = df_no_duplicates['translated_review'].apply(self.get_textblob_sentiment)
            
            df_filtered = df_no_duplicates.dropna(subset=['sentiment_label'])
            logging.info(f"Distribusi label setelah filtering:\n{df_filtered['sentiment_label'].value_counts()}")
            
            return df_filtered
            
        except Exception as e:
            logging.error(f"Error dalam prepare_data: {e}")
            raise

    def get_sentiment_label(self, text):
        """Memproses teks dan menentukan sentiment menggunakan translasi dan TextBlob."""
        try:
            # 1. Preprocess teks
            processed_text = self.preprocess_text(text)
            
            # 2. Translate ke bahasa Inggris
            translated_text = self.translate_to_english(processed_text)
            
            # 3. Dapatkan sentiment menggunakan TextBlob
            sentiment = self.get_textblob_sentiment(translated_text)
            
            return sentiment
        except Exception as e:
            logging.error(f"Error dalam pipeline analisis sentiment: {e}")
            return 'neutral'

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
        """Train both SVM and Naive Bayes models and save them"""
        train_data = self.prepare_data(train_data)
        
        X_train = train_data['review']
        y_train = train_data['sentiment_label']

        # Preprocessing dan vectorisasi
        X_train_processed = X_train.apply(self.preprocess_text)
        X_train_tfidf = self.vectorizer.fit_transform(X_train_processed)

        # Train both models
        logging.info("Training SVM classifier...")
        self.svm_classifier.fit(X_train_tfidf, y_train)
        logging.info("SVM training completed")

        logging.info("Training Naive Bayes classifier...")
        self.nb_classifier.fit(X_train_tfidf, y_train)
        logging.info("Naive Bayes training completed")

        # Save models and vectorizer
        joblib.dump(self.svm_classifier, 'sentiment_model_svm.pkl')
        joblib.dump(self.nb_classifier, 'sentiment_model_nb.pkl')
        joblib.dump(self.vectorizer, 'vectorizer.pkl')
    
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

    def evaluate(self, test_data, hotel_unit):
        """Evaluate both models and return their results"""
        test_data = test_data.fillna('')

        X_test = test_data['review']
        y_test = test_data['sentiment_label']

        # Vectorize test data
        X_test_processed = X_test.apply(self.preprocess_text)
        X_test_tfidf = self.vectorizer.transform(X_test_processed)

        results = {}

        # Evaluate SVM
        y_pred_svm = self.svm_classifier.predict(X_test_tfidf)
        svm_accuracy = accuracy_score(y_test, y_pred_svm)
        svm_report = classification_report(y_test, y_pred_svm, 
                                        target_names=['Negative', 'Neutral', 'Positive'],
                                        output_dict=True)
        svm_cm = confusion_matrix(y_test, y_pred_svm)
        
        # Count label occurrences for SVM using Counter
        svm_label_counts = dict(Counter(y_pred_svm))

        # Plot SVM confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Neutral', 'Positive'], 
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title('SVM Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        app = create_app()
        svm_cm_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                    f'confusion_matrix_svm_{hotel_unit}.png')
        plt.savefig(svm_cm_path, format='png')
        plt.close()

        # Evaluate Naive Bayes
        y_pred_nb = self.nb_classifier.predict(X_test_tfidf)
        nb_accuracy = accuracy_score(y_test, y_pred_nb)
        nb_report = classification_report(y_test, y_pred_nb, 
                                        target_names=['Negative', 'Neutral', 'Positive'],
                                        output_dict=True)
        nb_cm = confusion_matrix(y_test, y_pred_nb)
        
        # Count label occurrences for Naive Bayes using Counter
        nb_label_counts = dict(Counter(y_pred_nb))

        # Plot Naive Bayes confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Neutral', 'Positive'], 
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title('Naive Bayes Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        nb_cm_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                f'confusion_matrix_nb_{hotel_unit}.png')
        plt.savefig(nb_cm_path, format='png')
        plt.close()

        return {
            'svm': {
                'accuracy': svm_accuracy,
                'report': svm_report,
                'confusion_matrix': svm_cm_path,
                'label_counts': svm_label_counts
            },
            'naive_bayes': {
                'accuracy': nb_accuracy,
                'report': nb_report,
                'confusion_matrix': nb_cm_path,
                'label_counts': nb_label_counts
            }
        }

        
    def predict(self, texts, model='svm'):
        """Predicts sentiment using the specified model"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        vectors = self.vectorizer.transform(processed_texts)
        
        if model.lower() == 'svm':
            return self.svm_classifier.predict(vectors)
        elif model.lower() == 'nb':
            return self.nb_classifier.predict(vectors)
        else:
            raise ValueError("Model must be either 'svm' or 'nb'")


    def save_model(self, base_path='models'):
        """Save all models and vectorizer"""
        try:
            joblib.dump(self.vectorizer, os.path.join(base_path, 'vectorizer.pkl'))
            joblib.dump(self.svm_classifier, os.path.join(base_path, 'sentiment_model_svm.pkl'))
            joblib.dump(self.nb_classifier, os.path.join(base_path, 'sentiment_model_nb.pkl'))
            logging.info("All models saved successfully")
        except Exception as e:
            logging.error(f"Error saving models: {e}")

    def load_model(self, base_path='models'):
        """Load all models and vectorizer"""
        try:
            if all(os.path.exists(os.path.join(base_path, f)) for f in 
                ['vectorizer.pkl', 'sentiment_model_svm.pkl', 'sentiment_model_nb.pkl']):
                self.vectorizer = joblib.load(os.path.join(base_path, 'vectorizer.pkl'))
                self.svm_classifier = joblib.load(os.path.join(base_path, 'sentiment_model_svm.pkl'))
                self.nb_classifier = joblib.load(os.path.join(base_path, 'sentiment_model_nb.pkl'))
                logging.info("All models loaded successfully")
            else:
                raise FileNotFoundError("One or more model files not found")
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise
