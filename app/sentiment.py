import numpy as np
import pandas as pd
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import os
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
import io
import string
import logging

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

        self.vectorizer = CountVectorizer()
        self.classifier = SVC(kernel='linear')

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
        """Preprocesses input text for sentiment analysis."""
        if not isinstance(text, str):
            logging.warning(f"Warning: Expected string input but got {type(text)}.")
            return ''

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
    
    def create_wordcloud(self, texts, output_path='wordcloud.png'):
        """Membuat word cloud dari teks yang diberikan."""
        # Menggabungkan semua teks menjadi satu string
        all_text = ' '.join(texts)
        
        # Preprocessing
        all_text = self.preprocess_text(all_text)

        # Membuat WordCloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            stopwords=self.stop_words,
            colormap='viridis'
        ).generate(all_text)

        # Simpan WordCloud ke file
        wordcloud.to_file(output_path)
        logging.info(f"WordCloud saved to {output_path}")

        # Return image path
        return output_path

    def create_wordcloud_base64(self, texts):
        """Membuat *Word Cloud* dan mengembalikannya dalam format base64 untuk dashboard."""
        all_text = ' '.join(texts)
        all_text = self.preprocess_text(all_text)

        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            stopwords=self.stop_words,
            colormap='viridis'
        ).generate(all_text)

        # Konversi ke base64
        img_buffer = io.BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_buffer.seek(0)
        base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return base64_image

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
        # Cek input DataFrame
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty or None")

        # Cek keberadaan kolom yang diperlukan
        required_columns = ['review']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

        # Remove duplicate reviews
        df_no_duplicates = df.drop_duplicates(subset=['review'])
        logging.info(f"Removed {len(df) - len(df_no_duplicates)} duplicate reviews")
        
        # Process the reviews
        df_no_duplicates['processed_review'] = df_no_duplicates['review'].apply(self.preprocess_text)
        
        # Create sentiment labels using custom Indonesian method
        df_no_duplicates['label'] = df_no_duplicates['review'].apply(self.get_sentiment_label)
        
        # Hapus baris dengan label None
        df_filtered = df_no_duplicates.dropna(subset=['label'])
        
        logging.info(f"Label distribution after filtering:\n{df_filtered['label'].value_counts()}")
        logging.info(f"Total reviews after preprocessing: {len(df_filtered)}")
        
        # Fallback: Jika tidak ada data setelah filtering, kembalikan data asli
        if len(df_filtered) == 0:
            logging.warning("No valid data found. Using original data with neutral labels.")
            df_filtered = df_no_duplicates.copy()
            df_filtered['label'] = 'neutral'  # Set default label
        
        return df_filtered

    def train(self, X_train, y_train):
        # Konversi X_train dan y_train ke list jika masih berupa DataFrame atau Series
        if hasattr(X_train, 'values'):
            X_train = X_train.values.tolist()
        if hasattr(y_train, 'values'):
            y_train = y_train.values.tolist()

        # Validasi input
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data is empty. Check your data preparation.")
        
        # Pastikan X_train dan y_train memiliki panjang yang sama
        if len(X_train) != len(y_train):
            raise ValueError(f"Mismatch in data lengths. X_train: {len(X_train)}, y_train: {len(y_train)}")
        
        # Filter data dengan label yang valid
        valid_indices = [i for i, label in enumerate(y_train) if 
                        label is not None and 
                        str(label).lower() in ['positive', 'negative', 'neutral']]
        
        if not valid_indices:
            raise ValueError("No valid training data found after filtering.")
        
        # Gunakan list comprehension dengan aman
        X_train_filtered = [X_train[i] for i in valid_indices]
        y_train_filtered = [y_train[i] for i in valid_indices]

        # Transform data fitur dengan vectorizer
        X_train_vectors = self.vectorizer.fit_transform(X_train_filtered)

        # Latih classifier
        logging.info("Training sentiment classifier...")
        self.classifier.fit(X_train_vectors, y_train_filtered)
        logging.info("Classifier training completed.")

    def train_and_save(self, df):
        """Trains and saves the model using provided data and labels."""
        try:
            # Prepare the data and remove duplicates
            df_prepared = self.prepare_data(df)  
            
            # Validasi data setelah persiapan
            if len(df_prepared) == 0:
                raise ValueError("No valid data found for training after preprocessing.")
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                df_prepared['processed_review'], 
                df_prepared['label'], 
                test_size=0.2, 
                random_state=42, 
                stratify=df_prepared['label']  # Pastikan proporsi label tetap sama
            )
            
            # Validasi data training
            if len(X_train) == 0 or len(y_train) == 0:
                raise ValueError("No training data available after train_test_split.")
            
            self.train(X_train, y_train)
            self.save_model()
            
            logging.info("Model successfully trained and saved!")
        
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise

    def predict(self, texts):
        """Predicts the sentiment for the given texts."""
        processed_texts = [self.preprocess_text(text) for text in texts]
        vectors = self.vectorizer.transform(processed_texts)
        return self.classifier.predict(vectors)

    def save_model(self, vectorizer_path='vectorizer.pkl', classifier_path='classifier.pkl'):
        """Saves the trained model."""
        try:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(classifier_path, 'wb') as f:
                pickle.dump(self.classifier, f)
            logging.info(f"Model saved to {vectorizer_path} and {classifier_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self, vectorizer_path='vectorizer.pkl', classifier_path='classifier.pkl'):
        """Loads a pretrained model if it exists."""
        try:
            if os.path.exists(vectorizer_path) and os.path.exists(classifier_path):
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                logging.info("Model successfully loaded.")
            else:
                raise FileNotFoundError("Model files not found. Please train the model first.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise