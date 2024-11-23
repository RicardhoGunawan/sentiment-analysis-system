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
import string

# Download required NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

class SentimentAnalyzer:
    def __init__(self):
        # Gunakan stemmer dari Sastrawi
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        # Gunakan stopwords dari Sastrawi
        stopword_factory = StopWordRemoverFactory()
        self.stop_words = set(stopword_factory.get_stop_words())

        self.vectorizer = CountVectorizer()
        self.classifier = SVC(kernel='linear')

        # Kamus kata positif dan negatif bahasa Indonesia
        self.positive_words = set([
            'bagus', 'baik', 'suka', 'senang', 'puas', 'mantap', 'keren', 'enak', 'lembut',
            'recommended', 'rekomen', 'memuaskan', 'ramah', 'cepat', 'bersih', 'nyaman', 
            'top', 'oke', 'ok', 'worth', 'mantul', 'megah', 'indah', 'cantik', 'luas', 
            'strategis', 'mewah', 'murah', 'terjangkau', 'aman', 'hangat', 'asri', 
            'teratur', 'rapi', 'seru', 'friendly', 'helpful', 'tenang', 'privasi', 
            'istimewa', 'sopan', 'terbaik', 'istirahat', 'lega', 'relax', 'lux', 'perfect', 
            'nikmat', 'nyaman', 'senyap', 'sempurna', 'worth it', 'ramah lingkungan', 
            'fresh', 'profesional', 'bersahabat', 'inspiratif', 'kekinian', 'hebat', 'canggih','alternatif'
        ])
        
        self.negative_words = set([
            'buruk', 'jelek', 'kecewa', 'mahal', 'lambat', 'kotor', 'kasar', 'jorok', 
            'bising', 'ribut', 'berisik', 'mati', 'bau', 'rusak', 'kecoa', 'lalat', 
            'mengerikan', 'seram', 'hancur', 'busuk', 'basi', 'lecet', 'panas', 
            'sempit', 'tidak nyaman', 'tidak bersih', 'tidak aman', 'gelap', 'dingin', 
            'susah', 'membingungkan', 'menakutkan', 'terlalu mahal', 'tidak ramah', 
            'tidak profesional', 'pelit', 'berantakan', 'penuh', 'kurang baik', 
            'tidak memuaskan', 'tidak sopan', 'mengganggu', 'tidak layak', 
            'tidak terawat', 'lelet', 'melelahkan', 'tidak sesuai', 'penipuan', 
            'membosankan', 'menyesal', 'kuno', 'jadul', 'basi', 'tidak higienis', 
            'parah', 'kejam', 'terisolasi', 'tidak ramah lingkungan'
        ])

    def preprocess_text(self, text):
        """Preprocesses input text for sentiment analysis."""
        if isinstance(text, str):
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

            # Tokenizing (memecah kalimat menjadi kata-kata)
            tokens = word_tokenize(text)

            # Normalization (misalnya mengganti kata tidak baku dengan yang baku)
            normalization_dict = {
                'gak': 'tidak',
                'nggak': 'tidak',
                'ga': 'tidak',
                'aja': 'saja',
                'kok': 'tidak',
                'dong': '',
                'lah': 'sudah',
                'memakan': 'makan',
                'kmarin': 'kemarin',
                'cocok' : 'bagus',
                'muas' : 'puas',
                'sarap' : 'sarapan',
                'alternativ' : 'alternatif',
            }
            tokens = [normalization_dict.get(token, token) for token in tokens]

            # Stopwords removal (gunakan Sastrawi)
            tokens = [token for token in tokens if token not in self.stop_words]

            # Stemming (mengubah kata ke bentuk dasar dengan Sastrawi)
            tokens = [self.stemmer.stem(token) for token in tokens]

            # Gabungkan kembali token menjadi string
            return ' '.join(tokens)
        else:
            print(f"Warning: Expected string input but got {type(text)}.")
            return ''


    def get_sentiment_label(self, text):
        """Get sentiment label from text using custom Indonesian sentiment dictionary."""
        if not isinstance(text, str):
            return None
            
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
            return None

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
        # Gabungkan X_train dan y_train untuk menjaga konsistensi saat pembersihan
        combined_data = [(x, y) for x, y in zip(X_train, y_train) if y is not None and not pd.isnull(y)]
        
        # Pisahkan kembali menjadi X_train dan y_train setelah pembersihan
        X_train, y_train = zip(*combined_data)
        
        # Konversi kembali ke list jika dibutuhkan
        X_train = list(X_train)
        y_train = list(y_train)

        # Validasi panjang X_train dan y_train
        if not len(X_train) == len(y_train):
            raise ValueError("Jumlah X_train dan y_train tidak cocok setelah pembersihan.")
        
        # Transform data fitur dengan vectorizer
        X_train_vectors = self.vectorizer.fit_transform(X_train)

        # Latih classifier
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