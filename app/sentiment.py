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
            'nyaman', 'top', 'oke', 'ok', 'recommended', 'worth', 'mantul','memuaskan'
        ])
        
        self.negative_words = set([
            'buruk', 'jelek', 'kecewa', 'mahal', 'lambat', 'kotor', 'kasar',
            'tidak', 'jangan', 'hancur', 'rusak', 'busuk', 'mahal', 'kurang',
            'cacad', 'rugi', 'bau', 'pahit', 'basi', 'lecet','tolong','kecoa',
            'mempersulit','kurang','jorok','mati'
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
            text = re.compile(r'(\d{1,2})?(st|nd|rd|th)?[-./,]?\s?(of)?\s?([J|j]an(uary)?|[F|f]eb(ruary)?|[Mm]ar(ch)?|[Aa]pr(il)?|[Mm]ay|[Jj]un(e)?|[Jj]ul(y)?|[Aa]ug(ust)?|[Ss]ep(tember)?|[Oo]ct(ober)?|[Nn]ov(ember)?|[Dd]ec(ember)?)\s?(\d{1,2})?(st|nd|rd|th)?\s?[-./,]?\s?(\d{2,4})?'
                    ).sub('', text)

            # 5. HTML tags removal
            text = BeautifulSoup(text, 'html.parser').get_text()

            # 6. Emojis removal
            text = re.sub(r"["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags
                        u"\U00002500-\U00002BEF"  # chinese char
                        u"\U00002702-\U000027B0"
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        u"\U0001f926-\U0001f937"
                        u"\U00010000-\U0010ffff"
                        u"\u2640-\u2642"
                        u"\u2600-\u2B55"
                        u"\u200d"
                        u"\u23cf"
                        u"\u23e9"
                        u"\u231a"
                        u"\ufe0f"  # dingbats
                        u"\u3030"
                        "]+", '', text)

            # 7. Remove emoticons based on the provided dictionary
            EMOTICONS = {
                u":‑$$": "Happy face or smiley",
                u":$$": "Happy face or smiley",
                u":-$$": "Happy face or smiley",
                u":$$": "Happy face or smiley",
                u":-3": "Happy face smiley",
                u":3": "Happy face smiley",
                u":->": "Happy face smiley",
                u":>": "Happy face smiley",
                u"8-$$": "Happy face smiley",
                u":o$$": "Happy face smiley",
                u":-\\}": "Happy face smiley",  # Memperbaiki escape
                u":\\}": "Happy face smiley",   # Memperbaiki escape
                u":-D": "Laughing, big grin or laugh with glasses",
                u":D": "Laughing, big grin or laugh with glasses",
                u"8‑D": "Laughing, big grin or laugh with glasses",
                u"8D": "Laughing, big grin or laugh with glasses",
                u"X‑D": "Laughing, big grin or laugh with glasses",
                u"XD": "Laughing, big grin or laugh with glasses",
                # Lanjutkan untuk emotikon lainnya, pastikan semua telah diperbaiki
                u":-\\|": "Frown, sad, angry or pouting",
                u">:\$$": "Frown, sad, angry or pouting",
                u":\\{": "Frown, sad, angry or pouting",
                u":@": "Frown, sad, angry or pouting",
                            # Tambahkan emoticons lain yang sesuai
            }
            text = re.sub(u'(' + u'|'.join(emo for emo in EMOTICONS) + u')', '', text)

            # 8. Hashtags and mentions removal
            text = re.sub(r'(@\S+|#\S+)', '', text)

            # 9. Punctuation removal
            text = text.translate(str.maketrans('', '', string.punctuation))

            # 10. Number removal
            text = re.sub(r'\d+', '', text)

            # 11. Extra whitespaces removal
            text = ' '.join(text.split())

            # 12. Tokenization and stemming (jika diperlukan)
            tokens = word_tokenize(text)  # Pastikan Anda memiliki NLTK dengan tokenizer terinstal
            tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
            
            # Gabungkan kembali token menjadi string
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