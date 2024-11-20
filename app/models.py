from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from . import mysql
import logging

class User(UserMixin):
    def __init__(self, id: int, username: str, role: str, hotel_unit: str):
        self.id = id
        self.username = username
        self.role = role
        self.hotel_unit = hotel_unit

    @staticmethod
    def get_by_id(user_id: int) -> 'User':
        with mysql.connection.cursor() as cur:
            cur.execute('SELECT id, username, role, hotel_unit FROM users WHERE id = %s', (user_id,))
            user = cur.fetchone()
            
            if user:
                return User(user[0], user[1], user[2], user[3])
        return None

    @staticmethod
    def get_by_username(username: str) -> dict:
        with mysql.connection.cursor() as cur:
            cur.execute('SELECT id, username, password, role, hotel_unit FROM users WHERE username = %s', (username,))
            user = cur.fetchone()
            
            if user:
                return {'id': user[0], 'username': user[1], 'password': user[2], 'role': user[3], 'hotel_unit': user[4]}
        return None

    @staticmethod
    def create(username: str, password: str, role: str, hotel_unit: str) -> bool:
        hashed_password = generate_password_hash(password)
        with mysql.connection.cursor() as cur:
            try:
                cur.execute('''
                    INSERT INTO users (username, password, role, hotel_unit) 
                    VALUES (%s, %s, %s, %s)
                ''', (username, hashed_password, role, hotel_unit))
                mysql.connection.commit()
                return True
            except Exception as e:
                logging.error(f"Error adding user: {e}")
                mysql.connection.rollback()
                return False

class Review:
    @staticmethod
    def get_all_reviews() -> list:
        with mysql.connection.cursor() as cur:
            cur.execute('SELECT * FROM reviews ORDER BY review_date DESC')
            return cur.fetchall()

    @staticmethod
    def get_hotel_reviews(hotel_unit: str) -> list:
        with mysql.connection.cursor() as cur:
            cur.execute('SELECT * FROM reviews WHERE hotel_unit = %s ORDER BY review_date DESC', (hotel_unit,))
            return cur.fetchall()

    @staticmethod
    def get_review_by_content(content: str) -> dict:
        with mysql.connection.cursor() as cur:
            cur.execute('SELECT * FROM reviews WHERE review_text = %s', (content,))
            return cur.fetchone()

    @staticmethod
    def check_if_exists(review_text: str) -> bool:
        """ Cek apakah review sudah ada di dalam database. """
        with mysql.connection.cursor() as cur:
            cur.execute('SELECT 1 FROM reviews WHERE review_text = %s', (review_text,))
            return cur.fetchone() is not None

    @staticmethod
    @staticmethod
    def add_review(hotel_unit: str, guest_name: str, rating: int, review_date: str, review_text: str, sentiment_label: str) -> bool:
        # Validasi: Rating harus ada dan dalam rentang 1-10
        if rating is None or rating < 1 or rating > 10:
            logging.warning(f"Rating is required and must be between 1 and 10. Review not added for: {review_text}")
            return False  # Tidak menambahkan review jika rating tidak valid

        # Pastikan review_text tidak duplikat jika diisi
        if review_text and Review.check_if_exists(review_text):
            logging.warning(f"Review already exists: {review_text}")
            return False  # Jika review sudah ada, tidak menambahkannya

        # Siapkan data untuk disimpan
        hotel_unit = hotel_unit if hotel_unit else None
        guest_name = guest_name if guest_name else None
        review_text = review_text if review_text else ""  # Tetap gunakan string kosong jika review_text tidak ada
        sentiment_label = sentiment_label if sentiment_label else None

        with mysql.connection.cursor() as cur:
            try:
                cur.execute('''
                    INSERT INTO reviews (hotel_unit, guest_name, rating, review_date, review_text, sentiment_label)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (hotel_unit, guest_name, rating, review_date, review_text, sentiment_label))
                mysql.connection.commit()
                return True
            except Exception as e:
                logging.error(f"Error adding review: {e}")
                mysql.connection.rollback()
                return False


