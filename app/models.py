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
class Hotel:
    @staticmethod
    def get_all_hotels():
        """Mengambil semua data hotel."""
        with mysql.connection.cursor() as cur:
            cur.execute('SELECT id, name FROM hotels ORDER BY name')
            hotels = cur.fetchall()
            return [{'id': hotel[0], 'name': hotel[1]} for hotel in hotels]
    @staticmethod
    def get_by_id(hotel_id: int):
        """Mengambil data hotel berdasarkan ID."""
        with mysql.connection.cursor() as cur:
            cur.execute('SELECT id, name FROM hotels WHERE id = %s', (hotel_id,))
            hotel = cur.fetchone()
            if hotel:
                return {'id': hotel[0], 'name': hotel[1]}
        return None

    @staticmethod
    def add_hotel(name: str) -> bool:
        """Menambahkan hotel baru."""
        with mysql.connection.cursor() as cur:
            try:
                cur.execute('INSERT INTO hotels (name) VALUES (%s)', (name,))
                mysql.connection.commit()
                return True
            except Exception as e:
                logging.error(f"Error adding hotel: {e}")
                mysql.connection.rollback()
                return False

    @staticmethod
    def update_hotel(hotel_id: int, name: str) -> bool:
        """Mengupdate data hotel."""
        with mysql.connection.cursor() as cur:
            try:
                cur.execute('UPDATE hotels SET name = %s WHERE id = %s', (name, hotel_id))
                mysql.connection.commit()
                return True
            except Exception as e:
                logging.error(f"Error updating hotel: {e}")
                mysql.connection.rollback()
                return False

    @staticmethod
    def delete_hotel(hotel_id: int) -> bool:
        """Menghapus hotel."""
        with mysql.connection.cursor() as cur:
            try:
                cur.execute('DELETE FROM hotels WHERE id = %s', (hotel_id,))
                mysql.connection.commit()
                return True
            except Exception as e:
                logging.error(f"Error deleting hotel: {e}")
                mysql.connection.rollback()
                return False

class Review:
    @staticmethod
    def get_all_reviews() -> list:
        with mysql.connection.cursor() as cur:
            cur.execute('SELECT * FROM reviews ORDER BY review_date DESC')
            return cur.fetchall()
        
    @staticmethod
    def delete_review(review_id: int) -> bool:
        """Menghapus review berdasarkan ID."""
        with mysql.connection.cursor() as cur:
            try:
                cur.execute('DELETE FROM reviews WHERE id = %s', (review_id,))
                mysql.connection.commit()
                return True
            except Exception as e:
                logging.error(f"Error deleting review ID {review_id}: {e}")
                mysql.connection.rollback()
                return False

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
            
    @staticmethod
    def get_paginated_reviews(page, per_page=10):
        offset = (page - 1) * per_page
        cursor = mysql.connection.cursor()
        
        # Updated query to match your table schema
        cursor.execute("""
            SELECT 
                r.id, 
                h.name AS hotel_name, 
                r.guest_name, 
                r.rating, 
                r.review_date,  # Correct column name
                r.review_text, 
                r.sentiment_label  # Correct sentiment column name
            FROM 
                reviews r
            JOIN 
                hotels h ON r.hotel_unit = h.id  # Correct join condition
            ORDER BY 
                r.review_date DESC
            LIMIT %s OFFSET %s
        """, (per_page, offset))
        
        reviews = cursor.fetchall()
        
        # Count total reviews
        cursor.execute("SELECT COUNT(*) FROM reviews")
        total_reviews = cursor.fetchone()[0]
        
        cursor.close()
        return reviews, total_reviews

    @staticmethod
    def get_hotel_reviews_paginated(hotel_unit, page, per_page=10):
        offset = (page - 1) * per_page
        cursor = mysql.connection.cursor()
        
        # Query untuk mendapatkan reviews spesifik untuk hotel unit
        cursor.execute("""
            SELECT 
                r.id, 
                h.name AS hotel_name, 
                r.guest_name, 
                r.rating, 
                r.review_date,  
                r.review_text, 
                r.sentiment_label  
            FROM 
                reviews r
            JOIN 
                hotels h ON r.hotel_unit = h.id
            WHERE 
                r.hotel_unit = %s
            ORDER BY 
                r.review_date DESC
            LIMIT %s OFFSET %s
        """, (hotel_unit, per_page, offset))
        
        reviews = cursor.fetchall()
        
        # Hitung total reviews untuk hotel unit ini
        cursor.execute("SELECT COUNT(*) FROM reviews WHERE hotel_unit = %s", (hotel_unit,))
        total_reviews = cursor.fetchone()[0]
        
        cursor.close()
        return reviews, total_reviews
    
    @staticmethod
    def get_overall_sentiment_distribution():
        """Menghitung distribusi sentimen untuk semua review."""
        with mysql.connection.cursor() as cur:
            cur.execute('''
                SELECT 
                    CASE 
                        WHEN sentiment_label IS NULL THEN 'Neutral'
                        ELSE sentiment_label 
                    END AS sentiment, 
                    COUNT(*) as count 
                FROM reviews 
                GROUP BY sentiment
            ''')
            result = cur.fetchall()
            
            # Konversi ke dictionary dengan key 'Positive', 'Negative', 'Neutral'
            sentiment_dict = {}
            for sentiment, count in result:
                sentiment_dict[sentiment.capitalize()] = count
            
            return sentiment_dict

    @staticmethod
    def get_hotel_sentiment_distribution(hotel_unit: str):
        """Mengambil distribusi sentimen untuk hotel tertentu tanpa filter tanggal."""
        with mysql.connection.cursor() as cur:
            cur.execute('''
                SELECT sentiment_label, COUNT(*) as count 
                FROM reviews 
                WHERE sentiment_label IS NOT NULL 
                AND hotel_unit = %s
                GROUP BY sentiment_label
            ''', (hotel_unit,))

            result = cur.fetchall()

            return dict(result) if result else {"positive": 0, "negative": 0, "neutral": 0}



    @staticmethod
    def get_rating_distribution():
        """Menghitung distribusi rating untuk semua review."""
        with mysql.connection.cursor() as cur:
            cur.execute('''
                SELECT rating, COUNT(*) as count 
                FROM reviews 
                GROUP BY rating 
                ORDER BY rating
            ''')
            return dict(cur.fetchall())

    @staticmethod
    def get_hotel_rating_distribution(hotel_unit: str):
        """Menghitung distribusi rating untuk unit hotel tertentu."""
        with mysql.connection.cursor() as cur:
            cur.execute('''
                SELECT rating, COUNT(*) as count 
                FROM reviews 
                WHERE hotel_unit = %s 
                GROUP BY rating 
                ORDER BY rating
            ''', (hotel_unit,))
            return dict(cur.fetchall())
        
    @staticmethod
    def get_filtered_hotel_reviews_by_date(hotel_unit, start_date, end_date, page, per_page=10):
        offset = (page - 1) * per_page
        with mysql.connection.cursor() as cur:
            cur.execute('''
                SELECT * FROM reviews 
                WHERE hotel_unit = %s AND review_date BETWEEN %s AND %s 
                ORDER BY review_date DESC 
                LIMIT %s OFFSET %s
            ''', (hotel_unit, start_date, end_date, per_page, offset))
            reviews = cur.fetchall()
            cur.execute('''
                SELECT COUNT(*) FROM reviews 
                WHERE hotel_unit = %s AND review_date BETWEEN %s AND %s
            ''', (hotel_unit, start_date, end_date))
            total_reviews = cur.fetchone()[0]
        return reviews, total_reviews

    @staticmethod
    def get_filtered_hotel_sentiment_distribution(hotel_unit, start_date, end_date):
        """
        Menghitung total sentimen (Positive, Negative, Neutral) untuk hotel tertentu berdasarkan tanggal.
        """
        with mysql.connection.cursor() as cur:
            cur.execute('''
                SELECT 
                    CASE 
                        WHEN sentiment_label IS NULL THEN 'Neutral'
                        ELSE sentiment_label 
                    END AS sentiment, 
                    COUNT(*) as count 
                FROM reviews 
                WHERE hotel_unit = %s AND review_date BETWEEN %s AND %s
                GROUP BY sentiment
            ''', (hotel_unit, start_date, end_date))
            result = cur.fetchall()

            # Konversi hasil ke dictionary dengan key default
            sentiment_dict = {
                'Positive': 0,
                'Negative': 0,
                'Neutral': 0,
            }
            for sentiment, count in result:
                sentiment_dict[sentiment.capitalize()] = count

            return sentiment_dict


    @staticmethod
    def get_filtered_hotel_rating_distribution(hotel_unit, start_date, end_date):
        with mysql.connection.cursor() as cur:
            cur.execute('''
                SELECT rating, COUNT(*) as count 
                FROM reviews 
                WHERE hotel_unit = %s AND review_date BETWEEN %s AND %s
                GROUP BY rating 
                ORDER BY rating
            ''', (hotel_unit, start_date, end_date))
            return dict(cur.fetchall())
        
    
    