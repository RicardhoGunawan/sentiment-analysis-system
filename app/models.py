from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from . import mysql

class User(UserMixin):
    def __init__(self, id, username, role, hotel_unit):
        self.id = id
        self.username = username
        self.role = role
        self.hotel_unit = hotel_unit

    @staticmethod
    def get_by_id(user_id):
        cur = mysql.connection.cursor()
        cur.execute('SELECT id, username, role, hotel_unit FROM users WHERE id = %s', (user_id,))
        user = cur.fetchone()
        cur.close()
        
        if user:
            return User(user[0], user[1], user[2], user[3])
        return None

    @staticmethod
    def get_by_username(username):
        cur = mysql.connection.cursor()
        cur.execute('SELECT id, username, password, role, hotel_unit FROM users WHERE username = %s', (username,))
        user = cur.fetchone()
        cur.close()
        
        if user:
            return {'id': user[0], 'username': user[1], 'password': user[2], 'role': user[3], 'hotel_unit': user[4]}
        return None

    @staticmethod
    def create(username, password, role, hotel_unit):
        hashed_password = generate_password_hash(password)  # Meng-hash password
        cur = mysql.connection.cursor()
        try:
            cur.execute('''
                INSERT INTO users (username, password, role, hotel_unit) 
                VALUES (%s, %s, %s, %s)
            ''', (username, hashed_password, role, hotel_unit))
            mysql.connection.commit()
            return True  # Pengguna ditambahkan dengan sukses
        except Exception as e:
            print(f"Terjadi kesalahan saat menambahkan pengguna: {e}")
            return False  # Pengguna gagal ditambahkan
        finally:
            cur.close()

class Review:
    @staticmethod
    def get_all_reviews():
        cur = mysql.connection.cursor()
        cur.execute('SELECT * FROM reviews ORDER BY review_date DESC')
        reviews = cur.fetchall()
        cur.close()
        return reviews

    @staticmethod
    def get_hotel_reviews(hotel_unit):
        cur = mysql.connection.cursor()
        cur.execute('SELECT * FROM reviews WHERE hotel_unit = %s ORDER BY review_date DESC', (hotel_unit,))
        reviews = cur.fetchall()
        cur.close()
        return reviews

    @staticmethod
    def get_review_by_content(content):
        """Check if a review with the same content already exists."""
        cur = mysql.connection.cursor()
        cur.execute('SELECT * FROM reviews WHERE review_text = %s', (content,))
        result = cur.fetchone()
        cur.close()
        return result

    @staticmethod
    def add_review(hotel_unit, guest_name, rating, review_date, review_text, sentiment_label):
        """Add a review to the database, making sure it doesn't already exist."""
        # Check for existing review
        existing_review = Review.get_review_by_content(review_text)
        if existing_review:
            print(f"Review already exists: {review_text}")
            return False  # Return false if the review already exists

        cur = mysql.connection.cursor()
        try:
            cur.execute('''
                INSERT INTO reviews (hotel_unit, guest_name, rating, review_date, review_text, sentiment_label)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (hotel_unit, guest_name, rating, review_date, review_text, sentiment_label))
            mysql.connection.commit()
            return True  # Review added successfully
        except Exception as e:
            print(f"Terjadi kesalahan saat menambahkan review: {e}")
            mysql.connection.rollback()  # Rollback on error
            return False  # Review failed to add
        finally:
            cur.close()
