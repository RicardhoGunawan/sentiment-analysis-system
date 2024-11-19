from flask import render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
import os
import pandas as pd
from . import mysql, login_manager
from .models import User, Review
from .sentiment import SentimentAnalyzer  # Import kelas SentimentAnalyzer

def init_routes(app):
    sentiment_analyzer = SentimentAnalyzer()  # Inisialisasi SentimentAnalyzer

    @login_manager.user_loader
    def load_user(user_id):
        return User.get_by_id(user_id)

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']

            user_data = User.get_by_username(username)
            if user_data and check_password_hash(user_data['password'], password):
                user = User(user_data['id'], user_data['username'], user_data['role'], user_data['hotel_unit'])
                login_user(user)
                return redirect(url_for('dashboard'))

            flash('Invalid username or password')
        return render_template('auth/login.html')

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if current_user.is_authenticated:
            flash('Anda sudah terdaftar. Silakan login.')
            return redirect(url_for('dashboard'))

        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            role = request.form['role']
            hotel_unit = request.form['hotel_unit']

            if User.create(username, password, role, hotel_unit):
                flash('Pengguna berhasil ditambahkan.')
                return redirect(url_for('login'))

            flash('Terjadi kesalahan saat pendaftaran. Silakan coba lagi.')

        return render_template('auth/register.html')

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('login'))

    @app.route('/')
    @login_required
    def dashboard():
        if current_user.role == 'admin':
            reviews = Review.get_all_reviews()
        else:
            reviews = Review.get_hotel_reviews(current_user.hotel_unit)
        return render_template('dashboard.html', reviews=reviews)

    @app.route('/upload', methods=['GET', 'POST'])
    @login_required
    def upload():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file selected')
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                flash('No file selected')
                return redirect(request.url)

            if file:
                # Use secure filename and save to static uploads folder
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # UPLOAD_FOLDER harus menunjuk ke app/static/uploads
                file.save(filepath)

                # Baca file
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath, sep=';')
                else:
                    df = pd.read_excel(filepath)

                # Proses ulasan dan melatih model
                sentiment_analyzer.train_and_save(df)  # Latih dan simpan model

                # Simpan review ke database
                for _, row in df.iterrows():
                    review = row['review'] if pd.notna(row['review']) else None
                    if review:
                        cleaned_review = sentiment_analyzer.preprocess_text(review)  # Bersihkan ulasan
                        sentiment = sentiment_analyzer.predict([cleaned_review])[0]  # Dapatkan prediksi sentimen
                        
                        Review.add_review(
                            row['hotel_unit'] if pd.notna(row['hotel_unit']) else None,
                            row['name'] if pd.notna(row['name']) else None,
                            row['rating'] if pd.notna(row['rating']) else None,
                            row['date'] if pd.notna(row['date']) else None,
                            cleaned_review,  # Gunakan teks ulasan yang telah dibersihkan
                            sentiment  # Gunakan prediksi sentimen
                        )

                os.remove(filepath)  # Bersihkan file yang di-upload
                flash('File uploaded and processed successfully')
                return redirect(url_for('dashboard'))

        return render_template('upload.html')

    @app.route('/reviews')
    @login_required
    def reviews():
        if current_user.role == 'admin':
            reviews = Review.get_all_reviews()
        else:
            reviews = Review.get_hotel_reviews(current_user.hotel_unit)
        return render_template('reviews.html', reviews=reviews)

    @app.route('/update_sentiment', methods=['POST'])
    @login_required
    def update_sentiment():
        data = request.get_json()
        review_id = data.get('review_id')
        sentiment = data.get('sentiment')

        cur = mysql.connection.cursor()
        cur.execute('UPDATE reviews SET sentiment_label = %s WHERE id = %s', (sentiment, review_id))
        mysql.connection.commit()
        cur.close()

        return jsonify({'success': True})
