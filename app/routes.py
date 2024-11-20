from flask import render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import os
import pandas as pd
import logging
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
    
    @app.route('/add_hotel', methods=['GET', 'POST'])
    @login_required
    def add_hotel():
        if current_user.role != 'admin':
            flash('Anda tidak memiliki izin untuk menambahkan hotel.')
            return redirect(url_for('dashboard'))

        if request.method == 'POST':
            hotel_name = request.form['name']
            if hotel_name:
                cur = mysql.connection.cursor()
                cur.execute('INSERT INTO hotels (name) VALUES (%s)', (hotel_name,))
                mysql.connection.commit()
                cur.close()
                flash('Hotel berhasil ditambahkan.')
                return redirect(url_for('manage_hotels'))

        return render_template('add_hotel.html')# Pastikan ini berada di luar if di atas

    @app.route('/manage_hotels', methods=['GET'])
    @login_required
    def manage_hotels():
        if current_user.role != 'admin':
            flash('Anda tidak memiliki izin untuk mengelola hotel.')
            return redirect(url_for('dashboard'))

        # Mendapatkan semua hotel
        cur = mysql.connection.cursor()
        cur.execute('SELECT * FROM hotels')
        hotels = cur.fetchall()
        cur.close()
        
        return render_template('manage_hotels.html', hotels=hotels)

    @app.route('/edit_hotel/<int:hotel_id>', methods=['GET', 'POST'])
    @login_required
    def edit_hotel(hotel_id):
        if current_user.role != 'admin':
            flash('Anda tidak memiliki izin untuk mengedit hotel.')
            return redirect(url_for('dashboard'))

        cur = mysql.connection.cursor()

        if request.method == 'POST':
            new_name = request.form['name']
            cur.execute('UPDATE hotels SET name = %s WHERE id = %s', (new_name, hotel_id))
            mysql.connection.commit()
            cur.close()
            flash('Hotel berhasil diperbarui.')
            return redirect(url_for('manage_hotels'))
        
        cur.execute('SELECT * FROM hotels WHERE id = %s', (hotel_id,))
        hotel = cur.fetchone()
        cur.close()
        
        return render_template('edit_hotel.html', hotel=hotel)

    @app.route('/delete_hotel/<int:hotel_id>', methods=['POST'])
    @login_required
    def delete_hotel(hotel_id):
        if current_user.role != 'admin':
            flash('Anda tidak memiliki izin untuk menghapus hotel.')
            return redirect(url_for('dashboard'))

        cur = mysql.connection.cursor()
        cur.execute('DELETE FROM hotels WHERE id = %s', (hotel_id,))
        mysql.connection.commit()
        cur.close()
        flash('Hotel berhasil dihapus.')
        return redirect(url_for('manage_hotels'))
    
    @app.route('/add_hotel_unit', methods=['GET', 'POST'])
    @login_required
    def add_hotel_unit():
        if current_user.role != 'admin':
            flash('Anda tidak memiliki izin untuk menambahkan hotel unit.')
            return redirect(url_for('dashboard'))

        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            hotel_unit = request.form['hotel_unit']

            if User.create(username, password, 'unit_hotel', hotel_unit):
                flash('Akun hotel unit berhasil ditambahkan.')
                return redirect(url_for('dashboard'))

            flash('Terjadi kesalahan saat pendaftaran. Silakan coba lagi.')

        return render_template('add_hotel_unit.html')
    
    
    @app.route('/manage_hotel_accounts', methods=['GET'])
    @login_required
    def manage_hotel_accounts():
        if current_user.role != 'admin':
            flash('Anda tidak memiliki izin untuk mengelola akun hotel unit.')
            return redirect(url_for('dashboard'))
        
        # Mendapatkan semua akun hotel unit dari database
        cur = mysql.connection.cursor()
        cur.execute('SELECT id, username, hotel_unit FROM users WHERE role = %s', ('unit_hotel',))
        hotel_accounts = cur.fetchall()  # Ini akan mengembalikan list of tuples
        cur.close()
        return render_template('manage_hotel_accounts.html', hotel_accounts=hotel_accounts)

    @app.route('/edit_hotel_unit/<int:user_id>', methods=['GET', 'POST'])
    @login_required
    def edit_hotel_unit(user_id):
        if current_user.role != 'admin':
            flash('Anda tidak memiliki izin untuk mengedit akun hotel unit.')
            return redirect(url_for('dashboard'))

        cur = mysql.connection.cursor()

        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']  # Ganti password jika ada

            # Update username dan password jika tidak kosong
            if username:
                cur.execute('UPDATE users SET username = %s WHERE id = %s', (username, user_id))
            
            if password:
                hashed_password = generate_password_hash(password)
                cur.execute('UPDATE users SET password = %s WHERE id = %s', (hashed_password, user_id))

            mysql.connection.commit()
            cur.close()
            flash('Akun hotel unit berhasil diperbarui.')
            return redirect(url_for('manage_hotel_accounts'))

        # Mengambil data pengguna untuk ditampilkan di form
        cur.execute('SELECT id, username FROM users WHERE id = %s', (user_id,))
        user = cur.fetchone()
        cur.close()

        if not user:
            flash('Akun tidak ditemukan.')
            return redirect(url_for('manage_hotel_accounts'))

        return render_template('edit_hotel_unit.html', user=user)
    
    
    @app.route('/delete_hotel_unit/<int:user_id>', methods=['POST'])
    @login_required
    def delete_hotel_unit(user_id):
        if current_user.role != 'admin':
            flash('Anda tidak memiliki izin untuk menghapus akun hotel unit.')
            return redirect(url_for('dashboard'))

        cur = mysql.connection.cursor()
        try:
            cur.execute('DELETE FROM users WHERE id = %s', (user_id,))
            mysql.connection.commit()
            flash('Akun hotel unit berhasil dihapus.')
        except Exception as e:
            mysql.connection.rollback()  # Rollback jika ada kesalahan
            flash('Terjadi kesalahan saat menghapus akun hotel unit.')

        cur.close()
        return redirect(url_for('manage_hotel_accounts'))



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
                # Ambil kolom review
                review = row['review'] if pd.notna(row['review']) else ""  # Tetapkan ke string kosong jika tidak ada review
                rating = row['rating'] if pd.notna(row['rating']) else None
                # Validasi rating
                if rating is not None and (rating < 1 or rating > 10):
                    logging.warning(f"Review skipped due to invalid rating: {rating}")
                    continue  # Lewati entri ini jika rating tidak valid
                
                cleaned_review = sentiment_analyzer.preprocess_text(review)  # Bersihkan ulasan
                sentiment = sentiment_analyzer.predict([cleaned_review])[0]  # Dapatkan data prediksi sentimen

                Review.add_review(
                    row['hotel_unit'] if pd.notna(row['hotel_unit']) else None,
                    row['name'] if pd.notna(row['name']) else None,
                    rating,  # Menggunakan rating yang sudah divalidasi
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
