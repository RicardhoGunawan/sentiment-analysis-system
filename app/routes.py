from flask import render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import os
import json
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from . import mysql, login_manager
from .models import User, Review, Hotel  
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
         # Inisialisasi SentimentAnalyzer
        analyzer = SentimentAnalyzer.get_instance()
        page = request.args.get('page', default=1, type=int)
        per_page = 10
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        selected_hotel_id = request.args.get('hotel_id', type=int)

        # Validasi nilai `page`
        if page < 1:
            page = 1

        # Validasi format tanggal
        def validate_date(date_str):
            try:
                return datetime.strptime(date_str, '%Y-%m-%d').date()
            except (ValueError, TypeError):
                return None

        start_date = validate_date(start_date)
        end_date = validate_date(end_date)

        if start_date and end_date and start_date > end_date:
            start_date, end_date = None, None

        sentiment_distribution = Review.get_overall_sentiment_distribution()
        hotels = []

        if current_user.role == 'admin':
            hotels = Hotel.get_all_hotels()

      # Ambil data berdasarkan role pengguna
        # Ambil data berdasarkan role pengguna
        if current_user.role == 'admin':
            if selected_hotel_id:
                # Jika admin memilih hotel tertentu
                if start_date or end_date:
                    # Dengan filter tanggal
                    reviews, total_reviews = Review.get_filtered_hotel_reviews_by_date(
                        selected_hotel_id, start_date, end_date, page
                    )
                    sentiment_distribution = Review.get_filtered_hotel_sentiment_distribution(
                        selected_hotel_id, start_date, end_date
                    )
                    rating_distribution = Review.get_filtered_hotel_rating_distribution(
                        selected_hotel_id, start_date, end_date
                    )
                else:
                    # Tanpa filter tanggal, tampilkan total keseluruhan sentimen
                    reviews, total_reviews = Review.get_hotel_reviews_paginated(selected_hotel_id, page)
                    sentiment_distribution = Review.get_hotel_sentiment_distribution(selected_hotel_id)
                    rating_distribution = Review.get_hotel_rating_distribution(selected_hotel_id)
            else:
                # Jika admin tidak memilih hotel tertentu
                if start_date or end_date:
                    # Dengan filter tanggal
                    reviews, total_reviews = Review.get_filtered_reviews_by_date(start_date, end_date, page)
                    sentiment_distribution = Review.get_filtered_sentiment_distribution(start_date, end_date)
                    rating_distribution = Review.get_filtered_rating_distribution(start_date, end_date)
                else:
                    # Tanpa filter tanggal
                    reviews, total_reviews = Review.get_paginated_reviews(page)
                    sentiment_distribution = Review.get_overall_sentiment_distribution()
                    rating_distribution = Review.get_rating_distribution()
        else:
            # Jika role adalah user biasa
            if start_date or end_date:
                # Dengan filter tanggal
                reviews, total_reviews = Review.get_filtered_hotel_reviews_by_date(
                    current_user.hotel_unit, start_date, end_date, page
                )
                sentiment_distribution = Review.get_filtered_hotel_sentiment_distribution(
                    current_user.hotel_unit, start_date, end_date
                )
                rating_distribution = Review.get_filtered_hotel_rating_distribution(
                    current_user.hotel_unit, start_date, end_date
                )
            else:
                # Tanpa filter tanggal
                reviews, total_reviews = Review.get_hotel_reviews_paginated(current_user.hotel_unit, page)
                sentiment_distribution = Review.get_hotel_sentiment_distribution(current_user.hotel_unit)
                rating_distribution = Review.get_hotel_rating_distribution(current_user.hotel_unit)




        # Debug: Log nilai distribusi sentimen
        logging.debug(f'Sentiment Distribution: {sentiment_distribution}')

        # Pastikan distribusi tidak kosong
        sentiment_distribution_keys = list(sentiment_distribution.keys()) if sentiment_distribution else []
        sentiment_distribution_values = list(sentiment_distribution.values()) if sentiment_distribution else []
        rating_distribution_keys = list(rating_distribution.keys()) if rating_distribution else []
        rating_distribution_values = list(rating_distribution.values()) if rating_distribution else []

        total_pages = (total_reviews + per_page - 1) // per_page
        
      

        # Pisahkan ulasan berdasarkan sentimen
        positive_reviews = []
        negative_reviews = []

        if reviews:
            for review in reviews:
                sentiment = analyzer.get_sentiment_label(review[5])  # Asumsikan review[5] adalah teks ulasan
                if sentiment == "positive":
                    positive_reviews.append(review[5])
                elif sentiment == "negative":
                    negative_reviews.append(review[5])

        # Buat WordCloud untuk ulasan positif dan negatif
        positive_wordcloud_base64 = (
            analyzer.create_wordcloud_base64(positive_reviews) if positive_reviews else None
        )
        negative_wordcloud_base64 = (
            analyzer.create_wordcloud_base64(negative_reviews) if negative_reviews else None
        )

        # Periksa apakah permintaan adalah AJAX
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'reviews_html': render_template('partials/review_rows_dashboard.html', reviews=reviews),
                'pagination_html': render_template('partials/pagination.html', current_page=page, total_pages=total_pages),
                'showing_info': f"Showing {len(reviews)} out of {total_reviews} reviews",
                'sentiment_data': sentiment_distribution,
                'rating_data': rating_distribution
            })

        # Render template dengan semua data yang diperlukan
        return render_template('dashboard.html', 
                            hotels=hotels,  
                            selected_hotel_id=selected_hotel_id,  
                            reviews=reviews, 
                            current_page=page, 
                            total_reviews=total_reviews, 
                            total_pages=total_pages,
                            sentiment_distribution_keys=sentiment_distribution_keys,
                            sentiment_distribution_values=sentiment_distribution_values,
                            sentiment_distribution=sentiment_distribution,
                            rating_distribution_keys=rating_distribution_keys,
                            rating_distribution_values=rating_distribution_values,
                            positive_wordcloud_image=positive_wordcloud_base64,  # WordCloud positif
                            negative_wordcloud_image=negative_wordcloud_base64,
                            loading=True,
                            )
        

        
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
        # Cek apakah pengguna adalah admin
        if current_user.role != 'admin':
            flash('Anda tidak memiliki izin untuk menambahkan hotel unit.', 'danger')
            return redirect(url_for('dashboard'))

        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            hotel_unit = request.form.get('hotel_unit')

            # Validasi input
            if not username or not password or not hotel_unit:
                flash('Semua kolom harus diisi.', 'warning')
                return redirect(url_for('add_hotel_unit'))

            # Validasi ID hotel
            hotel = Hotel.get_by_id(hotel_unit)
            if not hotel:
                flash('Hotel unit tidak valid.', 'danger')
                return redirect(url_for('add_hotel_unit'))

            # Simpan akun unit hotel
            if User.create(username, password, 'unit_hotel', hotel_unit):
                flash(f'Akun hotel unit untuk {hotel["name"]} berhasil ditambahkan.', 'success')
                return redirect(url_for('manage_hotel_accounts'))

            flash('Terjadi kesalahan saat pendaftaran. Silakan coba lagi.', 'danger')

        # Ambil daftar hotel untuk dropdown di form
        hotels = Hotel.get_all_hotels()
        return render_template('add_hotel_unit.html', hotels=hotels)

    
    
    @app.route('/manage_hotel_accounts', methods=['GET'])
    @login_required
    def manage_hotel_accounts():
        if current_user.role != 'admin':
            flash('Anda tidak memiliki izin untuk mengelola akun hotel unit.', 'danger')
            return redirect(url_for('dashboard'))

        with mysql.connection.cursor() as cur:
            cur.execute(
                '''SELECT u.id, u.username, h.id AS hotel_id, h.name AS hotel_name 
                FROM users u 
                JOIN hotels h ON u.hotel_unit = h.id 
                WHERE u.role = %s''', 
                ('unit_hotel',)
            )
            hotel_accounts = cur.fetchall()

        # Ambil semua data hotel untuk dropdown
        hotels = Hotel.get_all_hotels()
        
        return render_template('manage_hotel_accounts.html', 
                            hotel_accounts=hotel_accounts, 
                            hotels=hotels)


    @app.route('/edit_hotel_unit/<int:user_id>', methods=['GET', 'POST'])
    @login_required
    def edit_hotel_unit(user_id):
        if current_user.role != 'admin':
            flash('Anda tidak memiliki izin untuk mengedit akun hotel unit.', 'danger')
            return redirect(url_for('dashboard'))

        cur = mysql.connection.cursor()

        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            hotel_unit = request.form['hotel_unit']

            try:
                # Update username
                if username:
                    cur.execute('UPDATE users SET username = %s WHERE id = %s', (username, user_id))
                
                # Update password if provided
                if password:
                    hashed_password = generate_password_hash(password)
                    cur.execute('UPDATE users SET password = %s WHERE id = %s', (hashed_password, user_id))
                
                # Update hotel unit
                if hotel_unit:
                    cur.execute('UPDATE users SET hotel_unit = %s WHERE id = %s', (hotel_unit, user_id))

                mysql.connection.commit()
                flash('Akun hotel unit berhasil diperbarui.', 'success')
            except Exception as e:
                mysql.connection.rollback()
                flash('Terjadi kesalahan saat memperbarui akun.', 'danger')
            finally:
                cur.close()
            
            return redirect(url_for('manage_hotel_accounts'))

        # GET request - fetch user data
        cur.execute('''
            SELECT u.id, u.username, u.hotel_unit, h.name 
            FROM users u 
            JOIN hotels h ON u.hotel_unit = h.id 
            WHERE u.id = %s
        ''', (user_id,))
        user = cur.fetchone()
        cur.close()

        if not user:
            flash('Akun tidak ditemukan.', 'danger')
            return redirect(url_for('manage_hotel_accounts'))

        return render_template('edit_hotel_unit.html', user=user, hotels=Hotel.get_all_hotels())
    
    
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
            try:
                # Ambil nama hotel dari user yang sedang login
                with mysql.connection.cursor() as cur:
                    cur.execute('SELECT name FROM hotels WHERE id = %s', (current_user.hotel_unit,))
                    current_user_hotel_name = cur.fetchone()[0].lower()

                # Buat folder untuk menyimpan hasil evaluasi
                evaluation_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'evaluations', current_user_hotel_name)
                os.makedirs(evaluation_folder, exist_ok=True)
                evaluation_file = os.path.join(evaluation_folder, 'evaluation_results.json')

                # Periksa apakah file dipilih
                if 'file' not in request.files or request.files['file'].filename == '':
                    flash('No file selected')
                    return redirect(request.url)

                file = request.files['file']
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Baca file
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath, sep=';')
                else:
                    df = pd.read_excel(filepath)

                # Validasi hotel_unit
                invalid_hotel_units = df[df['hotel_unit'].str.lower() != current_user_hotel_name]
                if not invalid_hotel_units.empty:
                    os.remove(filepath)
                    invalid_hotels_list = ", ".join(invalid_hotel_units['hotel_unit'].unique())
                    flash(f'Upload gagal. Hotel unit tidak sesuai: {invalid_hotels_list}')
                    return redirect(request.url)

                # Konversi format tanggal
                df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

                # Ambil daftar hotel
                with mysql.connection.cursor() as cur:
                    cur.execute('SELECT id, name FROM hotels')
                    hotels_map = {hotel[1].lower(): hotel[0] for hotel in cur.fetchall()}

                # Split data
                train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

                # Save original split data
                train_data_path = os.path.join(evaluation_folder, f'train_data.csv')
                test_data_path = os.path.join(evaluation_folder, f'test_data.csv')
                train_data.to_csv(train_data_path, index=False)
                test_data.to_csv(test_data_path, index=False)

                # Translate and save training data
                translated_train_path = os.path.join(evaluation_folder, 'translated_train.csv')
                translated_test_path = os.path.join(evaluation_folder, 'translated_test.csv')

                translated_train_data = sentiment_analyzer.save_translated_data(train_data, translated_train_path)
                translated_test_data = sentiment_analyzer.save_translated_data(test_data, translated_test_path)

                # Train models with translated data
                sentiment_analyzer.train_and_save(translated_train_data)

                # Add sentiment labels using TextBlob on translated data
                if 'sentiment_label' not in test_data.columns:
                    test_data['sentiment_label'] = translated_test_data['translated_review'].apply(
                        sentiment_analyzer.get_textblob_sentiment
                    )

                # Evaluate models
                evaluation_results = sentiment_analyzer.evaluate(test_data, current_user.hotel_unit)

                # Save evaluation results
              # Save evaluation results
                evaluation_data = {
                    'svm_results': {
                        'accuracy': f"SVM Model Accuracy: {evaluation_results['svm']['accuracy']:.2f}%",
                        'confusion_matrix': (
                            evaluation_results['svm']['confusion_matrix'].tolist()
                            if hasattr(evaluation_results['svm']['confusion_matrix'], 'tolist') 
                            else evaluation_results['svm']['confusion_matrix']
                        ),
                        'report': evaluation_results['svm']['report']
                    },
                    'nb_results': {
                        'accuracy': f"Naive Bayes Model Accuracy: {evaluation_results['naive_bayes']['accuracy']:.2f}%",
                        'confusion_matrix': (
                            evaluation_results['naive_bayes']['confusion_matrix'].tolist()
                            if hasattr(evaluation_results['naive_bayes']['confusion_matrix'], 'tolist') 
                            else evaluation_results['naive_bayes']['confusion_matrix']
                        ),
                        'report': evaluation_results['naive_bayes']['report']
                    },
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'hotel_unit': current_user.hotel_unit
                }
                with open(evaluation_file, 'w') as f:
                    json.dump(evaluation_data, f)

                # Save reviews to database with translated sentiment
                for _, row in train_data.iterrows():
                    hotel_name = row['hotel_unit'].lower()
                    hotel_id = hotels_map.get(hotel_name)

                    if hotel_id is None:
                        logging.warning(f"Hotel not found: {row['hotel_unit']}")
                        continue

                    review = row['review'] if pd.notna(row['review']) else ""
                    rating = row['rating'] if pd.notna(row['rating']) else None

                    if rating is not None and (rating < 1 or rating > 10):
                        logging.warning(f"Review skipped due to invalid rating: {rating}")
                        continue

                    cleaned_review = sentiment_analyzer.preprocess_text(review)
                    translated_review = sentiment_analyzer.translate_to_english(cleaned_review)
                    sentiment = sentiment_analyzer.get_textblob_sentiment(translated_review)

                    Review.add_review(
                        hotel_id,
                        row['name'] if pd.notna(row['name']) else None,
                        rating,
                        row['date'] if pd.notna(row['date']) else None,
                        cleaned_review,
                        sentiment
                    )

                # Cleanup
                for file_path in [filepath]:
                    if os.path.exists(file_path):
                        os.remove(file_path)

                flash('File uploaded, translated, and processed successfully')
                return redirect(url_for('evaluate', model='svm'))

            except Exception as e:
                logging.error(f"Error processing file: {str(e)}")
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)

        return render_template('upload.html')



    @app.route('/reviews')
    @login_required
    def reviews():
        page = request.args.get('page', default=1, type=int)
        hotel_id = request.args.get('hotel_id', type=int)
        sentiment = request.args.get('sentiment')
        per_page = 10
        
        if current_user.role != 'admin':
            # Non-admin users can only see their own hotel's reviews
            reviews, total_reviews = Review.get_hotel_reviews_paginated(current_user.hotel_unit, page, sentiment=sentiment)
        else:
            if hotel_id:
                # If hotel_id is provided, filter reviews by that hotel
                reviews, total_reviews = Review.get_paginated_reviews_by_hotel(hotel_id, page, sentiment=sentiment)
            else:
                # Show all reviews for admin
                reviews, total_reviews = Review.get_paginated_reviews(page, sentiment=sentiment)
        
        total_pages = (total_reviews + per_page - 1) // per_page

        # For AJAX requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'status': 'success',
                'reviews_html': render_template('partials/review_rows.html', reviews=reviews),
                'pagination_html': render_template('partials/pagination.html', 
                                                current_page=page, 
                                                total_pages=total_pages),
                'showing_info': f"{len(reviews)} out of {total_reviews} reviews"
            })

        # Fetch hotels for the dropdown (only for admin)
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT id, name FROM hotels")
        hotels = cursor.fetchall()
        cursor.close()

        return render_template('reviews.html', 
                            reviews=reviews,
                            total_reviews=total_reviews,
                            current_page=page, 
                            total_pages=total_pages,
                            hotels=hotels)
    
    @app.route('/delete-reviews', methods=['POST'])
    @login_required
    def delete_reviews():
        try:
            # Ambil list ID review dari request
            review_ids = request.json.get('review_ids', [])
            
            if not review_ids:
                return jsonify({
                    'status': 'error', 
                    'message': 'No review IDs provided'
                }), 400
            
            # Jika bukan admin, hanya bisa menghapus review di hotel unitnya sendiri
            if current_user.role != 'admin':
                reviews_to_delete = Review.get_hotel_reviews(current_user.hotel_unit)
                reviews_to_delete_ids = [review[0] for review in reviews_to_delete]  # Ambil ID review yang bisa dihapus
                review_ids = [rid for rid in review_ids if rid in reviews_to_delete_ids]  # Filter ID yang valid
            
            # Hapus review
            deleted_count = 0
            for review_id in review_ids:
                if Review.delete_review(review_id):  # Pastikan ada metode delete_review di model
                    deleted_count += 1
            
            return jsonify({
                'status': 'success', 
                'deleted_count': deleted_count
            }), 200
        
        except Exception as e:
            logging.error(f"Error deleting reviews: {e}")
            return jsonify({
                'status': 'error', 
                'message': str(e)
            }), 500

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
    
    @app.route('/evaluate/<model>', methods=['GET'])
    @login_required
    def evaluate(model):
        try:
            # Ambil nama hotel dari user yang sedang login
            with mysql.connection.cursor() as cur:
                cur.execute('SELECT name FROM hotels WHERE id = %s', (current_user.hotel_unit,))
                current_user_hotel_name = cur.fetchone()[0].lower()

            # Path ke file hasil evaluasi
            evaluation_file = os.path.join(
                app.config['UPLOAD_FOLDER'],
                'evaluations',
                current_user_hotel_name,
                'evaluation_results.json'
            )

            # Cek apakah file hasil evaluasi ada dan tidak kosong
            if not os.path.exists(evaluation_file) or os.path.getsize(evaluation_file) == 0:
                flash('Tidak ada hasil evaluasi untuk hotel Anda. Silakan upload data terlebih dahulu.')
                return render_template('evaluate.html', model=model)

            # Baca hasil evaluasi dari file JSON
            with open(evaluation_file, 'r') as f:
                evaluation_data = json.load(f)

            # Ambil hasil evaluasi berdasarkan model yang dipilih
            if model == 'svm':
                svm_results = evaluation_data.get('svm_results', None)
                svm_report = svm_results.get('report', None) if svm_results else None
                return render_template(
                    'evaluate.html',
                    model=model,
                    svm_results=svm_results,
                    svm_report=svm_report
                )
            elif model == 'naive_bayes':
                nb_results = evaluation_data.get('nb_results', None)
                nb_report = nb_results.get('report', None) if nb_results else None
                return render_template(
                    'evaluate.html',
                    model=model,
                    nb_results=nb_results,
                    nb_report=nb_report
                )
            else:
                flash('Model yang dipilih tidak valid.')
                return render_template('evaluate.html', model=model)

        except Exception as e:
            logging.error(f"Error in evaluate route: {str(e)}")
            flash('Terjadi kesalahan saat membaca hasil evaluasi.')
            return render_template('evaluate.html', model=model)


