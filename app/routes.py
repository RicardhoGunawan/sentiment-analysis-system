from flask import render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import os
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
        
       # Inisialisasi SentimentAnalyzer
        analyzer = SentimentAnalyzer()  # Pastikan path file kamus benar

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
            # Ambil nama hotel dari user yang sedang login
            with mysql.connection.cursor() as cur:
                cur.execute('SELECT name FROM hotels WHERE id = %s', (current_user.hotel_unit,))
                current_user_hotel_name = cur.fetchone()[0].lower()

            if 'file' not in request.files:
                flash('No file selected')
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                flash('No file selected')
                return redirect(request.url)

            try:
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

                # Simpan data
                train_data_path = os.path.join(app.config['UPLOAD_FOLDER'], f'train_data_{current_user.hotel_unit}.csv')
                test_data_path = os.path.join(app.config['UPLOAD_FOLDER'], f'test_data_{current_user.hotel_unit}.csv')
                train_data.to_csv(train_data_path, index=False)
                test_data.to_csv(test_data_path, index=False)

                # Train models
                sentiment_analyzer.train_and_save(train_data)

                # Add sentiment labels if needed
                if 'sentiment_label' not in test_data.columns:
                    test_data['sentiment_label'] = test_data['review'].apply(sentiment_analyzer.get_sentiment_label)

                # Evaluate models
                evaluation_results = sentiment_analyzer.evaluate(test_data, current_user.hotel_unit)
                
                # Save results to session
                session['svm_results'] = {
                    'accuracy': f"SVM Model Accuracy: {evaluation_results['svm']['accuracy']:.2f}%",
                    'confusion_matrix': evaluation_results['svm']['confusion_matrix'],
                    'report': evaluation_results['svm']['report']
                }
                
                session['nb_results'] = {
                    'accuracy': f"Naive Bayes Model Accuracy: {evaluation_results['naive_bayes']['accuracy']:.2f}%",
                    'confusion_matrix': evaluation_results['naive_bayes']['confusion_matrix'],
                    'report': evaluation_results['naive_bayes']['report']
                }
                
                session['hotel_unit'] = current_user.hotel_unit

                # Save reviews to database
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
                    # Use SVM model for prediction
                    sentiment = sentiment_analyzer.predict([cleaned_review], model='svm')[0]

                    Review.add_review(
                        hotel_id,
                        row['name'] if pd.notna(row['name']) else None,
                        rating,
                        row['date'] if pd.notna(row['date']) else None,
                        cleaned_review,
                        sentiment
                    )

                # Cleanup
                os.remove(filepath)
                os.remove(train_data_path)
                os.remove(test_data_path)

                flash('File uploaded and processed successfully')
                return redirect(url_for('evaluate'))

            except Exception as e:
                # Cleanup on error
                if os.path.exists(filepath):
                    os.remove(filepath)
                if 'train_data_path' in locals() and os.path.exists(train_data_path):
                    os.remove(train_data_path)
                if 'test_data_path' in locals() and os.path.exists(test_data_path):
                    os.remove(test_data_path)
                
                logging.error(f"Error processing file: {str(e)}")
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)

        return render_template('upload.html')



    @app.route('/reviews')
    @login_required
    def reviews():
        page = request.args.get('page', default=1, type=int)
        hotel_id = request.args.get('hotel_id', type=int)
        per_page = 10
        
        if current_user.role != 'admin':
            # Non-admin users can only see their own hotel's reviews
            reviews, total_reviews = Review.get_hotel_reviews_paginated(current_user.hotel_unit, page)
        else:
            if hotel_id:
                # If hotel_id is provided, filter reviews by that hotel
                reviews, total_reviews = Review.get_paginated_reviews_by_hotel(hotel_id, page)
            else:
                # Show all reviews for admin
                reviews, total_reviews = Review.get_paginated_reviews(page)
        
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
        # Ambil hotel_unit dari user yang sedang login
        current_user_hotel_unit = current_user.hotel_unit

        # Periksa apakah hasil evaluasi tersedia dan sesuai dengan hotel_unit user
        stored_hotel_unit = session.get('hotel_unit')

        if stored_hotel_unit is None or stored_hotel_unit != current_user_hotel_unit:
            flash('Tidak ada hasil evaluasi untuk hotel Anda.')
            return render_template('evaluate.html', svm_results=None, nb_results=None, svm_label_counts=None, nb_label_counts=None)

        # Ambil hasil evaluasi dari session atau database berdasarkan model yang dipilih
        if model == 'svm':
            svm_results = session.get('svm_results')
            nb_results = None
            svm_label_counts = session.get('svm_label_counts')  # Ambil svm_label_counts
            nb_label_counts = None
        elif model == 'naive_bayes':
            svm_results = None
            nb_results = session.get('nb_results')
            svm_label_counts = None
            nb_label_counts = session.get('nb_label_counts')  # Ambil nb_label_counts
        else:
            svm_results = None
            nb_results = None
            svm_label_counts = None
            nb_label_counts = None

        # Format classification report untuk ditampilkan jika ada
        svm_report = None
        nb_report = None

        # Proses hasil evaluasi untuk SVM jika ada
        if svm_results and svm_results.get('report'):
            svm_report = {
                'Negative': {
                    'Precision': f"{svm_results['report']['Negative']['precision']:.2f}",
                    'Recall': f"{svm_results['report']['Negative']['recall']:.2f}",
                    'F1-Score': f"{svm_results['report']['Negative']['f1-score']:.2f}",
                    'Support': svm_results['report']['Negative']['support']
                },
                'Positive': {
                    'Precision': f"{svm_results['report']['Positive']['precision']:.2f}",
                    'Recall': f"{svm_results['report']['Positive']['recall']:.2f}",
                    'F1-Score': f"{svm_results['report']['Positive']['f1-score']:.2f}",
                    'Support': svm_results['report']['Positive']['support']
                },
                'Accuracy': f"{svm_results['report']['accuracy']:.2f}",
                'Macro Avg': {
                    'Precision': f"{svm_results['report']['macro avg']['precision']:.2f}",
                    'Recall': f"{svm_results['report']['macro avg']['recall']:.2f}",
                    'F1-Score': f"{svm_results['report']['macro avg']['f1-score']:.2f}",
                    'Support': svm_results['report']['macro avg']['support']
                },
                'Weighted Avg': {
                    'Precision': f"{svm_results['report']['weighted avg']['precision']:.2f}",
                    'Recall': f"{svm_results['report']['weighted avg']['recall']:.2f}",
                    'F1-Score': f"{svm_results['report']['weighted avg']['f1-score']:.2f}",
                    'Support': svm_results['report']['weighted avg']['support']
                }
            }

        # Proses hasil evaluasi untuk Naive Bayes jika ada
        if nb_results and nb_results.get('report'):
            nb_report = {
                'Negative': {
                    'Precision': f"{nb_results['report']['Negative']['precision']:.2f}",
                    'Recall': f"{nb_results['report']['Negative']['recall']:.2f}",
                    'F1-Score': f"{nb_results['report']['Negative']['f1-score']:.2f}",
                    'Support': nb_results['report']['Negative']['support']
                },
                'Positive': {
                    'Precision': f"{nb_results['report']['Positive']['precision']:.2f}",
                    'Recall': f"{nb_results['report']['Positive']['recall']:.2f}",
                    'F1-Score': f"{nb_results['report']['Positive']['f1-score']:.2f}",
                    'Support': nb_results['report']['Positive']['support']
                },
                'Accuracy': f"{nb_results['report']['accuracy']:.2f}",
                'Macro Avg': {
                    'Precision': f"{nb_results['report']['macro avg']['precision']:.2f}",
                    'Recall': f"{nb_results['report']['macro avg']['recall']:.2f}",
                    'F1-Score': f"{nb_results['report']['macro avg']['f1-score']:.2f}",
                    'Support': nb_results['report']['macro avg']['support']
                },
                'Weighted Avg': {
                    'Precision': f"{nb_results['report']['weighted avg']['precision']:.2f}",
                    'Recall': f"{nb_results['report']['weighted avg']['recall']:.2f}",
                    'F1-Score': f"{nb_results['report']['weighted avg']['f1-score']:.2f}",
                    'Support': nb_results['report']['weighted avg']['support']
                }
            }

        return render_template('evaluate.html', 
                            svm_results=svm_results, 
                            nb_results=nb_results, 
                            svm_report=svm_report, 
                            nb_report=nb_report, 
                            svm_label_counts=svm_label_counts,  # Pass svm_label_counts to template
                            nb_label_counts=nb_label_counts,    # Pass nb_label_counts to template
                            model=model)
