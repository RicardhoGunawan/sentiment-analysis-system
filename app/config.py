import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = 'your-secret-key-here'
    
    # MySQL configurations
    MYSQL_HOST = 'localhost'
    MYSQL_USER = 'root'
    MYSQL_PASSWORD = '@mysql'
    MYSQL_DB = 'hotel_sentiment'
    
    # Ensure utf8mb4 charset
    MYSQL_CHARSET = 'utf8mb4'  # Adding charset configuration for utf8mb4

    # Upload folder
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
    ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
