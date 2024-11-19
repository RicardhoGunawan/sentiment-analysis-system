from flask import Flask
from flask_mysqldb import MySQL
from flask_login import LoginManager
from .config import Config

mysql = MySQL()
login_manager = LoginManager()
login_manager.login_view = 'login'

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    mysql.init_app(app)
    login_manager.init_app(app)
    
    from .routes import init_routes
    init_routes(app)
    
    return app