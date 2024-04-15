from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

#db = SQLAlchemy()

cor = CORS()

def create_app():
    app = Flask(__name__)
    
    cor.init_app(app)
    
 #   app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    
 #   db.init_app(app)
    
 #   from . import model

 #   with app.app_context():
 #       db.create_all()
    
    from .views import main
    app.register_blueprint(main)
    
    
    
    return app