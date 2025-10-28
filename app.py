# rason_backend/app.py
from flask import Flask
from config.settings import SECRET_KEY, UPLOAD_FOLDER
from core.db import db_init
from routes.pages import pages
from routes.api import api
from routes.admin import admin

def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.secret_key = SECRET_KEY
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    # Init DB
    db_init()

    # Blueprints
    app.register_blueprint(pages)
    app.register_blueprint(api)
    app.register_blueprint(admin)

    # 403 handler (copy dari versi lama)
    @app.errorhandler(403)
    def forbidden_error(e):
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Access Denied</title>
            <style>
                body {background: linear-gradient(to bottom right, #0a2a6b, #004aad, #0078d7);font-family: 'Inter', sans-serif;color: white;height: 100vh;display: flex;align-items: center;justify-content: center;margin: 0;}
                .box {background: rgba(255,255,255,0.12);padding: 30px 40px;border-radius: 12px;text-align: center;box-shadow: 0 4px 20px rgba(0,0,0,0.3);backdrop-filter: blur(10px);max-width: 400px;width: 90%;}
                h1 {font-size: 1.6em;margin-bottom: 10px;color: #f87171;}
                p {opacity: 0.9;margin-bottom: 20px;line-height: 1.5;}
                button {background: #38bdf8;color: white;border: none;border-radius: 8px;padding: 10px 18px;font-weight: 600;cursor: pointer;transition: all 0.2s;}
                button:hover { background: #0ea5e9; }
            </style>
        </head>
        <body>
            <div class="box">
                <h1>🚫 Access Restricted</h1>
                <p>You are not allowed to access this page.<br>
                Please contact your administrator if you think this is a mistake.</p>
                <button onclick="window.location.href='/'">🏠 Back to Home</button>
            </div>
        </body>
        </html>
        """, 403

    @app.after_request
    def add_no_cache_headers(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    return app

