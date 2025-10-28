# rason_backend/routes/__init__.py
"""Blueprint routes for Flask application (pages, api, admin)."""

# optional auto-import for easy registration
from .api import api
from .pages import pages
from .admin import admin

__all__ = ["api", "pages", "admin"]
