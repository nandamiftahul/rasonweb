# rason_backend/core/auth.py
import os, json, secrets
from datetime import datetime
from functools import wraps
from flask import session, redirect, url_for, abort
from werkzeug.security import check_password_hash, generate_password_hash

from config.settings import USERS_FILE, SESSION_TIMEOUT_MINUTES
from .utils import encrypt_value, decrypt_value

ACTIVE_USERS = set()
GLOBAL_SESSION_VERSION = 1
USER_SESSION_TOKENS = {}
USER_STATE = {}  # username -> dict(meta/levels/last_active)

def get_user_store():
    """Return the current user's store dict."""
    user = session.get("user")
    return USER_STATE[user]

def clear_user_store():
    """Wipe current user's store on logout."""
    user = session.get("user")
    if user in USER_STATE:
        del USER_STATE[user]

def get_global_session_version():
    global GLOBAL_SESSION_VERSION
    return GLOBAL_SESSION_VERSION

def bump_global_session_version():
    """Naikkan versi sesi global → semua user auto logout."""
    global GLOBAL_SESSION_VERSION
    GLOBAL_SESSION_VERSION += 1
    USER_SESSION_TOKENS.clear()
    ACTIVE_USERS.clear()
    print(f"🔒 Global session version bumped to {GLOBAL_SESSION_VERSION}")

def load_users_from_file():
    """Load hashed users from JSON (keep encrypted fields as-is). Decrypt on return."""
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print("⚠️ Failed to read users.json:", e)
        return {}

    # decrypt non-password fields (expiry, pages, etc.)
    result = {}
    for username, info in raw.items():
        if not isinstance(info, dict):
            result[username] = info
            continue

        decrypted = {}
        for k, v in info.items():
            if k == "password":
                decrypted[k] = v
            else:
                # attempt decrypt; decrypt_value will return original if not encrypted
                decrypted[k] = decrypt_value(v) if isinstance(v, str) else v
        result[username] = decrypted

    return result

VALID_USERS = load_users_from_file()

def save_users(path=USERS_FILE):
    try:
        out = {}
        for username, info in VALID_USERS.items():
            if isinstance(info, dict):
                to_save = {}
                for k,v in info.items():
                    if k=="password": to_save[k]=v
                    else:
                        to_save[k] = v if (isinstance(v,str) and v.startswith("gAAAA")) else encrypt_value(v)
                out[username]=to_save
            else:
                out[username]=info
        json.dump(out, open(path,"w",encoding="utf-8"), indent=2, ensure_ascii=False)
    except Exception as e:
        print("⚠️ Failed to save users.json:", e)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("pages.login"))

        if session.get("session_version") != get_global_session_version():
            session.clear()
            return redirect(url_for("pages.login"))

        global USER_SESSION_TOKENS, ACTIVE_USERS
        if "USER_SESSION_TOKENS" in globals():
            user = session.get("user")
            token = session.get("user_token")
            current_token = USER_SESSION_TOKENS.get(user)
            if current_token and token != current_token:
                print(f"🔄 Session replaced for user '{user}' (device switched)")
                session.clear()
                return redirect(url_for("pages.login"))

        # 🕒 Timeout check (30 menit idle)
        last_active = session.get("last_active")
        now = datetime.utcnow()
        if last_active:
            last_active_dt = datetime.strptime(last_active, "%Y-%m-%d %H:%M:%S")
            inactive_minutes = (now - last_active_dt).total_seconds() / 60
            user = session.get("user")
            
            # Tentukan timeout berbeda
            if user == "display":
                timeout_limit = 60 * 24 * 30   # 30 hari dalam menit
            else:
                timeout_limit = SESSION_TIMEOUT_MINUTES
            
            if inactive_minutes > timeout_limit:
                print(f"⏰ Session expired for user '{user}' "
                      f"(inactive {inactive_minutes:.1f} min / limit {timeout_limit} min)")
                session.clear()
                return redirect(url_for("pages.login"))     
        session["last_active"] = now.strftime("%Y-%m-%d %H:%M:%S")

        # 🧠 Simpan ke global USER_STATE
        global USER_STATE
        user = session.get("user")
        if user:
            if user not in USER_STATE:
                USER_STATE[user] = {}
            USER_STATE[user]["last_active"] = now.strftime("%Y-%m-%d %H:%M:%S")

        return f(*args, **kwargs)
    return decorated_function

def page_access_required(page_name):
    """Batasi akses halaman berdasarkan izin per user."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            user = session.get("user")
            if not user or user not in VALID_USERS:
                return redirect(url_for("pages.login"))

            user_info = VALID_USERS[user]
            allowed_pages = user_info.get("pages", [])

            # admin punya akses ke semua halaman
            if user == "admin" or "*" in allowed_pages or page_name in allowed_pages:
                return f(*args, **kwargs)

            print(f"🚫 Access denied for '{user}' → {page_name}")
            return abort(403)  # Forbidden
        return wrapper
    return decorator

def load_users():
    """Load user data from JSON file or fallback to default."""
    global VALID_USERS
    if os.path.exists(USERS_FILE):
        try:
            VALID_USERS = load_users_from_file()
            # Pastikan struktur benar (jika value masih string password)
            for u, info in list(VALID_USERS.items()):
                if isinstance(info, str):
                    VALID_USERS[u] = {"password": info}
            return
        except Exception as e:
            print("⚠️ Failed to read users.json:", e)
    # fallback default
    VALID_USERS = {"admin": {"password": "admin123", "expiry": "2099-01-01", "pages": ["*"]}}
    save_users()

# Jalankan sekali saat startup
load_users()
