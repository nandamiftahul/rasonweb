#! /usr/bin/python3
import os, re, io, configparser, warnings, time
import json
import ftplib
import subprocess
import pandas as pd
import numpy as np
from flask import (
    Flask, render_template, request, redirect, abort,
    url_for, jsonify, session, send_file, Response
)
from functools import wraps
from collections import defaultdict
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")  # safe for server
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from functools import wraps

import metpy.calc as mpcalc
from metpy.units import units
from metpy.plots import SkewT, Hodograph
from scipy.signal import medfilt
from geopy.distance import geodesic

# --- BEGIN: Encryption helpers (Fernet) ---
import json as _json
import base64 as _base64
import hashlib as _hashlib

from cryptography.fernet import Fernet

def _derive_fernet_key(secret: str) -> bytes:
    """
    Derive a 32-byte key suitable for Fernet from a secret string using SHA256,
    then urlsafe_b64encode it (Fernet expects 32-byte base64-encoded key).
    """
    if not secret:
        raise ValueError("Secret key for Fernet derivation is empty.")
    h = _hashlib.sha256(secret.encode("utf-8")).digest()
    return _base64.urlsafe_b64encode(h)  # 32 bytes -> base64

def _get_fernet():
    key = _derive_fernet_key(app.secret_key if app.secret_key else cfg.get("secretkey", ""))
    return Fernet(key)

def encrypt_value(val):
    """
    Encrypts any JSON-serializable value (str/list/dict/int/etc) into a Fernet token string.
    """
    f = _get_fernet()
    # convert to JSON text to allow dict/list storage
    raw = _json.dumps(val, ensure_ascii=False).encode("utf-8")
    token = f.encrypt(raw)
    return token.decode("utf-8")

def decrypt_value(token_str):
    """
    Decrypt a Fernet token string and return original Python object (via JSON).
    If the token doesn't look like a Fernet token or fails to decrypt, returns the original value.
    """
    if token_str is None:
        return None
    if not isinstance(token_str, str):
        return token_str
    # quick fingerprint: Fernet tokens usually start with 'gAAAA' when base64-urlencoded
    if not token_str.startswith("gAAAA"):
        # likely plaintext — attempt to parse JSON (for lists/dicts) or return raw string
        try:
            return _json.loads(token_str)
        except Exception:
            return token_str
    try:
        f = _get_fernet()
        raw = f.decrypt(token_str.encode("utf-8"))
        return _json.loads(raw.decode("utf-8"))
    except Exception as e:
        # failed to decrypt — return original token (fallback)
        print("⚠️ decrypt_value failed:", e)
        return token_str
# --- END: Encryption helpers (Fernet) ---


# Load local .env file (ignored in production)
load_dotenv()

# === Database SQLite for Radiosonde Cache ===
import sqlite3
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash
import secrets

DB_PATH = "rason_data.db"

def db_init():
    """Initialize 4 main tables for BUFR/BFR/BFH/BIN caching."""
    with sqlite3.connect(DB_PATH) as conn:
        for t in ["bufr", "bfr", "bfh", "bin"]:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site TEXT,
                    filename TEXT UNIQUE,
                    filetype TEXT,
                    file_date TEXT,
                    meta_json TEXT,
                    levels_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
db_init()

def db_get(filetype, site, filename):
    """Return (df_meta, df_levels) if exists in DB; else None."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"SELECT meta_json, levels_json FROM {filetype} WHERE site=? AND filename=?",
            (site, filename)
        ).fetchone()
    if not row:
        return None
    try:
        meta_df = pd.read_json(io.StringIO(row[0]))
        levels_df = pd.read_json(io.StringIO(row[1]))
        return meta_df, levels_df
    except Exception as e:
        print(f"[DB] decode error {filename}: {e}")
        return None

def db_insert(filetype, site, filename, file_date, df_meta, df_levels):
    """Insert parsed BUFR into DB."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(f"""
                INSERT OR REPLACE INTO {filetype}
                (site, filename, filetype, file_date, meta_json, levels_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                site,
                filename,
                filetype,
                file_date,
                df_meta.to_json(),
                df_levels.to_json()
            ))
            conn.commit()
        print(f"[DB] ✅ inserted {filename}")
    except Exception as e:
        print(f"[DB] insert error {filename}: {e}")

# --- Load configuration from environment ---
CONFIG = {
    "ftp": {
        "name": os.getenv("FTP_NAME", "Unknown"),
        "host": os.getenv("FTP_HOST", "localhost"),
        "port": int(os.getenv("FTP_PORT", "21")),
        "user": os.getenv("FTP_USER", "anonymous"),
        "password": os.getenv("FTP_PASS", ""),
        "base_path": os.getenv("FTP_BASE_PATH", "/UA"),
        "file_ext": os.getenv("FTP_FILE_EXT", ".bufr,.bfh,.bfr,.bin").split(","),
        "limit": int(os.getenv("FTP_LIMIT", "30")),
        "secretkey": os.getenv("SECRETKEY", "Unknown"),
    }
}
cfg = CONFIG["ftp"]
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
#app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")
app.secret_key = cfg["secretkey"]

ACTIVE_USERS = set()

# === 🧩 Global session versioning (logout-all control) ===
GLOBAL_SESSION_VERSION = 1  # default awal, naikkan untuk memaksa logout semua

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

# In-memory per-user store (username -> {"metadata": {...}, "levels": [...]})
USER_STATE = defaultdict(lambda: {"metadata": {}, "levels": []})

def get_user_store():
    """Return the current user's store dict."""
    user = session.get("user")
    return USER_STATE[user]

def clear_user_store():
    """Wipe current user's store on logout."""
    user = session.get("user")
    if user in USER_STATE:
        del USER_STATE[user]

SITES_FILE = "sites.json"
SITE_LIST = ["aceh", "tarakan", "sorong", "cilacap", "pangkalanbun", "ranai"]
def load_sites():
    """Load site list from JSON file or use defaults."""
    global SITE_LIST
    if os.path.exists(SITES_FILE):
        try:
            with open(SITES_FILE, "r") as f:
                data = json.load(f)
                SITE_LIST = data.get("sites", [])
        except Exception as e:
            print("⚠️ Failed to read sites.json:", e)
            SITE_LIST = SITE_LIST
    else:
        SITE_LIST = SITE_LIST
        save_sites()

def save_sites():
    """Save site list to JSON file."""
    try:
        with open(SITES_FILE, "w") as f:
            json.dump({"sites": SITE_LIST}, f, indent=2)
    except Exception as e:
        print("⚠️ Failed to save sites.json:", e)

load_sites()

# --- Authentication ---
USERS_FILE = "users.json"

USER_FILE = "users.json"

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

def load_users():
    """Load user data from JSON file or fallback to default."""
    global VALID_USERS
    if os.path.exists(USER_FILE):
        try:
            VALID_USERS = load_users_from_file()
            # Ensure structure compatibility: if value is plain string password, wrap it
            for u, info in list(VALID_USERS.items()):
                if isinstance(info, str):
                    VALID_USERS[u] = {"password": info}
            return
        except Exception as e:
            print("⚠️ Failed to read users.json:", e)
    # fallback default
    VALID_USERS = {"admin": {"password": "admin123", "expiry": "2099-01-01", "pages": ["*"]}}
    save_users()

def save_users():
    """Save user data to JSON file (encrypt non-password fields)."""
    try:
        out = {}
        for username, info in VALID_USERS.items():
            if isinstance(info, dict):
                to_save = {}
                for k, v in info.items():
                    if k == "password":
                        to_save[k] = v
                    else:
                        # encrypt value (but if already looks encrypted, keep it)
                        if isinstance(v, str) and v.startswith("gAAAA"):
                            to_save[k] = v
                        else:
                            to_save[k] = encrypt_value(v)
                out[username] = to_save
            else:
                out[username] = info
        with open(USER_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("⚠️ Failed to save users.json:", e)

load_users()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 🚪 Tidak ada sesi login
        if "user" not in session:
            return redirect(url_for("login"))

        # 🌍 Global logout: semua user dipaksa keluar
        if session.get("session_version") != get_global_session_version():
            session.clear()
            return redirect(url_for("login"))

        # 👤 Per-user logout: token mismatch (admin force logout)
        global USER_SESSION_TOKENS, ACTIVE_USERS
        if "USER_SESSION_TOKENS" in globals():
            user = session.get("user")
            token = session.get("user_token")
            current_token = USER_SESSION_TOKENS.get(user)

            if current_token and token != current_token:
                # 🟢 Jangan hapus dari ACTIVE_USERS — user masih aktif di session baru
                print(f"🔄 Session replaced for user '{user}' (device switched)")
                session.clear()
                return redirect(url_for("login"))

        # ✅ Semua aman, lanjut ke route
        return f(*args, **kwargs)
    return decorated_function

def page_access_required(page_name):
    """Batasi akses halaman berdasarkan izin per user."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            user = session.get("user")
            if not user or user not in VALID_USERS:
                return redirect(url_for("login"))

            user_info = VALID_USERS[user]
            allowed_pages = user_info.get("pages", [])

            # admin punya akses ke semua halaman
            if user == "admin" or "*" in allowed_pages or page_name in allowed_pages:
                return f(*args, **kwargs)

            print(f"🚫 Access denied for '{user}' → {page_name}")
            return abort(403)  # Forbidden
        return wrapper
    return decorator

@app.route("/login", methods=["GET", "POST"])
def login():
    print("CHECK ===>",request.headers)
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Pastikan user ada di daftar
        if username not in VALID_USERS:
            return render_template("login.html", error="Invalid credentials")

        user_info = VALID_USERS[username]

        # Ambil password hash dan tanggal expiry
        user_hash = user_info.get("password")
        expiry_str = user_info.get("expiry")

        # 🔒 Cek password
        if not check_password_hash(user_hash, password):
            return render_template("login.html", error="Invalid credentials")

        # ⏰ Cek apakah masa aktif sudah lewat
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            if datetime.utcnow().date() > expiry_date:
                return render_template("login.html", error=f"❌ Subscription expired on {expiry_str}")
        except Exception:
            print(f"⚠️ Invalid expiry date format for user {username}: {expiry_str}")

        # ✅ Jika valid → lanjutkan login
        session["user"] = username
        session["session_version"] = get_global_session_version()

        # 🧩 Generate token baru setiap kali login (invalidate sesi lama)
        global USER_SESSION_TOKENS
        if "USER_SESSION_TOKENS" not in globals():
            USER_SESSION_TOKENS = {}
        new_token = secrets.token_hex(8)
        USER_SESSION_TOKENS[username] = new_token
        session["user_token"] = new_token

        # 🟢 Tambahkan ke daftar user aktif
        ACTIVE_USERS.add(username)
        print(f"✅ User logged in: {username} (active: {list(ACTIVE_USERS)})")

        return redirect(url_for("main_page"))

    # GET method
    return render_template("login.html")

@app.route("/logout")
def logout():
    user = session.get("user")

    # 🧩 Hapus user dari daftar aktif
    global ACTIVE_USERS
    if "ACTIVE_USERS" in globals() and user in ACTIVE_USERS:
        ACTIVE_USERS.discard(user)
        print(f"👋 User logged out manually: {user}")

    # 🧩 Bersihkan session token user (tidak mengubah global token)
    session.clear()

    # 🧩 Hapus session cookie
    resp = redirect(url_for("login"))
    resp.set_cookie('session', '', expires=0)
    return resp

@app.route("/admin/logout_all", methods=["POST"])
@login_required
def logout_all_users():
    if session.get("user") != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    bump_global_session_version()
    return jsonify({"success": True, "new_version": get_global_session_version()})

@app.route("/admin/logout_user/<username>", methods=["POST"])
@login_required
def logout_user(username):
    if session.get("user") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    # 🧩 Gunakan per-user session token cache
    # Buat dict global jika belum ada
    global USER_SESSION_TOKENS
    if "USER_SESSION_TOKENS" not in globals():
        USER_SESSION_TOKENS = {}

    # Set token user ke random baru → invalidate sesi mereka
    import secrets
    
    if username in USER_SESSION_TOKENS:
        del USER_SESSION_TOKENS[username]

    # 2) remove from ACTIVE_USERS (agar UI admin langsung tidak menampilkan)
    ACTIVE_USERS.discard(username)
    USER_SESSION_TOKENS[username] = secrets.token_hex(8)
    print(f"🔒 User {username} forced logout.")
    return jsonify({"success": True, "user": username})

@app.route("/api/active_users")
@login_required
def api_active_users():
    # Hanya admin yang boleh melihat daftar login aktif
    if session.get("user") != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify({"active": sorted(list(ACTIVE_USERS))})

@app.route("/api/whoami")
@login_required
def whoami():
    return jsonify({"user": session.get("user", None)})

@app.route("/api/user_expiry", methods=["PUT"])
@login_required
def update_user_expiry():
    if session.get("user") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json(force=True)
    username = data.get("username")
    expiry = data.get("expiry")

    if username not in VALID_USERS:
        return jsonify({"error": "User not found"}), 404

    VALID_USERS[username]["expiry"] = expiry
    save_users()
    print(f"🗓️ Updated expiry for {username} → {expiry}")
    return jsonify({"success": True, "username": username, "expiry": expiry})

@app.route("/api/user_pages", methods=["PUT"])
@login_required
def update_user_pages():
    if session.get("user") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json(force=True)
    username = data.get("username")
    pages = data.get("pages", [])

    if username not in VALID_USERS:
        return jsonify({"error": "User not found"}), 404

    VALID_USERS[username]["pages"] = pages
    save_users()
    print(f"🔧 Updated allowed pages for {username}: {pages}")
    return jsonify({"success": True, "user": username, "pages": pages})

# --- BUFR decode ---
def decode_bufr(filepath):
    result = subprocess.run(
        ["pybufrkit", "decode", "-a", filepath],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"BUFR decode failed: {result.stderr}")
    return result.stdout

# --- Parse BUFR (dynamic + site-specific + fallback) ---
def parse_bufr(decoded_text, site="default"):
    import os, json, re
    import pandas as pd, numpy as np
    from datetime import datetime, timezone

    meta = {}
    levels = []
    current = {}
    station_lat, station_lon = None, None

    # --- Helper functions ---
    def _extract_bytes_field(line: str) -> str:
        """Extract b'..' or b"..". Returns stripped string, else last token."""
        m = re.search(r"b[\"'](.*?)[\"']", line)
        if m:
            return m.group(1).strip()
        return line.split()[-1]

    def _safe_float_tail(line: str):
        try:
            return float(line.split()[-1])
        except Exception:
            return None

    # ==================================================================
    # 🔹 1. Load site-specific mapping (priority)
    # ==================================================================
    CONFIG_DIR = "config"
    site_key = str(site).lower().strip().replace(" ", "_")
    site_file = os.path.join(CONFIG_DIR, f"bufr_mapping_{site_key}.json")
    default_file = "bufr_mapping_full.json"

    mapping = None

    # Try site-specific mapping first
    if os.path.exists(site_file):
        try:
            with open(site_file) as f:
                mapping = json.load(f)
            print(f"🗺️ Using site mapping: {site_file}")
        except Exception as e:
            print(f"⚠️ Failed to read mapping for site '{site_key}': {e}")

    # Fallback to global bufr_mapping_full.json
    if not mapping and os.path.exists(default_file):
        try:
            with open(default_file) as f:
                mapping = json.load(f)
            print(f"ℹ️ Using global mapping: {default_file}")
        except Exception:
            mapping = None

    # ==================================================================
    # 🔹 2. Default fallback mapping (original unchanged)
    # ==================================================================
    default_mapping = {
        "meta": [
            {"original": "WMO BLOCK NUMBER", "variable": "wmo_block"},
            {"original": "WMO STATION NUMBER", "variable": "wmo_station"},
            {"original": "004001 YEAR", "variable": "year"},
            {"original": "004002 MONTH", "variable": "month"},
            {"original": "004003 DAY", "variable": "day"},
            {"original": "004004 HOUR", "variable": "hour"},
            {"original": "004005 MINUTE", "variable": "minute"},
            {"original": "004006 SECOND", "variable": "second"},
            {"original": "LATITUDE (HIGH ACCURACY)", "variable": "station_lat"},
            {"original": "LONGITUDE (HIGH ACCURACY)", "variable": "station_lon"},
            {"original": "HEIGHT OF STATION GROUND", "variable": "station_height_m"},
            {"original": "RADIOSONDE SERIAL NUMBER", "variable": "radiosonde_serial_number"},
            {"original": "RADIOSONDE ASCENSION NUMBER", "variable": "radiosonde_ascension_number"},
            {"original": "RADIOSONDE RELEASE NUMBER", "variable": "radiosonde_release_number"},
            {"original": "RADIOSONDE GROUND RECEIVING SYSTEM", "variable": "radiosonde_ground_rx_system"},
            {"original": "RADIOSONDE OPERATING FREQUENCY", "variable": "radiosonde_operating_frequency"},
            {"original": "BALLOON MANUFACTURER", "variable": "balloon_manufacturer"},
            {"original": "WEIGHT OF BALLOON", "variable": "balloon_weight_kg"},
            {"original": "TYPE OF GAS USED IN BALLOON", "variable": "balloon_gas_type"},
            {"original": "TYPE OF PRESSURE SENSOR", "variable": "pressure_sensor_type"},
            {"original": "TYPE OF TEMPERATURE SENSOR", "variable": "temperature_sensor_type"},
            {"original": "TYPE OF HUMIDITY SENSOR", "variable": "humidity_sensor_type"},
            {"original": "SOFTWARE IDENTIFICATION AND VERSION NUMBER", "variable": "software_version"},
            {"original": "REASON FOR TERMINATION", "variable": "reason_for_termination"},
            {"original": "TRACKING TECHNIQUE/STATUS OF SYSTEM USED", "variable": "system_status"}
        ],
        "level": [
            {"original": "PRESSURE", "variable": "pressure_hPa"},
            {"original": "GEOPOTENTIAL HEIGHT", "variable": "height_m"},
            {"original": "TEMPERATURE/AIR TEMPERATURE", "variable": "temp_C"},
            {"original": "DEW-POINT TEMPERATURE", "variable": "dewpoint_C"},
            {"original": "WIND DIRECTION", "variable": "wind_dir_deg"},
            {"original": "WIND SPEED", "variable": "wind_speed_mps"},
            {"original": "LATITUDE DISPLACEMENT", "variable": "lat_disp"},
            {"original": "LONGITUDE DISPLACEMENT", "variable": "lon_disp"},
            {"original": "LONG TIME PERIOD OR DISPLACEMENT", "variable": "time_s"},
            {"original": "EXTENDED VERTICAL SOUNDING SIGNIFICANCE", "variable": "status_flag"}
        ]
    }

    # ==================================================================
    # 🔹 3. Choose active mapping (same as before)
    # ==================================================================
    active_map = mapping if (mapping and "meta" in mapping and "level" in mapping) else default_mapping

    # ==================================================================
    # 🔹 4. Original parsing logic (unchanged)
    # ==================================================================
    for line in decoded_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # ========== METADATA ==========
        matched_meta = False
        for item in active_map["meta"]:
            if item["original"] in line:
                key = item["variable"]
                val = _safe_float_tail(line)

                if "LATITUDE" in item["original"] and "005001" in line:
                    try:
                        station_lat = float(line.split()[-1])
                        meta[key] = station_lat
                    except Exception:
                        pass
                    matched_meta = True
                    break

                elif "LONGITUDE" in item["original"] and "006001" in line:
                    try:
                        station_lon = float(line.split()[-1])
                        meta[key] = station_lon
                    except Exception:
                        pass
                    matched_meta = True
                    break

                elif "SERIAL" in item["original"] or "VERSION" in item["original"]:
                    meta[key] = _extract_bytes_field(line)
                    matched_meta = True
                    break

                elif val is not None:
                    meta[key] = val
                    matched_meta = True
                    break

        if matched_meta:
            continue

        # --- Level separator ---
        if line.startswith("# ---") and current:
            levels.append(current)
            current = {}
            continue

        # ========== PER-LEVEL DATA ==========
        matched_level = False
        for lv in active_map["level"]:
            if lv["original"] in line:
                key = lv["variable"]
                val = _safe_float_tail(line)

                if "TEMPERATURE" in lv["original"]:
                    try: val = val - 273.15
                    except Exception: pass
                elif "DEW-POINT" in lv["original"]:
                    try: val = val - 273.15
                    except Exception: pass
                elif key == "pressure_hPa" and val is not None:
                    val = val / 100.0

                current[key] = val
                matched_level = True
                break

        if matched_level:
            continue

    if current:
        levels.append(current)

    df_meta = pd.DataFrame([meta])
    df_levels = pd.DataFrame(levels).replace({None: np.nan})

    if "time_s" in df_levels and "height_m" in df_levels:
        delta_h = df_levels["height_m"].diff()
        delta_t = df_levels["time_s"].diff()
        delta_t[delta_t <= 0] = np.nan
        df_levels["ascent_rate_mps"] = delta_h / delta_t
        df_levels.loc[df_levels["ascent_rate_mps"] > 20, "ascent_rate_mps"] = np.nan
        df_levels.loc[df_levels["ascent_rate_mps"] < 0, "ascent_rate_mps"] = np.nan

    if station_lat is not None and "lat_disp" in df_levels:
        df_levels["latitude"] = station_lat + df_levels["lat_disp"].fillna(0)
    if station_lon is not None and "lon_disp" in df_levels:
        df_levels["longitude"] = station_lon + df_levels["lon_disp"].fillna(0)

    if all(k in meta for k in ("year", "month", "day", "hour", "minute", "second")):
        try:
            dt_utc = datetime(
                int(meta["year"]), int(meta["month"]), int(meta["day"]),
                int(meta["hour"]), int(meta["minute"]), int(meta["second"]),
                tzinfo=timezone.utc
            )
            meta["launch_time"] = dt_utc.isoformat().replace("+00:00", "Z")
        except Exception:
            meta["launch_time"] = "-"
        for k in ("year", "month", "day", "hour", "minute", "second"):
            meta.pop(k, None)

    return pd.DataFrame([meta]), df_levels

# --- Mappings ---
REASON_MAP = {
    0: "Not specified",
    1: "Balloon burst",
    2: "Battery exhausted",
    3: "Ascent Stop",
    4: "Telemetry interrupted",
    5: "Manual termination",
    6: "Other",
    11: "Temperature KO",
}

SENSOR_MAPS = {
    "pressure": {0:"Unknown",1:"Aneroid",2:"Capacitive",3:"Other"},
    "temperature": {0:"Unknown",1:"Thermistor",2:"Platinum",3:"Other"},
    "humidity": {0:"Unknown",1:"Hair",2:"Capacitive",3:"Carbon",4:"Other"},
    "balloon": {0:"Unknown",1:"Latex",2:"Polyethylene",3:"Other"},
    "balloon_gas": {0:"Unknown",1:"Hydrogen",2:"Helium"},
    "balloon_manufacturer": {0:"Unknown",1:"Totex",2:"Kaysam",3:"Other"},
}

# --- FTP utility ---
def fetch_all_sites(ext_filter=None, limit=None, with_meta=False,
                    start_date=None, end_date=None):
    """
    Fetch list of radiosonde files from FTP or (if cached) from local SQLite DB.
    Untuk file hari ini dan kemarin, data selalu diambil langsung dari FTP (bypass cache).
    """
    _tz = None  # fallback (pakai UTC kalau zoneinfo tidak ada)
    #print("=========================================>",start_date,end_date)
    # --- Default window: today 00:00 UTC -> tomorrow 00:00 UTC ---
    if start_date is None or end_date is None:
        now_local = datetime.now(_tz) if _tz else datetime.utcnow().replace(tzinfo=timezone.utc)
        today_local_midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow_local_midnight = today_local_midnight + timedelta(days=1)

        def to_utc_naive(dt_local):
            if dt_local.tzinfo is None:
                return dt_local
            return dt_local.astimezone(timezone.utc).replace(tzinfo=None)

        if start_date is None:
            start_date = to_utc_naive(today_local_midnight)
        if end_date is None:
            end_date = to_utc_naive(tomorrow_local_midnight)

    result = {}
    
    exts = [e.lower() for e in (ext_filter or cfg.get("file_ext", [".bufr"]))]

    try:
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(cfg["base_path"])
            sites = ftp.nlst()

            for site in sites:
                site_files = []
                try:
                    ftp.cwd(f"{cfg['base_path']}/{site}")
                    all_files = ftp.nlst()
                    print(f"[DEBUG] {site}: total files returned by FTP = {len(all_files)}")

                    selected = [f for f in all_files if any(f.lower().endswith(ext) for ext in exts)]
                    items = []

                    for fname in selected:
                        dt_str = extract_date_from_filename(fname)
                        try:
                            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S UTC")
                            if start_date or end_date:
                                if start_date and dt < start_date:
                                    continue
                                if end_date and dt >= end_date:
                                    continue
                        except Exception:
                            dt = datetime.min
                        items.append((fname, dt))

                    # Sort newest first
                    items.sort(key=lambda x: x[1], reverse=True)
                    if limit:
                        items = items[:limit]
                    selected = [fname for fname, _ in items]

                    for fname in selected:
                        item = {
                            "name": fname,
                            "site": site,
                            "file_date": extract_date_from_filename(fname)
                        }

                        # ==========================================================
                        # 🔹 Jika with_meta=True, gunakan DB cache (kecuali hari ini & kemarin)
                        # ==========================================================
                        if with_meta:
                            try:
                                ftype = fname.split(".")[-1].lower()
                                if ftype not in ["bufr", "bfr", "bfh", "bin"]:
                                    ftype = "bufr"

                                # --- Tentukan apakah file hari ini atau kemarin ---
                                use_cache = True
                                try:
                                    file_dt = datetime.strptime(extract_date_from_filename(fname), "%Y-%m-%d %H:%M:%S UTC")
                                    now_utc = datetime.utcnow()
                                    yesterday_utc = now_utc - timedelta(days=1)
                                    if file_dt.date() in (now_utc.date(), yesterday_utc.date()):
                                        use_cache = False
                                except Exception:
                                    pass

                                # --- Cek di database hanya jika bukan hari ini/kemarin ---
                                cached = db_get(ftype, site, fname) if use_cache else None
                                if cached:
                                    df_meta, df_levels = cached
                                    print(f"[DB] ✅ fetch_all cache hit for {fname}")
                                else:
                                    # --- Ambil baru dari FTP ---
                                    local_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
                                    with open(local_path, "wb") as f:
                                        ftp.retrbinary(f"RETR " + fname, f.write)

                                    decoded = decode_bufr(local_path)
                                    df_meta, df_levels = parse_bufr(decoded, site=site)

                                    db_insert(ftype, site, fname,
                                              extract_date_from_filename(fname),
                                              df_meta, df_levels)
                                    print(f"[DB] 💾 cached {fname}")

                                # --- Ambil metadata dari DataFrame ---
                                if not df_meta.empty:
                                    meta_row = df_meta.iloc[0]

                                    # ✅ Launch time (auto-handle any format)
                                    raw_lt = meta_row.get("launch_time")
                                    lt = None
                                    if raw_lt is not None and not pd.isna(raw_lt):
                                        try:
                                            if isinstance(raw_lt, (datetime, pd.Timestamp)):
                                                lt = raw_lt.astimezone(timezone.utc) if raw_lt.tzinfo else raw_lt.replace(tzinfo=timezone.utc)
                                            elif isinstance(raw_lt, (int, float)):
                                                lt = pd.to_datetime(raw_lt, utc=True, errors="coerce")
                                            elif isinstance(raw_lt, str):
                                                clean_lt = raw_lt.strip().replace("Z", "").replace(" UTC", "")
                                                lt = pd.to_datetime(clean_lt, utc=True, errors="coerce")
                                        except Exception as e:
                                            print(f"[WARN] launch_time parse failed for {fname}: {e}")

                                    if lt is not None and pd.notna(lt):
                                        item["launch_time"] = lt.strftime("%Y-%m-%d %H:%M:%S UTC")
                                    else:
                                        item["launch_time"] = "-"

                                    # Radiosonde serial number
                                    sn = meta_row.get("radiosonde_serial_number")
                                    if sn:
                                        item["radiosonde_serial_number"] = sn

                                    # Reason for termination
                                    term_code = meta_row.get("reason_for_termination")
                                    if pd.notna(term_code):
                                        code = int(term_code)
                                        meaning = REASON_MAP.get(code, "Unknown")
                                        item["reason_for_termination"] = f"{code} – {meaning}"

                                    # Sensor & balloon types
                                    alias_groups = {
                                        "pressure": ["pressure_sensor_type", "type_of_pressure_sensor"],
                                        "temperature": ["temperature_sensor_type", "type_of_temperature_sensor"],
                                        "humidity": ["humidity_sensor_type", "type_of_humidity_sensor"],
                                        "balloon": ["balloon_type", "type_of_balloon"],
                                        "balloon_gas": ["balloon_gas_type", "type_of_gas_used_in_balloon"],
                                        "balloon_manufacturer": ["balloon_manufacturer"],
                                    }
                                    for group, keys in alias_groups.items():
                                        mapping = SENSOR_MAPS[group]
                                        for k in keys:
                                            code = meta_row.get(k)
                                            if pd.notna(code):
                                                item[k] = f"{int(code)} – {mapping.get(int(code), 'Unknown')}"

                                # --- Flight issues ---
                                issues = analyze_flight(df_meta, df_levels)
                                if issues:
                                    item["flight_issues"] = issues

                                # --- Derived flight metadata ---
                                try:
                                    if not df_levels.empty:
                                        # End pressure
                                        end_pressure = df_levels["pressure_hPa"].dropna().min()
                                        if pd.notna(end_pressure):
                                            item["end_pressure"] = round(float(end_pressure), 1)

                                        # Max height
                                        max_height = df_levels["height_m"].dropna().max()
                                        if pd.notna(max_height):
                                            item["max_height"] = round(float(max_height), 0)

                                        # End time
                                        if "time_s" in df_levels and not df_meta.empty:
                                            if "launch_time" in item and item["launch_time"] != "-":
                                                try:
                                                    lt = pd.to_datetime(item["launch_time"].replace(" UTC", ""), utc=True)
                                                    end_time = lt + pd.to_timedelta(df_levels["time_s"].max(), unit="s")
                                                    item["end_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                                                except Exception as e:
                                                    print(f"[WARN] end_time calc failed for {fname}: {e}")

                                        # Distance from station
                                        if {"latitude", "longitude"} <= set(df_levels.columns) and not df_meta.empty:
                                            last_lat = df_levels["latitude"].dropna().iloc[-1]
                                            last_lon = df_levels["longitude"].dropna().iloc[-1]
                                            st_lat = meta_row.get("station_lat")
                                            st_lon = meta_row.get("station_lon")
                                            if st_lat and st_lon and pd.notna(last_lat) and pd.notna(last_lon):
                                                dist = geodesic((st_lat, st_lon), (last_lat, last_lon)).km
                                                item["end_distance"] = round(float(dist), 1)

                                        # Avg ascent rate
                                        if "height_m" in df_levels and "time_s" in df_levels:
                                            elapsed = df_levels["time_s"].max() - df_levels["time_s"].min()
                                            if elapsed > 0 and pd.notna(max_height):
                                                ascent_rate = max_height / elapsed
                                                item["avg_ascent_rate"] = round(float(ascent_rate), 2)
                                except Exception as e:
                                    print("Extra metadata calc failed in fetch_all_sites:", e)

                            except Exception as e:
                                item["launch_time"] = f"Error: {e}"
                                item["flight_issues"] = [f"Error: {e}"]

                        # ==========================================================
                        # ✅ Append hasil per file
                        # ==========================================================
                        site_files.append(item)

                    ftp.cwd(cfg["base_path"])
                    result[site] = site_files

                except Exception as e:
                    result[site] = [{"name": f"Error: {e}", "site": site, "file_date": "-"}]

    except Exception as e:
        result["GLOBAL"] = [{"name": f"FTP Error: {e}", "site": "GLOBAL", "file_date": "-"}]

    return result

# --- API routes ---
@app.route("/api/sites")
@login_required
def api_sites():
    ext = request.args.get("ext") or None
    limit = request.args.get("limit")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    sites = fetch_all_sites(
        ext_filter=[ext] if ext else None,
        limit=int(limit) if limit else None,
        with_meta=False,
        start_date=start_dt,
        end_date=end_dt
    )
    return jsonify(sites)

@app.route("/api/sites_with_meta")
@login_required
def api_sites_with_meta():
    ext = request.args.get("ext") or None
    limit = request.args.get("limit")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    sites = fetch_all_sites(
        ext_filter=[ext] if ext else None,
        limit=int(limit) if limit else None,
        with_meta=True,
        start_date=start_dt,
        end_date=end_dt
    )
    # 🔹 Merge manufactured info from MODEM_LOOKUP based on serial number
    for site, files in sites.items():
        for f in files:
            serial_raw = f.get("radiosonde_serial_number")
            
            sn_int = parse_serial_to_int(serial_raw)
            if sn_int and sn_int in MODEM_LOOKUP:
                f["manufactured"] = MODEM_LOOKUP[sn_int]
            else:
                f["manufactured"] = "-"
    
    return jsonify(sites)

@app.route("/api/latest_status")
@login_required
def api_latest_status():
    """
    Ambil status terakhir dari setiap site radiosonde.
    Mengembalikan 6 site terbaru dengan kolom:
    Site, Launch Time, End Time, Status, Termination,
    End Pressure, Max Height, End Distance, Ascent Rate.
    """
    now_utc = datetime.utcnow()
    start_date = now_utc - timedelta(days=1)
    end_date = now_utc + timedelta(days=1)

    # 🔹 Ambil hanya beberapa file terbaru dari window waktu 2 hari
    sites = fetch_all_sites(with_meta=True, ext_filter= [".bfr",".bin"], start_date=start_date, end_date=end_date)

    summary = []
    for site, files in sites.items():
        if not files:
            continue
        f = files[0]
        summary.append({
            "site": site,
            "launch_time": f.get("launch_time", "-"),
            "end_time": f.get("end_time", "-"),
            "status": "✅ OK" if f.get("flight_issues") == ["OK"] else "⚠️ Check",
            "termination": f.get("reason_for_termination", "-"),
            "end_pressure": f.get("end_pressure", "-"),
            "max_height": f.get("max_height", "-"),
            "end_distance": f.get("end_distance", "-"),
            "ascent_rate": f.get("avg_ascent_rate", "-")
        })
    # Urutkan biar tampil konsisten (misal abjad)
    summary = sorted(summary, key=lambda x: x["site"])[:6]
    return jsonify(summary)

@app.route("/api/status")
@login_required
def api_status():
    """
    Membaca status balon dari file EOSCAN*.log di FTP tiap site.
    Status diambil dari pesan terakhir yang mengandung kata kunci tertentu.
    Jika waktu 'preparing' (Sonde ON) ditemukan di jam di luar 00Z/12Z,
    dianggap log menggunakan waktu lokal dan dikonversi ke UTC.
    """
    import ftplib, io, re
    from datetime import datetime, timedelta

    cfg = CONFIG["ftp"]
    sites = SITE_LIST
    results = []

    # Offset per site
    site_utc_offset = {
        "aceh": 7,
        "cilacap": 7,
        "pangkalanbun": 7,
        "tarakan": 8,
        "sorong": 9,
        "ranai": 7
    }

    status_patterns = {
        "offline": ["station off", "eoscan manual close"],
        "inflight": ["start sounding"],
        "preparing": ["sonde on"],
        "online": ["station ok", "running eoscan", "sounding over"],
    }

    try:
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])

            for site in sites:
                status = "offline"
                update = "-"
                try:
                    ftp.cwd(f"{cfg['base_path']}/{site}")
                    files = ftp.nlst()
                    log_files = [f for f in files if re.match(r"EOSCAN.*\.log$", f, re.IGNORECASE)]
                    if not log_files:
                        raise FileNotFoundError("Tidak ada file EOSCAN*.log")

                    log_files.sort(reverse=True)
                    latest_log = log_files[0]

                    # Ambil tanggal dari nama file
                    date_match = re.search(r"(\d{1,2}) ([A-Za-z]{3}) (\d{2})", latest_log)
                    file_date = None
                    if date_match:
                        day, mon, yy = date_match.groups()
                        try:
                            file_date = datetime.strptime(f"{day} {mon} 20{yy}", "%d %b %Y")
                        except:
                            pass

                    # Unduh isi file log
                    f = io.BytesIO()
                    ftp.retrbinary(f"RETR " + latest_log, f.write)
                    f.seek(0)
                    lines = f.read().decode(errors="ignore").splitlines()

                    candidates = []
                    for line in lines:
                        m = re.match(r"(\d{2}:\d{2}:\d{2})\s+\d+\s+(.*)", line.strip())
                        if not m:
                            continue
                        time_str, msg = m.groups()
                        msg_lower = msg.lower()

                        for s, keys in status_patterns.items():
                            if any(k in msg_lower for k in keys):
                                candidates.append((time_str, s))
                                break

                    if not candidates:
                        results.append({"site": site, "status": "unknown", "update": file_date.strftime("%Y-%m-%d") if file_date else "-"})
                        continue

                    # Ambil status terakhir
                    last_time, last_status = candidates[-1]
                    status = last_status
                    time_dt = None

                    if file_date:
                        try:
                            time_dt = datetime.strptime(f"{file_date.strftime('%Y-%m-%d')} {last_time}", "%Y-%m-%d %H:%M:%S")
                        except:
                            pass

                    # Jika status "preparing" (Sonde ON) dan jamnya bukan sekitar 00Z/12Z, konversi lokal → UTC
                    if status == "preparing" and time_dt:
                        if not (23 <= time_dt.hour or time_dt.hour <= 1 or 11 <= time_dt.hour <= 13):
                            offset = site_utc_offset.get(site, 7)
                            time_dt_utc = time_dt - timedelta(hours=offset)
                            update = time_dt_utc.strftime("%Y-%m-%d %H:%M UTC")
                        else:
                            update = time_dt.strftime("%Y-%m-%d %H:%M UTC")
                    elif time_dt:
                        update = time_dt.strftime("%Y-%m-%d %H:%M UTC")
                    else:
                        update = f"{file_date.strftime('%Y-%m-%d')} {last_time} UTC" if file_date else last_time

                except Exception as e:
                    print(f"⚠️ Gagal baca {site}: {e}")
                    status = "offline"

                results.append({
                    "site": site,
                    "status": status,
                    "update": update
                })

    except Exception as e:
        print("❌ FTP connection error:", e)

    return jsonify(results)

@app.route("/api/filter", methods=["POST"])
@login_required
def api_filter():
    """
    Mengambil file radiosonde dari FTP berdasarkan filter:
    - site: nama folder
    - date: YYYY-MM-DD
    - hour: 00 atau 12
    - ftype: ekstensi file (.bfr/.bufr/.bfh/.bin)
    """
    cfg = CONFIG["ftp"]
    site = request.form.get("site")
    date = request.form.get("date")
    hour = request.form.get("hour")
    ftype = request.form.get("ftype", "").lower()

    if not site or not date or not hour or not ftype:
        return jsonify({"error": "Missing site/date/hour/ftype"}), 400

    try:
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(f"{cfg['base_path']}/{site}")

            all_files = ftp.nlst()
            target_files = []

            date_str = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
            pattern = f"{date_str}{hour}"

            for fname in all_files:
                if pattern in fname and fname.lower().endswith(ftype):
                    target_files.append(fname)

            if not target_files:
                return jsonify({
                    "status": "no_files",
                    "message": f"Tidak ditemukan file {ftype} untuk {site} {pattern}"
                })

            return jsonify({
                "status": "ok",
                "files": target_files
            })

    except Exception as e:
        print("❌ FTP filter error:", e)
        return jsonify({"status": "error", "message": str(e)})

def download_and_process(site, filename):
    """
    Fetch BUFR/BFR/BFH/BIN data either from local SQLite DB (if cached)
    or from FTP (then decode via pybufrkit and save into DB).
    Return results into current user's session store.
    """
    ftype = filename.split(".")[-1].lower()
    if ftype not in ["bufr", "bfr", "bfh", "bin"]:
        ftype = "bufr"

    cfg = CONFIG["ftp"]
    local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    store = get_user_store()

    try:
        # ==========================================================
        # 1️⃣ Coba baca dari DATABASE (cache)
        # ==========================================================
        cached = db_get(ftype, site, filename)
        if cached:
            print(f"[DB] ✅ Loaded {filename} from cache ({ftype})")
            df_meta, df_levels = cached
            issues = analyze_flight(df_meta, df_levels)

            store["metadata"] = df_meta.to_dict("records")[0] if not df_meta.empty else {}
            store["metadata"]["flight_issues"] = issues
            store["levels"] = df_levels.to_dict("records") if not df_levels.empty else []
            return  # 🚀 Selesai — tidak perlu FTP atau decode

        # ==========================================================
        # 2️⃣ Jika belum ada di DB, download dari FTP
        # ==========================================================
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(f"{cfg['base_path']}/{site}")
            with open(local_path, "wb") as f:
                ftp.retrbinary(f"RETR " + filename, f.write)
        print(f"[FTP] ✅ Downloaded {filename} to {local_path}")

        # ==========================================================
        # 3️⃣ Decode & parse pakai pybufrkit sekali saja
        # ==========================================================
        decoded = decode_bufr(local_path)
        df_meta, df_levels = parse_bufr(decoded, site=site)
        issues = analyze_flight(df_meta, df_levels)

        # ==========================================================
        # 4️⃣ Simpan hasil decode ke DATABASE untuk cache
        # ==========================================================
        db_insert(ftype, site, filename, extract_date_from_filename(filename), df_meta, df_levels)
        print(f"[DB] 💾 Cached {filename} ({ftype})")

        # ==========================================================
        # 5️⃣ Isi user session store (hasil decode)
        # ==========================================================
        store["metadata"] = df_meta.to_dict("records")[0] if not df_meta.empty else {}
        meta = store["metadata"]

        # --- Normalize radiosonde frequency (Hz → MHz) ---
        if "radiosonde_operating_frequency" in meta:
            try:
                hz_val = float(meta["radiosonde_operating_frequency"])
                meta["radiosonde_operating_frequency"] = f"{hz_val/1e6:.3f} MHz"
            except Exception:
                pass

        # === Map numeric codes to "code – meaning" (keep original key names) ===
        # Reason for termination
        if "reason_for_termination" in meta and meta["reason_for_termination"] not in (None, ""):
            try:
                code = int(meta["reason_for_termination"])
                reason_map = {
                    0: "Not specified",
                    1: "Balloon burst",
                    2: "Battery exhausted",
                    3: "Ascent Stop",
                    4: "Telemetry interrupted",
                    5: "Manual termination",
                    6: "Other",
                    11: "Temperature KO",
                }
                meta["reason_for_termination"] = f"{code} – {reason_map.get(code, 'Unknown')}"
            except Exception:
                pass

        # --- Sensor/Balloon type code tables ---
        sensor_maps = {
            "pressure": {0:"Unknown",1:"Aneroid",2:"Capacitive",3:"Other"},
            "temperature": {0:"Unknown",1:"Thermistor",2:"Platinum",3:"Other"},
            "humidity": {0:"Unknown",1:"Hair",2:"Capacitive",3:"Carbon",4:"Other"},
            "balloon": {0:"Unknown",1:"Latex",2:"Polyethylene",3:"Other"},
            "balloon_gas": {0:"Unknown",1:"Hydrogen",2:"Helium"},
            "balloon_manufacturer": {0:"Unknown",1:"Totex",2:"Kaysam",3:"Other"},
        }
        alias_groups = {
            "pressure": ["pressure_sensor_type", "type_of_pressure_sensor"],
            "temperature": ["temperature_sensor_type", "type_of_temperature_sensor"],
            "humidity": ["humidity_sensor_type", "type_of_humidity_sensor"],
            "balloon": ["balloon_type", "type_of_balloon"],
            "balloon_gas": ["balloon_gas_type", "type_of_gas_used_in_balloon"],
            "balloon_manufacturer": ["balloon_manufacturer"],
        }
        for group, keys in alias_groups.items():
            mapping = sensor_maps[group]
            for k in keys:
                if k in meta and meta[k] not in (None, ""):
                    try:
                        code = int(meta[k])
                        meta[k] = f"{code} – {mapping.get(code, 'Unknown')}"
                    except Exception:
                        pass  # keep original if not int

        # --- Derived flight metadata (end_time, end_pressure, etc.) ---
        try:
            if not df_levels.empty:
                # End pressure
                if "pressure_hPa" in df_levels:
                    end_p = df_levels["pressure_hPa"].dropna().min()
                    if pd.notna(end_p):
                        meta["end_pressure"] = f"{end_p:.1f} hPa"

                # Max height
                if "height_m" in df_levels:
                    max_h = df_levels["height_m"].dropna().max()
                    if pd.notna(max_h):
                        meta["max_height"] = f"{max_h:.0f} m"

                # End time = launch_time + max(time_s)
                end_time = None
                if "time_s" in df_levels and not df_meta.empty:
                    lt = pd.to_datetime(df_meta.iloc[0].get("launch_time"))
                    if pd.notna(lt):
                        end_time = lt + pd.to_timedelta(df_levels["time_s"].max(), unit="s")
                        meta["end_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")

                # Distance from station to last lat/lon
                if {"latitude","longitude"} <= set(df_levels.columns) and not df_meta.empty:
                    last_lat = df_levels["latitude"].dropna().iloc[-1] if df_levels["latitude"].notna().any() else None
                    last_lon = df_levels["longitude"].dropna().iloc[-1] if df_levels["longitude"].notna().any() else None
                    st_lat = meta.get("station_lat")
                    st_lon = meta.get("station_lon")
                    if all(v is not None for v in (last_lat,last_lon,st_lat,st_lon)):
                        try:
                            dist_km = geodesic((float(st_lat), float(st_lon)), (float(last_lat), float(last_lon))).km
                            meta["end_distance"] = f"{dist_km:.1f} km"
                        except Exception:
                            pass

                # Avg ascent rate
                if "height_m" in df_levels and "time_s" in df_levels and df_levels["time_s"].notna().any():
                    max_h = df_levels["height_m"].dropna().max()
                    elapsed = df_levels["time_s"].max() - df_levels["time_s"].min()
                    if pd.notna(max_h) and elapsed and elapsed > 0:
                        meta["avg_ascent_rate"] = f"{(max_h/elapsed):.2f} m/s"
        except Exception as e:
            print("Extra metadata calc failed in download_and_process:", e)

        # --- Save to user store & exit ---
        meta["flight_issues"] = issues
        store["levels"] = df_levels.to_dict("records") if not df_levels.empty else []

    except Exception as e:
        print(f"[ERROR] download_and_process failed for {filename}: {e}")
        store["metadata"], store["levels"] = {}, []

def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan

def analyze_flight(df_meta, df_levels):
    issues = []

    # --- Temperature check ---
    if df_levels["temp_C"].isna().sum() > len(df_levels) * 0.3:
        issues.append("Bad Temp: too many missing values")
    if (df_levels["temp_C"] > 60).any():
        issues.append("Temp KO: unrealistic values > ±60 °C")

    # --- Ascent stop ---
    if "ascent_rate_mps" in df_levels:
        if (df_levels["ascent_rate_mps"] <= 0).rolling(5, min_periods=1).sum().max() >= 5:
            issues.append("Ascent Stop: balloon stopped rising")

    # --- Max height check ---
    if "pressure_hPa" in df_levels:
        min_p = df_levels["pressure_hPa"].min()
        if min_p > 100:
            issues.append("Not reaching 100 hPa")
        if min_p > 30:
            issues.append("Not reaching 30 hPa")

    # --- GPS check ---
    if "latitude" in df_levels and df_levels["latitude"].isna().sum() > len(df_levels) * 0.2:
        issues.append("GPS Fail: too many missing positions")

    return issues or ["OK"]

def extract_date_from_filename(fname: str):
    # Match 14-digit date string like 20250905000000
    m = re.search(r"(\d{14})", fname)
    if m:
        s = m.group(1)
        try:
            dt = datetime.strptime(s, "%Y%m%d%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            return "-"
    return "-"

def generate_wmo_temp(df_meta, df_levels):
    """
    Generate WMO TEMP message (TTAA, TTBB, TTCC, TTDD).
    Encoded into real 5-digit group format (5 groups per line, WMO style).
    """

    if df_meta.empty or df_levels.empty:
        return "No data"

    block = int(df_meta.iloc[0].get("wmo_block", 99))
    station = int(df_meta.iloc[0].get("wmo_station", 999))
    d = df_meta.iloc[0].get("day", 1)
    h = df_meta.iloc[0].get("hour", 0)

    # Sort levels by pressure descending
    df_levels = df_levels.sort_values("pressure_hPa", ascending=False)

    # --- Helpers ---
    def encode_temp_group(p, t, td):
        if pd.isna(p) or pd.isna(t):
            return "/////"
        p100 = int(round(p / 10)) % 1000  # compress to 3 digits
        t10 = int(round(t * 10)) % 1000
        td_dep = int(round((t - td) * 10)) if pd.notna(td) else 99
        return f"{p100:03d}{t10:02d}{td_dep:02d}"[:5]

    def encode_wind_group(wd, ws):
        if pd.isna(wd) or pd.isna(ws):
            return "/////"
        dd = int(round(wd / 10)) % 36  # tens of degrees
        ff = int(round(ws)) % 1000
        return f"{dd:02d}{ff:03d}"[:5]

    def pack_groups(groups):
        """Pack groups 5 per line (WMO style)."""
        lines = []
        for i in range(0, len(groups), 5):
            lines.append(" ".join(groups[i:i+5]))
        return lines

    lines = []

    # --- TTAA ---
    lines.append(f"TTAA {block:02d}{station:03d} {d:02d}{h:02d}00")
    groups = []
    for _, row in df_levels[df_levels["pressure_hPa"] >= 100].iterrows():
        tgrp = encode_temp_group(row.get("pressure_hPa"), row.get("temp_C"), row.get("dewpoint_C"))
        wgrp = encode_wind_group(row.get("wind_dir_deg"), row.get("wind_speed_mps"))
        groups.extend([tgrp, wgrp])
    lines.extend(pack_groups(groups))

    # --- TTBB ---
    lines.append(f"TTBB {block:02d}{station:03d} {d:02d}{h:02d}00")
    groups = []
    for _, row in df_levels[(df_levels["pressure_hPa"] < 1000) & (df_levels["pressure_hPa"] > 100)].iterrows():
        if pd.notna(row.get("temp_C")) and pd.notna(row.get("dewpoint_C")):
            groups.append(encode_temp_group(row.get("pressure_hPa"), row.get("temp_C"), row.get("dewpoint_C")))
    if groups:
        lines.extend(pack_groups(groups))

    # --- TTCC ---
    lines.append(f"TTCC {block:02d}{station:03d} {d:02d}{h:02d}00")
    groups = []
    for _, row in df_levels[df_levels["pressure_hPa"] < 100].iterrows():
        tgrp = encode_temp_group(row.get("pressure_hPa"), row.get("temp_C"), row.get("dewpoint_C"))
        wgrp = encode_wind_group(row.get("wind_dir_deg"), row.get("wind_speed_mps"))
        groups.extend([tgrp, wgrp])
    if groups:
        lines.extend(pack_groups(groups))

    # --- TTDD ---
    lines.append(f"TTDD {block:02d}{station:03d} {d:02d}{h:02d}00")
    groups = []
    for _, row in df_levels.iterrows():
        if pd.notna(row.get("wind_dir_deg")) and pd.notna(row.get("wind_speed_mps")):
            groups.append(encode_wind_group(row.get("wind_dir_deg"), row.get("wind_speed_mps")))
    if groups:
        lines.extend(pack_groups(groups))

    lines.append("NNNN")
    return "\n".join(lines)

def combine_t_p_files(site, base_name):
    """
    Cari file T*.X dan P*.X dari FTP dengan timestamp sama (contoh: 2025100900).
    Gabungkan konten keduanya jadi satu TXT file sementara untuk proses WMO.
    """
    cfg = CONFIG["ftp"]
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    combined_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{base_name}_TP.txt")

    pattern = re.search(r"(\d{10})", base_name)
    if not pattern:
        raise ValueError("No valid timestamp found in filename")
    timestamp = pattern.group(1)

    try:
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(f"{cfg['base_path']}/{site}")
            all_files = ftp.nlst()

            # Cari file T dan P yang cocok
            t_files = [f for f in all_files if re.search(fr"T\d+.*{timestamp}.*\.[xX]$", f)]
            p_files = [f for f in all_files if re.search(fr"P\d+.*{timestamp}.*\.[xX]$", f)]

            if not t_files or not p_files:
                raise FileNotFoundError(f"T or P file not found for timestamp {timestamp}")

            t_file = t_files[0]
            p_file = p_files[0]

            # Unduh dua file
            t_data = io.BytesIO()
            ftp.retrbinary(f"RETR {t_file}", t_data.write)
            p_data = io.BytesIO()
            ftp.retrbinary(f"RETR {p_file}", p_data.write)

            # Gabungkan kontennya
            with open(combined_path, "wb") as out:
                out.write(t_data.getvalue())
                out.write(b"\n")
                out.write(p_data.getvalue())

            print(f"✅ Combined {t_file} + {p_file} → {combined_path}")
            return combined_path

    except Exception as e:
        print("❌ combine_t_p_files failed:", e)
        raise

def generate_weather_analysis(df):
    text = []
    # --- Cloud layers ---
    clouds = df[df['rh_percent'] > 90]
    if not clouds.empty:
        base = clouds['height_m'].min()/1000
        top = clouds['height_m'].max()/1000
        text.append(f"Cloud layer detected between {base:.1f}–{top:.1f} km (RH > 90%).")

    # --- Freezing level ---
    df['temp_shift'] = df['temperature_C'].shift()
    zero_cross = df[(df['temperature_C'] * df['temp_shift']) < 0]
    if not zero_cross.empty:
        zf = zero_cross['height_m'].iloc[0]/1000
        text.append(f"Freezing level around {zf:.1f} km.")

    # --- Instability zones ---
    df['lapse_rate'] = -df['temperature_C'].diff() / (df['height_m'].diff()/1000)
    unstable = df[df['lapse_rate'] > 7]
    if not unstable.empty:
        minz, maxz = unstable['height_m'].min()/1000, unstable['height_m'].max()/1000
        text.append(f"Unstable layer (lapse rate > 7°C/km) from {minz:.1f}–{maxz:.1f} km.")

    # --- Wind shear ---
    df['wind_speed_diff'] = df['wind_speed_mps'].diff()
    df['shear_rate'] = df['wind_speed_diff'] / (df['height_m'].diff()/1000)
    strong_shear = df[df['shear_rate'] > 10]
    if not strong_shear.empty:
        minz, maxz = strong_shear['height_m'].min()/1000, strong_shear['height_m'].max()/1000
        text.append(f"Strong wind shear zone ({minz:.1f}–{maxz:.1f} km). Turbulence risk.")

    return "<br>".join(text) if text else "Atmospheric profile indicates generally stable conditions."

def download_from_ftp(site, filename):
    """Fetch file from FTP and return local path only (no processing)."""
    cfg = CONFIG["ftp"]
    local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(f"{cfg['base_path']}/{site}")
            with open(local_path, "wb") as f:
                ftp.retrbinary(f"RETR " + filename, f.write)
    except Exception as e:
        raise RuntimeError(f"FTP download error: {e}")
    return local_path

# --- helper: ambil angka saja, aman untuk "12345", "12345.0", "SN-12345", dll.
def parse_serial_to_int(val):
    if val is None:
        return None
    s = str(val).strip()
    # kalau float "12345.0" → "12345"
    try:
        f = float(s)
        return int(f)
    except:
        pass
    # jika format campur huruf, ambil digitnya
    digits = re.sub(r"\D", "", s)
    return int(digits) if digits.isdigit() else None

# --- load JSON list modem → dict {serial_int: manufactured_str}
def load_modem_lookup(json_path="list_data_modem.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lookup = {}
    for row in data:
        low = {str(k).strip().lower(): v for k, v in row.items()}

        serial_raw = (
            low.get("nomor serial")
            or low.get("serial")
            or low.get("no seri")
            or low.get("sn")
        )
        manufactured_raw = (
            low.get("manufactured")
            or low.get("tanggal manufactured")
            or low.get("tgl manufactured")
            or low.get("mfg")
        )

        sn_int = parse_serial_to_int(serial_raw)
        manufactured_str = excel_date_to_str(manufactured_raw)
        if sn_int:
            lookup[sn_int] = manufactured_str
    return lookup

def excel_date_to_str(val):
    """Konversi nilai Excel float (misal 45680.0) menjadi YYYY-MM-DD."""
    try:
        if isinstance(val, (int, float)) and val > 30000:
            base = datetime(1899, 12, 30)
            return (base + timedelta(days=int(val))).strftime("%Y-%m-%d")
        if isinstance(val, str):
            return val.strip()
    except Exception:
        pass
    return "-"

MODEM_LOOKUP = load_modem_lookup("list_data_modem.json")

@app.route("/download/<site>/<filename>")
@login_required
def download_file(site, filename):
    cfg = CONFIG["ftp"]
    local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(f"{cfg['base_path']}/{site}")
            with open(local_path, "wb") as f:
                ftp.retrbinary(f"RETR " + filename, f.write)
    except Exception as e:
        return f"FTP download error: {e}", 500

    # Only support .bfr download for now
    return send_file(local_path, as_attachment=True)

# --- Routes ---
@app.route("/dashboard")
@login_required
@page_access_required("dashboard")
def dashboard():
    # Default to ".bfr" if ext is missing or empty
    selected_ext = request.args.get("ext")
    limit = request.args.get("limit", type=int)
    
    if selected_ext is not None:
        session["dash_ext"] = selected_ext
    if limit is not None:
        session["dash_limit"] = limit
    
    selected_ext = session.get("dash_ext", ".bfr")
    limit = session.get("dash_limit", 2)
    
    sites = fetch_all_sites(ext_filter=selected_ext, limit=limit)
    return render_template(
        "dashboard.html",
        sites=sites,
        selected_ext=selected_ext,
        limit=limit
    )

@app.route("/api/filemeta/<site>/<filename>")
@login_required
def file_metadata(site, filename):
    """
    Return minimal metadata (launch_time, serial_number) from BUFR cache.
    - Cek dulu di SQLite cache
    - Jika belum ada → download dari FTP → decode → parse → simpan ke DB
    """
    try:
        ftype = filename.split(".")[-1].lower()
        if ftype not in ["bufr", "bfr", "bfh", "bin"]:
            ftype = "bufr"

        # ==========================================================
        # 1️⃣ Coba dari database dulu
        # ==========================================================
        cached = db_get(ftype, site, filename)
        if cached:
            print(f"[DB] ✅ filemeta cache hit for {filename}")
            df_meta, _ = cached
        else:
            # ======================================================
            # 2️⃣ Jika tidak ada di DB → ambil dari FTP + decode
            # ======================================================
            local_path = download_from_ftp(site, filename)
            decoded = decode_bufr(local_path)
            df_meta, df_levels = parse_bufr(decoded, site=site)

            # ======================================================
            # 3️⃣ Simpan hasil parse ke DB
            # ======================================================
            db_insert(
                ftype,
                site,
                filename,
                extract_date_from_filename(filename),
                df_meta,
                df_levels
            )
            print(f"[DB] 💾 Cached {filename} into {ftype} table")

        # ==========================================================
        # 4️⃣ Ambil data dari hasil decode/cache (df_meta)
        # ==========================================================
        meta = df_meta.to_dict("records")[0] if not df_meta.empty else {}
        return jsonify({
            "launch_time": meta.get("launch_time", "-"),
            "radiosonde_serial_number": meta.get("radiosonde_serial_number", "-")
        })

    except Exception as e:
        print(f"[ERROR] file_metadata failed for {filename}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/load_from_ftp/<site>/<filename>")
@login_required
def load_from_ftp(site, filename):
    download_and_process(site, filename)
    return redirect(url_for("map_view", t=int(time.time())))

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        f = request.files["rasonfiles"]
        if not f or not f.filename:
            return redirect(url_for("index"))
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
        f.save(filepath)

        decoded = decode_bufr(filepath)
        df_meta, df_levels = parse_bufr(decoded)

        issues = analyze_flight(df_meta, df_levels)
        store = get_user_store()
        store["metadata"] = df_meta.to_dict("records")[0] if not df_meta.empty else {}
        store["metadata"]["flight_issues"] = issues
        store["levels"] = df_levels.to_dict("records") if not df_levels.empty else []

        # 🔧 konversi frequency Hz → MHz
        if "radiosonde_operating_frequency" in store["metadata"]:
            try:
                hz_val = float(store["metadata"]["radiosonde_operating_frequency"])
                mhz_val = hz_val / 1e6
                store["metadata"]["radiosonde_operating_frequency"] = f"{mhz_val:.3f} MHz"
            except Exception:
                pass

        # ✅ setelah upload berhasil, redirect ke /map?t=timestamp
        return redirect(url_for("map_view", t=int(time.time())))

    # GET: tampilkan halaman utama / dashboard
    return render_template("main.html", user=session.get("user"))

@app.route("/value")
@login_required
def rason_value():
    store = get_user_store()
    levels = store["levels"]
    if not levels:
        return jsonify({"error": "No radiosonde"}), 404
    idx = int(request.args.get("frame", 0)) % len(levels)
    return jsonify(levels[idx])

@app.route("/metadata")
@login_required
def rason_metadata():
    store = get_user_store()
    meta = store.get("metadata", {})
    levels = store.get("levels", [])

    if not meta:
        return jsonify({"error": "No metadata"}), 404

    import pandas as pd, numpy as np
    from geopy.distance import geodesic

    if levels:
        df = pd.DataFrame(levels)

        # --- End pressure ---
        if "end_pressure" not in meta or meta["end_pressure"] in ("-", None, ""):
            if "pressure_hPa" in df:
                end_p = df["pressure_hPa"].dropna().min()
                if pd.notna(end_p):
                    meta["end_pressure"] = f"{end_p:.1f} hPa"

        # --- Max height ---
        if "max_height" not in meta or meta["max_height"] in ("-", None, ""):
            if "height_m" in df:
                max_h = df["height_m"].dropna().max()
                if pd.notna(max_h):
                    meta["max_height"] = f"{max_h:.0f} m"

        # --- End distance ---
        if "end_distance" not in meta or meta["end_distance"] in ("-", None, ""):
            if {"latitude", "longitude"} <= set(df.columns) and \
               meta.get("station_lat") and meta.get("station_lon"):
                try:
                    last_lat = df["latitude"].dropna().iloc[-1]
                    last_lon = df["longitude"].dropna().iloc[-1]
                    st_lat = float(meta["station_lat"])
                    st_lon = float(meta["station_lon"])
                    dist = geodesic((st_lat, st_lon), (last_lat, last_lon)).km
                    meta["end_distance"] = f"{dist:.1f} km"
                except Exception:
                    pass

        # --- Avg ascent rate ---
        if "avg_ascent_rate" not in meta or meta["avg_ascent_rate"] in ("-", None, ""):
            if {"height_m", "time_s"} <= set(df.columns):
                elapsed = df["time_s"].max() - df["time_s"].min()
                max_h = df["height_m"].dropna().max()
                if pd.notna(max_h) and elapsed and elapsed > 0:
                    meta["avg_ascent_rate"] = f"{(max_h/elapsed):.2f} m/s"

    return jsonify(meta)

@app.route("/all_levels")
@login_required
def all_levels_route():
    store = get_user_store()
    return jsonify(store["levels"])

@app.route("/download_wmo/<site>/<filename>")
@login_required
def download_wmo(site, filename):
    """
    Download combined WMO message.
    - Jika ada file T*.X dan P*.X di FTP → gabungkan isi mentah (tanpa header tambahan)
    - Jika tidak ada → generate TTAA–TTDD + PPAA–PPDD lengkap dengan header sandi Meteomodem,
      dimana kode ID (misal IUKG51 WITT) diambil otomatis dari nama file BUFR.
    """
    cfg = CONFIG["ftp"]
    try:
        ts_match = re.search(r"(\d{10,12})", filename)
        timestamp = ts_match.group(1) if ts_match else None
        combined_txt = ""

        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(f"{cfg['base_path']}/{site}")
            all_files = ftp.nlst()

            # --- Cari file T dan P di FTP ---
            t_files = [f for f in all_files if re.search(fr"T\d+[A-Z].*{timestamp}.*\.[xX]$", f)] if timestamp else []
            p_files = [f for f in all_files if re.search(fr"P\d+[A-Z].*{timestamp}.*\.[xX]$", f)] if timestamp else []

            # === CASE 1: Jika ada T/P file ===
            if t_files and p_files:
                t_buf, p_buf = io.BytesIO(), io.BytesIO()
                ftp.retrbinary(f"RETR {t_files[0]}", t_buf.write)
                ftp.retrbinary(f"RETR {p_files[0]}", p_buf.write)
                t_txt = t_buf.getvalue().decode(errors="ignore").strip()
                p_txt = p_buf.getvalue().decode(errors="ignore").strip()

                combined_txt = f"{t_txt}\n{p_txt}\nNNNN\n"
                return Response(
                    combined_txt,
                    mimetype="text/plain",
                    headers={"Content-Disposition": f"attachment; filename={timestamp}_WMO_TP.txt"}
                )

        # === CASE 2: fallback → generate dari BUFR ===
        local_path = download_from_ftp(site, filename)
        decoded = decode_bufr(local_path)
        df_meta, df_levels = parse_bufr(decoded, site=site)

        block = int(df_meta.iloc[0].get("wmo_block", 99))
        station = int(df_meta.iloc[0].get("wmo_station", 999))
        d = int(df_meta.iloc[0].get("day", 1))
        h = int(df_meta.iloc[0].get("hour", 0))
        timecode = f"{d:02d}{h:02d}00"

        # --- Ambil kode WMO (ID51 + callsign) dari nama file ---
        # contoh: A_IUKG51WITT090000_C_WIIX_20251009000000.bfr
        id_match = re.search(r"A_([A-Z0-9]{4,6})W([A-Z]{3,4})", filename)
        if id_match:
            base_code = id_match.group(1)  # contoh: IUKG51
            callsign = "W" + id_match.group(2)  # contoh: WITT
        else:
            base_code, callsign = "ID51", site.upper()

        # --- TEMP (TTAA–TTDD) ---
        wmo_temp_text = generate_wmo_temp(df_meta, df_levels)
        wmo_temp_lines = wmo_temp_text.strip().splitlines()

        # --- PILOT (PPAA–PPDD) ---
        def generate_wmo_pilot(df_meta, df_levels):
            if df_meta.empty or df_levels.empty:
                return ""
            block = int(df_meta.iloc[0].get("wmo_block", 99))
            station = int(df_meta.iloc[0].get("wmo_station", 999))
            d = df_meta.iloc[0].get("day", 1)
            h = df_meta.iloc[0].get("hour", 0)
            df_levels = df_levels.sort_values("pressure_hPa", ascending=False)

            def encode_wind_group(wd, ws):
                if pd.isna(wd) or pd.isna(ws): return "/////"
                dd = int(round(wd / 10)) % 36
                ff = int(round(ws)) % 1000
                return f"{dd:02d}{ff:03d}"[:5]

            def pack(groups):
                return [" ".join(groups[i:i+5]) for i in range(0, len(groups), 5)]

            lines = []
            for code, cond in zip(
                ["PPAA", "PPBB", "PPCC", "PPDD"],
                [lambda p:p>=100, lambda p:100<p<1000, lambda p:p<100, lambda p:True]
            ):
                sel = df_levels[df_levels["pressure_hPa"].apply(cond)]
                groups = [encode_wind_group(r.get("wind_dir_deg"), r.get("wind_speed_mps")) for _, r in sel.iterrows()]
                lines.append(f"{code} {block:02d}{station:03d} {d:02d}{h:02d}00")
                lines += pack(groups)
            return "\n".join(lines)

        wmo_pilot_text = generate_wmo_pilot(df_meta, df_levels)
        wmo_pilot_lines = wmo_pilot_text.strip().splitlines()

        # --- Bagi per section ---
        sections = {k: [] for k in ["TTAA","TTBB","TTCC","TTDD","PPAA","PPBB","PPCC","PPDD"]}
        current = None
        for line in wmo_temp_lines + wmo_pilot_lines:
            for key in sections:
                if line.startswith(key):
                    current = key
                    sections[key].append(line)
                    break
            else:
                if current:
                    sections[current].append(line)

        # --- Prefix Meteomodem (US, UK, UL, UE, UP, UG, UH, UQ) ---
        prefix_map = {
            "TTAA": "US", "TTBB": "UK", "TTCC": "UL", "TTDD": "UE",
            "PPAA": "UP", "PPBB": "UG", "PPCC": "UH", "PPDD": "UQ"
        }

        # --- Bangun format akhir ---
        lines_out = []
        for key, prefix in prefix_map.items():
            if not sections[key]:
                continue
            lines_out.append(f"{prefix}{base_code} {callsign} {timecode}")
            lines_out += sections[key]
            lines_out.append("=")

        lines_out.append("NNNN")
        final_txt = "\n".join(lines_out) + "\n"

        return Response(
            final_txt,
            mimetype="text/plain",
            headers={"Content-Disposition": f"attachment; filename={filename}_WMO_AUTO.txt"}
        )

    except Exception as e:
        print("❌ Error generating WMO:", e)
        return f"Error generating WMO: {e}", 500

@app.route("/raob/<site>/<filename>")
@login_required
def raob_analysis(site, filename):
    """
    Analisis RAOB lengkap dengan caching SQLite.
    - Ambil data dari DB kalau sudah ada.
    - Jika belum, unduh dari FTP, decode, parse, dan simpan ke DB.
    - Hasil: Skew-T, Hodograph, Indeks, dan Analisis cuaca.
    - Metadata dikonversi ke human-readable (sensor, gas, balloon, termination, dsb.)
    """
    try:
        # ==========================================================
        # 1️⃣ Tentukan tipe file (bufr/bfr/bfh/bin)
        # ==========================================================
        ftype = filename.split(".")[-1].lower()
        if ftype not in ["bufr", "bfr", "bfh", "bin"]:
            ftype = "bufr"

        # ==========================================================
        # 2️⃣ Coba ambil dari DATABASE dulu
        # ==========================================================
        cached = db_get(ftype, site, filename)
        if cached:
            print(f"[DB] ✅ RAOB cache hit for {filename}")
            df_meta, df_levels = cached
        else:
            # ======================================================
            # 3️⃣ Jika belum ada di DB → ambil dari FTP & decode
            # ======================================================
            local_path = download_from_ftp(site, filename)
            decoded = decode_bufr(local_path)
            df_meta, df_levels = parse_bufr(decoded, site=site)

            # ======================================================
            # 4️⃣ Simpan hasil parse ke DB
            # ======================================================
            db_insert(
                ftype,
                site,
                filename,
                extract_date_from_filename(filename),
                df_meta,
                df_levels
            )
            print(f"[DB] 💾 Cached {filename} into {ftype} table")

        # ==========================================================
        # 5️⃣ Pastikan ada data level untuk analisis
        # ==========================================================
        if df_levels.empty:
            return "No levels found", 500

        # --- Metadata dasar ---
        meta = df_meta.to_dict("records")[0] if not df_meta.empty else {}

        # --- Konversi launch_time ke string UTC ---
        launch_time = meta.get("launch_time", "-")
        if isinstance(launch_time, pd.Timestamp):
            launch_time = launch_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        elif isinstance(launch_time, datetime):
            launch_time = launch_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        elif not isinstance(launch_time, str):
            launch_time = str(launch_time)

        # ==========================================================
        # 🔹 Human-readable mappings
        # ==========================================================
        # Termination reason
        if "reason_for_termination" in meta and meta["reason_for_termination"] not in (None, ""):
            try:
                code = int(meta["reason_for_termination"])
                meta["reason_for_termination"] = f"{code} – {REASON_MAP.get(code, 'Unknown')}"
            except Exception:
                pass

        # Sensor & balloon types
        alias_groups = {
            "pressure": ["pressure_sensor_type", "type_of_pressure_sensor"],
            "temperature": ["temperature_sensor_type", "type_of_temperature_sensor"],
            "humidity": ["humidity_sensor_type", "type_of_humidity_sensor"],
            "balloon": ["balloon_type", "type_of_balloon"],
            "balloon_gas": ["balloon_gas_type", "type_of_gas_used_in_balloon"],
            "balloon_manufacturer": ["balloon_manufacturer"],
        }
        for group, keys in alias_groups.items():
            mapping = SENSOR_MAPS.get(group, {})
            for k in keys:
                if k in meta and meta[k] not in (None, ""):
                    try:
                        code = int(meta[k])
                        meta[k] = f"{code} – {mapping.get(code, 'Unknown')}"
                    except Exception:
                        pass

        # Derived max height & min pressure
        if "height_m" in df_levels:
            max_h = df_levels["height_m"].dropna().max()
            if pd.notna(max_h):
                meta["max_height"] = f"{max_h:.0f} m"
        if "pressure_hPa" in df_levels:
            min_p = df_levels["pressure_hPa"].dropna().min()
            if pd.notna(min_p):
                meta["end_pressure"] = f"{min_p:.1f} hPa"

        # ==========================================================
        # 🔹 Data preparation
        # ==========================================================
        df = df_levels.dropna(subset=["pressure_hPa"]).copy()
        df = df.sort_values("pressure_hPa", ascending=False)
        df["pressure_hPa"] = medfilt(df["pressure_hPa"].values, kernel_size=3)

        if "temp_C" in df:
            df["temp_C"] = pd.Series(df["temp_C"]).interpolate(limit_direction="both")
            df["temp_C"] = medfilt(df["temp_C"].values, kernel_size=3)
        if "dewpoint_C" in df:
            df["dewpoint_C"] = pd.Series(df["dewpoint_C"]).interpolate(limit_direction="both")
            df["dewpoint_C"] = medfilt(df["dewpoint_C"].values, kernel_size=3)

        df = df.drop_duplicates(subset=["pressure_hPa"]).sort_values("pressure_hPa", ascending=False).reset_index(drop=True)

        # --- Thermodynamic profile ---
        thermo = df.dropna(subset=["pressure_hPa", "temp_C", "dewpoint_C"]).copy()
        if thermo.empty:
            return "Insufficient thermo data", 500

        p = thermo["pressure_hPa"].values * units.hPa
        T = thermo["temp_C"].values * units.degC
        Td = thermo["dewpoint_C"].values * units.degC

        # --- Wind profile ---
        wind = df.dropna(subset=["pressure_hPa", "wind_dir_deg", "wind_speed_mps"]).copy()
        if wind.empty:
            u = v = p_w = hgt = None
        else:
            p_w = wind["pressure_hPa"].values * units.hPa
            ws = wind["wind_speed_mps"].values * (units.meter / units.second)
            wdir = wind["wind_dir_deg"].values * units.degree
            u, v = mpcalc.wind_components(ws, wdir)
            if "height_m" in wind:
                hgt = wind["height_m"].values * units.meter
            elif "height_m" in df:
                hgt = df["height_m"].interpolate(limit_direction="both").values * units.meter
            else:
                hgt = mpcalc.pressure_to_height_std(p_w)

        # ==========================================================
        # 🔹 Thermodynamic indices
        # ==========================================================
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                lcl_pressure, lcl_temp = mpcalc.lcl(p[0], T[0], Td[0])
                parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to("degC")
                cape, cin = mpcalc.cape_cin(p, T, Td, parcel_prof)
                li = mpcalc.lifted_index(p, T, parcel_prof)
                ki = mpcalc.k_index(p, T, Td)
            except Exception as e:
                print("Thermo calc failed:", e)
                cape = cin = li = ki = np.nan * units("dimensionless")

        # --- Wind shear ---
        shear_0_1km_mag = shear_0_6km_mag = np.nan * units("knot")
        if (u is not None) and (v is not None):
            try:
                if len(p_w) >= 2:
                    sh_u_1km, sh_v_1km = mpcalc.bulk_shear(p_w, u, v, depth=1000 * units.meter)
                    sh_u_6km, sh_v_6km = mpcalc.bulk_shear(p_w, u, v, depth=6000 * units.meter)
                    shear_0_1km_mag = mpcalc.wind_speed(sh_u_1km, sh_v_1km).to("knot")
                    shear_0_6km_mag = mpcalc.wind_speed(sh_u_6km, sh_v_6km).to("knot")
            except Exception as e:
                print("Shear calc failed:", e)

        # --- SRH ---
        srh_0_1km = srh_0_3km = np.nan * units("m^2/s^2")
        if (u is not None) and (v is not None) and (hgt is not None):
            try:
                sort_idx = np.argsort(hgt)
                hgt_sorted, u_sorted, v_sorted = hgt[sort_idx], u[sort_idx], v[sort_idx]
                if hgt_sorted[-1] > 1000 * units.m:
                    rmotion, _, _ = mpcalc.bunkers_storm_motion(p_w, u_sorted, v_sorted, hgt_sorted)
                    storm_u, storm_v = rmotion
                    srh_0_1km, _, _ = mpcalc.storm_relative_helicity(
                        hgt_sorted, u_sorted, v_sorted,
                        depth=1000 * units.m,
                        bottom=0 * units.m,
                        storm_u=storm_u, storm_v=storm_v)
                if hgt_sorted[-1] > 3000 * units.m:
                    srh_0_3km, _, _ = mpcalc.storm_relative_helicity(
                        hgt_sorted, u_sorted, v_sorted,
                        depth=3000 * units.m,
                        bottom=0 * units.m,
                        storm_u=storm_u, storm_v=storm_v)
            except Exception as e:
                print("SRH calculation failed:", e)

        # --- Freezing level ---
        freezing_level = "-"
        try:
            idx = np.where(np.diff(np.sign(T.m)))[0]
            if idx.size > 0:
                freezing_level = f"{thermo.iloc[idx[0]]['pressure_hPa']:.0f} hPa"
        except Exception:
            pass

        # --- Tropopause detection (Lapse-rate + Cold-point) ---
        tropopause_LRT = "-"
        tropopause_CPT = "-"
        try:
            if {"height_m", "temp_C", "pressure_hPa"} <= set(thermo.columns):
                T_vals = thermo["temp_C"].to_numpy()
                Z_vals = thermo["height_m"].to_numpy()
                P_vals = thermo["pressure_hPa"].to_numpy()
        
                # --- Cold Point Tropopause (minimum T) ---
                i_min = np.argmin(T_vals)
                Tmin = T_vals[i_min]
                Zmin = Z_vals[i_min]
                Pmin = P_vals[i_min]
                tropopause_CPT = f"{Pmin:.0f} hPa ({Zmin/1000:.1f} km, Tmin = {Tmin:.1f} °C)"
        
                # --- Lapse Rate Tropopause (WMO definition) ---
                mask = (P_vals < 400) & (P_vals > 30)
                if np.count_nonzero(mask) > 5:
                    Tm, Zm, Pm = T_vals[mask], Z_vals[mask], P_vals[mask]
                    lapse = -np.gradient(Tm, Zm) * 1000  # °C/km
                    for i in range(len(Zm) - 1):
                        if lapse[i] <= 2.0:
                            z_top = Zm[i] + 2000
                            m2 = (Zm >= Zm[i]) & (Zm <= z_top)
                            if np.mean(lapse[m2]) <= 2.0:
                                tropopause_LRT = f"{Pm[i]:.0f} hPa ({Zm[i]/1000:.1f} km)"
                                break
        except Exception as e:
            print("Tropopause calc failed:", e)
        

        # ==========================================================
        # 🔹 Generate plots (Skew-T + Hodograph)
        # ==========================================================
        fig1 = plt.figure(figsize=(7, 7))
        skew = SkewT(fig1, rotation=45)
        skew.ax.set_facecolor("#fff9ef")
        skew.ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
        skew.plot(p, T, color="red", linewidth=2, label="Temperature")
        skew.plot(p, Td, color="blue", linewidth=2, label="Dew Point")
        skew.plot(p, parcel_prof, color="black", linestyle="--", label="Parcel Path")
        if (u is not None) and (v is not None):
            skew.plot_barbs(p_w, u.to("m/s"), v.to("m/s"), xloc=1.05)
        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-40, 40)
        skew.ax.legend(fontsize=8, loc="best")
        skew.ax.set_title(f"{site.upper()}  {launch_time}",
                          fontsize=10, fontweight="bold", color="#222")
        buf1 = BytesIO()
        plt.savefig(buf1, format="png", bbox_inches="tight")
        buf1.seek(0)
        skewt_img = base64.b64encode(buf1.read()).decode("utf-8")
        plt.close(fig1)

        if (u is not None) and (v is not None):
            fig2, ax = plt.subplots(figsize=(6, 6))
            hodo = Hodograph(ax, component_range=60.0)
            hodo.add_grid(increment=10)
            u_rot, v_rot = -u.to("m/s"), -v.to("m/s")
            hodo.plot(u_rot, v_rot, color="#007bff", linewidth=2, label="Wind profile")
            if hgt is not None:
                mask = ~np.isnan(hgt.m)
                ax.scatter(u_rot[mask].to("m/s"), v_rot[mask].to("m/s"),
                           c=hgt[mask].m / 1000.0, cmap="viridis",
                           s=30, edgecolors="black", linewidths=0.3,
                           label="Height (km)")
            ax.set_facecolor("#f9f9f9")
            ax.grid(True, linestyle="--", color="gray", alpha=0.5)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("U wind (m/s)")
            ax.set_ylabel("V wind (m/s)")
            ax.legend(fontsize=8, loc="upper left")
            buf2 = BytesIO()
            plt.savefig(buf2, format="png", bbox_inches="tight")
            buf2.seek(0)
            hodo_img = base64.b64encode(buf2.read()).decode("utf-8")
            plt.close(fig2)
        else:
            hodo_img = None

        # ==========================================================
        # 🔹 Build indices + weather analysis
        # ==========================================================
        def scalar_str(x, fmt=".1f"):
            try:
                val = np.atleast_1d(x.m)[0]
                return format(val, fmt) if np.isfinite(val) else "-"
            except Exception:
                return "-"

        indices = {
            "LCL Pressure (hPa)": scalar_str(lcl_pressure),
            "CAPE (J/kg)": scalar_str(cape),
            "CIN (J/kg)": scalar_str(cin),
            "Lifted Index (°C)": scalar_str(li),
            "K Index (°C)": scalar_str(ki),
            "0–1 km Bulk Shear (kt)": scalar_str(shear_0_1km_mag),
            "0–6 km Bulk Shear (kt)": scalar_str(shear_0_6km_mag),
            "SRH 0–1 km (m²/s²)": scalar_str(srh_0_1km),
            "SRH 0–3 km (m²/s²)": scalar_str(srh_0_3km),
            "Freezing Level": freezing_level,
            "Tropopause LRT": tropopause_LRT,
            "Tropopause CPT": tropopause_CPT,
        }

        analysis_text = generate_weather_analysis(df)

        # ==========================================================
        # 🔹 Render template
        # ==========================================================
        return render_template(
            "raob.html",
            site=site,
            date_str=launch_time,
            meta=meta,
            indices=indices,
            skewt_img=skewt_img,
            hodo_img=hodo_img,
            analysis_text=analysis_text
        )

    except Exception as e:
        print(f"[ERROR] RAOB analysis failed for {filename}: {e}")
        return f"RAOB error: {e}", 500

# ==============================
# 🔍 AUTO WEATHER ANALYSIS LOGIC
# ==============================
def generate_weather_analysis(df):
    text = []

    if "rh_percent" in df:
        clouds = df[df["rh_percent"] > 90]
        if not clouds.empty:
            base = clouds["height_m"].min() / 1000
            top = clouds["height_m"].max() / 1000
            text.append(f"☁️ Cloud layer detected between {base:.1f}–{top:.1f} km (RH > 90%).")

    if "temp_C" in df:
        cross = df[df["temp_C"].diff().abs() > 0]
        if not cross.empty:
            freezing = df.loc[(df["temp_C"].shift() >= 0) & (df["temp_C"] <= 0)]
            if not freezing.empty:
                zf = freezing["height_m"].iloc[0] / 1000
                text.append(f"❄️ Freezing level around {zf:.1f} km.")

    if {"temp_C", "height_m"} <= set(df.columns):
        df["lapse_rate"] = -df["temp_C"].diff() / (df["height_m"].diff() / 1000)
        unstable = df[df["lapse_rate"] > 7]
        if not unstable.empty:
            zmin, zmax = unstable["height_m"].min()/1000, unstable["height_m"].max()/1000
            text.append(f"⚠️ Unstable layer (lapse rate > 7°C/km) from {zmin:.1f}–{zmax:.1f} km.")

    if {"wind_speed_mps", "height_m"} <= set(df.columns):
        df["shear_rate"] = df["wind_speed_mps"].diff() / (df["height_m"].diff()/1000)
        strong = df[df["shear_rate"].abs() > 10]
        if not strong.empty:
            zmin, zmax = strong["height_m"].min()/1000, strong["height_m"].max()/1000
            text.append(f"💨 Strong wind shear zone ({zmin:.1f}–{zmax:.1f} km). Turbulence possible.")

    if not text:
        text.append("✅ Atmosphere mostly stable, no significant weather hazards detected.")

    return "<br>".join(text)

@app.route("/main")
@login_required
@page_access_required("main")
def main_page():
    return render_template("main.html")

@app.route("/map")
@login_required
@page_access_required("map")
def map_view():
    t = request.args.get("t", str(int(time.time())))
    store = get_user_store()
    total = len(store.get("levels", []))
    user = session.get("user")
    return render_template("map.html", total=total, user=user, t=t)

@app.route("/underdev")
@login_required
def under_development():
    return render_template("under_development.html")

@app.route("/time_accuracy")
@login_required
@page_access_required("time_accuracy")
def time_accuracy():
    """
    Halaman grafik time accuracy (Launch→100hPa dan Launch→30hPa)
    """
    return render_template("time_accuracy.html")
    
@app.route("/api/time_accuracy/<site>")
@login_required
def api_time_accuracy(site):
    """
    Ambil data time accuracy (.bfr) untuk site tertentu berdasarkan bulan & tahun.
    🔹 Query param opsional: ?year=YYYY&month=MM
    🔹 Default: bulan berjalan
    """
    from datetime import datetime, timedelta, timezone
    import pandas as pd, re

    # ==========================================================
    # 🔹 Tangkap parameter dari query string (debug-friendly)
    # ==========================================================
    try:
        year_str = request.args.get("year")
        month_str = request.args.get("month")

        if year_str and month_str:
            year = int(year_str)
            month = int(month_str)
            print(f"🔹 Query params received: year={year}, month={month}")
        else:
            now = datetime.utcnow()
            year, month = now.year, now.month
            print(f"⚠️ No query params found — defaulting to {year}-{month:02d}")

    except Exception as e:
        now = datetime.utcnow()
        year, month = now.year, now.month
        print(f"⚠️ Error parsing query params: {e} → defaulting to {year}-{month:02d}")

    # ==========================================================
    # 🔹 Buat rentang tanggal UTC (bulan tersebut)
    # ==========================================================
    start_date = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    if month == 12:
        end_date = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    else:
        end_date = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    print(f"📅 [TimeAccuracy] Fetching {site} for {year}-{month:02d} ({start_date.date()} → {end_date.date()})")

    # ==========================================================
    # 🔹 Helper ambil datetime dari nama file
    # ==========================================================
    def extract_datetime_from_filename(fname: str):
        """Ambil datetime dari nama file (contoh: A202510050000.BFR -> 2025-10-05 00:00:00 UTC)."""
        match = re.search(r"(\d{10,14})", fname)
        if not match:
            return None
        dt_str = match.group(1)
        for fmt in ["%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d%H"]:
            try:
                return datetime.strptime(dt_str, fmt)
            except Exception:
                continue
        return None

    data = []
    try:
        # ==========================================================
        # 🔹 Ambil semua file .bfr dari FTP (via cache fetch_all_sites)
        # ==========================================================
        all_sites = fetch_all_sites(
            ext_filter=[".bfr"],
            with_meta=False,
            start_date=start_date,
            end_date=end_date
        )

        # Case-insensitive site match
        site_keys = {s.lower(): s for s in all_sites.keys()}
        site_key = site.lower()
        if site_key not in site_keys:
            print(f"⚠️ Site {site} not found in FTP list")
            return jsonify({"site": site, "data": [], "year": year, "month": month})

        true_site = site_keys[site_key]

        # ==========================================================
        # 🔁 Loop semua file .bfr di bulan & tahun terpilih
        # ==========================================================
        for f in all_sites[true_site]:
            fname = f["name"]
            try:
                file_dt = extract_datetime_from_filename(fname)
                if not file_dt:
                    continue

                # Lewati jika file di luar range bulan yang diminta
                if not (start_date <= file_dt.replace(tzinfo=timezone.utc) < end_date):
                    continue

                date_str = file_dt.strftime("%Y-%m-%d")
                hour_label = f"{file_dt.hour:02d}Z"
                if hour_label not in ["00Z", "12Z"]:
                    hour_label = "00Z" if file_dt.hour < 6 else "12Z"

                # ======================================================
                # 🔹 Ambil data dari cache atau decode baru
                # ======================================================
                ftype = fname.split(".")[-1].lower()
                if ftype not in ["bufr", "bfr", "bfh", "bin"]:
                    ftype = "bfr"

                cached = db_get(ftype, true_site, fname)
                if cached:
                    df_meta, df_levels = cached
                else:
                    local_path = download_from_ftp(true_site, fname)
                    decoded = decode_bufr(local_path)
                    df_meta, df_levels = parse_bufr(decoded, site=site)
                    db_insert(ftype, true_site, fname,
                              extract_date_from_filename(fname),
                              df_meta, df_levels)

                if df_levels.empty:
                    continue

                # ======================================================
                # 🔹 Ambil launch_time (real atau fallback dari filename)
                # ======================================================
                launch_time = None
                if not df_meta.empty:
                    for key in ["launch_time", "launch_datetime", "time_of_launch", "launch_time_UTC"]:
                        if key in df_meta.columns:
                            val = df_meta.iloc[0][key]
                            if pd.notna(val):
                                try:
                                    launch_time = pd.to_datetime(val)
                                    break
                                except Exception:
                                    pass

                if launch_time is None:
                    launch_time = file_dt  # fallback

                # ======================================================
                # 🔹 Hitung waktu ke 100 hPa dan burst (≈ 30 hPa)
                # ======================================================
                t100 = df_levels.loc[df_levels["pressure_hPa"] <= 100, "time_s"].min()
                t30 = df_levels.loc[df_levels["pressure_hPa"] <= 0, "time_s"].min()
                if pd.isna(t30):
                    t30 = df_levels["time_s"].max()

                if pd.notna(t100):
                    data.append({
                        "filename": fname,
                        "date": date_str,
                        "hour": hour_label,
                        "launch_time": launch_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "AB": round(t100 / 60.0, 1),
                        "CD": round(t30 / 60.0, 1) if pd.notna(t30) else None
                    })

            except Exception as e:
                print(f"⚠️ Error parsing {fname}: {e}")

    except Exception as e:
        print(f"❌ Time accuracy fetch failed for {site}: {e}")

    # ==========================================================
    # 🔹 Urutkan hasil: tanggal + jam (00Z dulu, 12Z setelahnya)
    # ==========================================================
    def sort_key(x):
        return (x["date"], 0 if x["hour"] == "00Z" else 1)

    data = sorted(data, key=sort_key)
    print(f"✅ Found {len(data)} records for {site} ({year}-{month:02d})")

    return jsonify({"site": site, "data": data, "year": year, "month": month})

@app.route("/height_reach")
@login_required
@page_access_required("height_reach")
def height_reach():
    return render_template("height_reach.html")

@app.route("/api/height_reach/<site>")
@login_required
def api_height_reach(site):
    """
    Ambil data ketinggian maksimum & tekanan minimum balon (.bfr) per hari
    untuk site tertentu berdasarkan bulan & tahun.
    🔹 Query param opsional: ?year=YYYY&month=MM
    🔹 Default: bulan berjalan (UTC)
    🔹 Data diambil dari SQLite cache (jika tidak ada, unduh & decode dari FTP)
    """
    from datetime import datetime, timedelta, timezone
    import pandas as pd, re

    # ==========================================================
    # 🔹 Tangkap parameter dari query string
    # ==========================================================
    try:
        year_str = request.args.get("year")
        month_str = request.args.get("month")

        if year_str and month_str:
            year = int(year_str)
            month = int(month_str)
            print(f"🔹 Query params received: year={year}, month={month}")
        else:
            now = datetime.utcnow()
            year, month = now.year, now.month
            print(f"⚠️ No query params found — defaulting to {year}-{month:02d}")
    except Exception as e:
        now = datetime.utcnow()
        year, month = now.year, now.month
        print(f"⚠️ Error parsing query params: {e} → defaulting to {year}-{month:02d}")

    # ==========================================================
    # 🔹 Rentang tanggal UTC bulan tersebut
    # ==========================================================
    start_date = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    if month == 12:
        end_date = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    else:
        end_date = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    print(f"📅 [HeightReach] Fetching {site} for {year}-{month:02d} ({start_date.date()} → {end_date.date()})")

    # ==========================================================
    # 🔹 Helper ekstrak datetime dari nama file
    # ==========================================================
    def extract_datetime_from_filename(fname: str):
        """Ambil datetime dari nama file (contoh: T2097502A202510050000.BFR)."""
        m = re.search(r"(\d{10,14})", fname)
        if not m:
            return None
        s = m.group(1)
        for fmt in ["%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d%H"]:
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        return None

    data = []
    try:
        # ==========================================================
        # 🔹 Ambil daftar file .bfr dari cache/FTP
        # ==========================================================
        all_sites = fetch_all_sites(
            ext_filter=[".bfr"],
            with_meta=False,
            start_date=start_date,
            end_date=end_date
        )

        # Case-insensitive site match
        site_keys = {s.lower(): s for s in all_sites.keys()}
        site_key = site.lower()
        if site_key not in site_keys:
            print(f"⚠️ Site {site} not found in FTP list")
            return jsonify({"site": site, "data": [], "year": year, "month": month})
        true_site = site_keys[site_key]

        # ==========================================================
        # 🔁 Loop semua file dalam bulan terpilih
        # ==========================================================
        for f in all_sites[true_site]:
            fname = f["name"]
            try:
                file_dt = extract_datetime_from_filename(fname)
                if not file_dt:
                    continue

                # Lewati jika di luar bulan target
                if not (start_date <= file_dt.replace(tzinfo=timezone.utc) < end_date):
                    continue

                date_str = file_dt.strftime("%Y-%m-%d")
                hour_label = "00Z" if file_dt.hour < 6 else "12Z"

                # ======================================================
                # 🔹 Ambil data dari cache DB atau decode baru
                # ======================================================
                ftype = fname.split(".")[-1].lower()
                if ftype not in ["bufr", "bfr", "bfh", "bin"]:
                    ftype = "bfr"

                cached = db_get(ftype, true_site, fname)
                if cached:
                    _, df_levels = cached
                    print(f"[DB] ✅ cache hit for {fname}")
                else:
                    local_path = download_from_ftp(true_site, fname)
                    decoded = decode_bufr(local_path)
                    df_meta, df_levels = parse_bufr(decoded, site=site)
                    db_insert(
                        ftype,
                        true_site,
                        fname,
                        extract_date_from_filename(fname),
                        df_meta,
                        df_levels
                    )
                    print(f"[DB] 💾 cached {fname}")

                if df_levels.empty:
                    continue
                if "height_m" not in df_levels.columns or "pressure_hPa" not in df_levels.columns:
                    continue

                # ======================================================
                # 🔹 Ambil nilai maksimum & minimum
                # ======================================================
                max_height = df_levels["height_m"].max()
                min_pres = df_levels["pressure_hPa"].min()

                if pd.notna(max_height) and max_height > 0:
                    data.append({
                        "filename": fname,
                        "date": date_str,
                        "hour": hour_label,
                        "max_height": round(float(max_height), 0),
                        "end_pressure": round(float(min_pres), 1) if pd.notna(min_pres) else None
                    })

            except Exception as e:
                print(f"⚠️ Error parsing {fname}: {e}")

    except Exception as e:
        print(f"❌ Height reach fetch failed for {site}: {e}")

    # ==========================================================
    # 🔹 Urutkan hasil: tanggal + jam (00Z dulu, 12Z setelahnya)
    # ==========================================================
    def sort_key(x):
        return (x["date"], 0 if x["hour"] == "00Z" else 1)
    data = sorted(data, key=sort_key)

    print(f"✅ Found {len(data)} records for {site} ({year}-{month:02d})")
    return jsonify({"site": site, "data": data, "year": year, "month": month})

@app.route("/settings")
@login_required
@page_access_required("settings")
def settings_page():
    # Hanya admin
    if session.get("user") != "admin":
        return "Access denied. Admin only.", 403
    return render_template("settings.html", users=VALID_USERS)

@app.route("/api/users", methods=["GET", "POST", "PUT", "DELETE"])
@login_required
def manage_users():
    if session.get("user") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    # --- GET: return all users
    if request.method == "GET":
        return jsonify(VALID_USERS)

    # --- POST: add new user
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"error": "Missing fields"}), 400
        if username in VALID_USERS:
            return jsonify({"error": "User already exists"}), 400

        # 🔒 Hash password sebelum disimpan
        hashed_pw = generate_password_hash(password)
        
        # Default expiry & pages (contoh 1 year)
        expiry_default = (datetime.utcnow().date() + timedelta(days=365)).strftime("%Y-%m-%d")
        VALID_USERS[username] = {
            "password": hashed_pw,
            "expiry": expiry_default,
            "pages": ["main", "dashboard"]
        }
        save_users()

        print(f"✅ Added new user (hashed): {username}")
        return jsonify({"success": True, "users": VALID_USERS})

    # --- PUT: update existing user's password
    if request.method == "PUT":
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"error": "Missing fields"}), 400
        if username not in VALID_USERS:
            return jsonify({"error": "User not found"}), 404

        # 🔒 Re-hash password saat diubah
        hashed_pw = generate_password_hash(password)

        # Kompatibel dengan struktur lama & baru
        if isinstance(VALID_USERS[username], dict):
            VALID_USERS[username]["password"] = hashed_pw
        else:
            VALID_USERS[username] = {"password": hashed_pw}

        save_users()
        print(f"🔑 Updated password (hashed) for user: {username}")
        return jsonify({"success": True, "users": VALID_USERS})

    # --- DELETE: remove user
    if request.method == "DELETE":
        data = request.get_json()
        username = data.get("username")
        if not username:
            return jsonify({"error": "Missing username"}), 400
        if username not in VALID_USERS:
            return jsonify({"error": "User not found"}), 404

        VALID_USERS.pop(username, None)
        save_users()
        print(f"🗑️ Deleted user: {username}")
        return jsonify({"success": True, "users": VALID_USERS})

@app.route("/api/sites_config", methods=["GET", "POST", "DELETE"])
@login_required
def manage_sites():
    # --- Semua user boleh GET ---
    if request.method == "GET":
        return jsonify({"sites": SITE_LIST})

    # --- Hanya admin boleh ubah ---
    if session.get("user") != "admin":
        return jsonify({"error": "Unauthorized (admin only)"}), 403

    data = request.get_json(force=True)
    name = data.get("name", "").strip().lower()
    if not name:
        return jsonify({"error": "Missing site name"}), 400

    if request.method == "POST":
        if name in SITE_LIST:
            return jsonify({"error": "Site already exists"}), 400
        SITE_LIST.append(name)
        save_sites()
        return jsonify({"success": True, "sites": SITE_LIST})

    if request.method == "DELETE":
        if name not in SITE_LIST:
            return jsonify({"error": "Site not found"}), 404
        SITE_LIST.remove(name)
        save_sites()
        return jsonify({"success": True, "sites": SITE_LIST})

@app.route("/raob_doc")
@login_required
@page_access_required("raob_doc")
def raob_doc():
    return render_template("raob_doc.html")

@app.route("/api/trajectory3d")
@login_required
def api_trajectory3d():
    """
    Keluarkan path radiosonde untuk visualisasi 3D:
    - path: [[lon, lat, alt_m], ...] urut waktu
    - timestamps: [detik sejak launch], untuk animasi
    - station: lon/lat stasiun
    - meta: info waktu, site, dsb (jika ada)
    """
    store = get_user_store()
    levels = store.get("levels", [])
    meta = store.get("metadata", {})

    if not levels:
        return jsonify({"error": "No radiosonde loaded"}), 404

    # Ambil kolom yang ada saja (aman terhadap NaN)
    lons, lats, hgts, ts = [], [], [], []
    for row in levels:
        lon = row.get("longitude")
        lat = row.get("latitude")
        h  = row.get("height_m")
        t  = row.get("time_s")
        if lon is None or lat is None or h is None or t is None:
            continue
        lons.append(float(lon))
        lats.append(float(lat))
        hgts.append(float(h))
        ts.append(float(t))

    if not lons:
        return jsonify({"error": "No lon/lat/height/time data"}), 400

    # Susun data untuk TripsLayer (1 trip)
    path = [[lo, la, hi] for lo, la, hi in zip(lons, lats, hgts)]
    timestamps = ts  # detik sejak launch
    station = {
        "lon": float(meta.get("station_lon")) if meta.get("station_lon") is not None else float(lons[0]),
        "lat": float(meta.get("station_lat")) if meta.get("station_lat") is not None else float(lats[0]),
        "name": f"{int(meta.get('wmo_block', 0)):02d}{int(meta.get('wmo_station', 0)):03d}" if meta else ""
    }

    # Durasi total animasi (detik)
    total_t = float(max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0.0

    return jsonify({
        "trip": {"path": path, "timestamps": timestamps},
        "station": station,
        "meta": {
            "launch_time": meta.get("launch_time", "-"),
            "max_height_m": max(hgts),
            "duration_s": total_t
        }
    })

@app.route("/trajectory3d")
@login_required
@page_access_required("trajectory3d")
def trajectory3d_page():
    return render_template("trajectory3d.html")

@app.route("/upload_bufr", methods=["POST"])
@login_required
def upload_bufr():
    """
    Upload file radiosonde (.bufr, .bfr, .bfh, .bin) dari web
    atau ambil dari FTP berdasarkan site + nama file.
    Semua file akan disimpan ke uploads/ lalu didecode seperti biasa.
    """
    try:
        uploads_dir = app.config.get("UPLOAD_FOLDER", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        # === MODE 1: dari FTP ===
        remote_site = request.form.get("remote_site")
        remote_file = request.form.get("remote_file")

        if remote_site and remote_file:
            cfg = CONFIG.get("ftp", {})
            ftp_host = cfg.get("host")
            ftp_user = cfg.get("user")
            ftp_pass = cfg.get("password")
            base_path = cfg.get("base_path", "/")

            local_dir = os.path.join(uploads_dir, remote_site)
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, remote_file)

            # --- Download dari FTP ke uploads_dir ---
            with ftplib.FTP() as ftp:
                ftp.connect(ftp_host, cfg.get("port", 21))
                ftp.login(ftp_user, ftp_pass)
                ftp.cwd(f"{base_path}/{remote_site}")
                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {remote_file}", f.write)
            print(f"[FTP] ✅ File downloaded to {local_path}")

            # Lanjut ke proses decode biasa
            filepath = local_path

        # === MODE 2: Upload manual ===
        else:
            file = request.files.get("file")
            if not file or not file.filename:
                return jsonify({"success": False, "error": "No file uploaded"}), 400

            filepath = os.path.join(uploads_dir, file.filename)
            file.save(filepath)
            print(f"[UPLOAD] ✅ File saved to {filepath}")

        # --- Decode ---
        decoded = decode_bufr(filepath)
        df_meta, df_levels = parse_bufr(decoded)
        issues = analyze_flight(df_meta, df_levels)

        df_meta = df_meta.replace({np.nan: None})
        df_levels = df_levels.replace({np.nan: None})

        # --- Simpan hasil ke store user ---
        store = get_user_store()
        store["metadata"] = df_meta.to_dict("records")[0] if not df_meta.empty else {}
        store["metadata"]["flight_issues"] = issues
        store["levels"] = df_levels.to_dict("records") if not df_levels.empty else []

        # --- Bangun data trajectory agar 3D viewer langsung bisa baca ---
        if not df_levels.empty:
            lon_col = next((c for c in df_levels.columns if "lon" in c.lower()), None)
            lat_col = next((c for c in df_levels.columns if "lat" in c.lower()), None)
            hgt_col = next((c for c in df_levels.columns if "height" in c.lower() or "alt" in c.lower()), None)
            time_col = next((c for c in df_levels.columns if "time" in c.lower()), None)

            if lon_col and lat_col and hgt_col:
                path = df_levels[[lon_col, lat_col, hgt_col]].dropna().values.tolist()
                timestamps = (df_levels[time_col] - df_levels[time_col].iloc[0]).fillna(0).tolist() \
                    if time_col else list(range(len(path)))

                station = {
                    "lon": float(store["metadata"].get("station_lon", df_levels[lon_col].iloc[0])),
                    "lat": float(store["metadata"].get("station_lat", df_levels[lat_col].iloc[0])),
                    "name": store["metadata"].get("station_name", remote_site or "Unknown")
                }
                meta_info = {
                    "launch_time": store["metadata"].get("launch_time", "-"),
                    "max_height_m": float(df_levels[hgt_col].max()),
                }

                store["trajectory"] = {
                    "trip": {"path": path, "timestamps": timestamps},
                    "station": station,
                    "meta": meta_info
                }
                print(f"[UPLOAD_BUFR] ✅ Trajectory stored ({len(path)} points)")
            else:
                print("[UPLOAD_BUFR] ⚠️ df_levels tidak punya kolom lon/lat/height")

        return jsonify({"success": True})

    except Exception as e:
        print("[UPLOAD_BUFR] ❌ Exception:", e)
        return jsonify({"success": False, "error": str(e)})

@app.route("/data_availability")
@login_required
@page_access_required("data_availability")
def data_availability_page():
    return render_template("data_availability.html")

@app.route("/api/data_availability")
@login_required
def api_data_availability():
    """
    Scan ketersediaan data radiosonde per hari dalam 1 bulan.
    File dicek: .bfr, .bin, T*.X, P*.X
    Format:
      - bfr/bin biasa mengandung tanggal di nama
      - T/P file: TxxxxxxAYYYYmmddHHMM.X  (seperti T2096011A202510080000.X)
    Output:
      {
        "year": 2025,
        "month": 10,
        "sites": [
          {
            "site": "aceh",
            "days": {
              1: {"00": {"color": "red","tooltip":"..."}, "12": {...}},
              ...
            }
          }
        ]
      }
    """
    from datetime import datetime
    import calendar, re

    month = int(request.args.get("month", datetime.utcnow().month))
    year = int(request.args.get("year", datetime.utcnow().year))

    start_date = datetime(year, month, 1)
    end_day = calendar.monthrange(year, month)[1]

    sites = SITE_LIST
    results = []

    try:
        cfg = CONFIG["ftp"]
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(cfg["base_path"])

            for site in sites:
                ftp.cwd(f"{cfg['base_path']}/{site}")
                try:
                    files = ftp.nlst()
                except Exception:
                    files = []
                day_data = {}

                for day in range(1, end_day + 1):
                    date_prefix = f"{year}{month:02d}{day:02d}"
                    entry = {}
                    for hour in ["00", "12"]:
                        # cari semua file yang cocok dengan tanggal dan jam
                        matched = [f for f in files if re.search(fr"{date_prefix}{hour}", f)]

                        # --- deteksi setiap jenis file ---
                        has_bfr = any(f.lower().endswith(".bfr") for f in matched)
                        has_bin = any(f.lower().endswith(".bin") for f in matched)

                        # format TxxxxxxAYYYYmmddHHMM.X dan PxxxxxxAYYYYmmddHHMM.X
                        has_tx = any(re.search(r"T\d+[A-Z].*?(\d{10})(?:[A-Z]+)?\.[xX]$", f) for f in matched)
                        has_px = any(re.search(r"P\d+[A-Z].*?(\d{10})(?:[A-Z]+)?\.[xX]$", f) for f in matched)


                        available = {"bfr": has_bfr, "bin": has_bin, "T": has_tx, "P": has_px}
                        total = sum(available.values())

                        # --- status warna ---
                        if total == 0:
                            color = "red"
                            tooltip = "no data"
                        elif total < 4:
                            color = "yellow"
                            missing = [k for k, v in available.items() if not v]
                            tooltip = f"missing: {', '.join(missing)}"
                        else:
                            color = "green"
                            tooltip = "bfr, bin, T, P available"

                        entry[hour] = {"color": color, "tooltip": tooltip}

                    day_data[day] = entry

                results.append({"site": site, "days": day_data})
                ftp.cwd(cfg["base_path"])

    except Exception as e:
        print("❌ FTP error in data_availability:", e)
        return jsonify({"error": str(e)}), 500

    return jsonify({"year": year, "month": month, "sites": results})

@app.route("/api/bufrmap_full", methods=["GET","POST","PUT","DELETE"])
@login_required
def api_bufrmap_full():
    import json, os

    MAP_FILE = "bufr_mapping_full.json"

    # --- Default mapping dari seluruh field di parse_bufr ---
    default_mapping = {
        "meta": [
            {"original": "WMO BLOCK NUMBER", "variable": "wmo_block"},
            {"original": "WMO STATION NUMBER", "variable": "wmo_station"},
            {"original": "004001 YEAR", "variable": "year"},
            {"original": "004002 MONTH", "variable": "month"},
            {"original": "004003 DAY", "variable": "day"},
            {"original": "004004 HOUR", "variable": "hour"},
            {"original": "004005 MINUTE", "variable": "minute"},
            {"original": "004006 SECOND", "variable": "second"},
            {"original": "LATITUDE (HIGH ACCURACY)", "variable": "station_lat"},
            {"original": "LONGITUDE (HIGH ACCURACY)", "variable": "station_lon"},
            {"original": "HEIGHT OF STATION GROUND", "variable": "station_height_m"},
            {"original": "RADIOSONDE SERIAL NUMBER", "variable": "radiosonde_serial_number"},
            {"original": "RADIOSONDE ASCENSION NUMBER", "variable": "radiosonde_ascension_number"},
            {"original": "RADIOSONDE RELEASE NUMBER", "variable": "radiosonde_release_number"},
            {"original": "RADIOSONDE GROUND RECEIVING SYSTEM", "variable": "radiosonde_ground_rx_system"},
            {"original": "RADIOSONDE OPERATING FREQUENCY", "variable": "radiosonde_operating_frequency"},
            {"original": "BALLOON MANUFACTURER", "variable": "balloon_manufacturer"},
            {"original": "WEIGHT OF BALLOON", "variable": "balloon_weight_kg"},
            {"original": "TYPE OF GAS USED IN BALLOON", "variable": "balloon_gas_type"},
            {"original": "TYPE OF PRESSURE SENSOR", "variable": "pressure_sensor_type"},
            {"original": "TYPE OF TEMPERATURE SENSOR", "variable": "temperature_sensor_type"},
            {"original": "TYPE OF HUMIDITY SENSOR", "variable": "humidity_sensor_type"},
            {"original": "SOFTWARE IDENTIFICATION AND VERSION NUMBER", "variable": "software_version"},
            {"original": "REASON FOR TERMINATION", "variable": "reason_for_termination"},
            {"original": "TRACKING TECHNIQUE/STATUS OF SYSTEM USED", "variable": "system_status"}
        ],
        "level": [
            {"original": "PRESSURE", "variable": "pressure_hPa"},
            {"original": "GEOPOTENTIAL HEIGHT", "variable": "height_m"},
            {"original": "TEMPERATURE/AIR TEMPERATURE", "variable": "temp_C"},
            {"original": "DEW-POINT TEMPERATURE", "variable": "dewpoint_C"},
            {"original": "WIND DIRECTION", "variable": "wind_dir_deg"},
            {"original": "WIND SPEED", "variable": "wind_speed_mps"},
            {"original": "LATITUDE DISPLACEMENT", "variable": "lat_disp"},
            {"original": "LONGITUDE DISPLACEMENT", "variable": "lon_disp"},
            {"original": "LONG TIME PERIOD OR DISPLACEMENT", "variable": "time_s"},
            {"original": "EXTENDED VERTICAL SOUNDING SIGNIFICANCE", "variable": "status_flag"},
        ]
    }

    # --- Load file mapping jika ada ---
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE) as f:
            mapping = json.load(f)
    else:
        mapping = default_mapping
        with open(MAP_FILE, "w") as f:
            json.dump(mapping, f, indent=2)

    # === Handle Methods ===
    if request.method == "GET":
        return jsonify(mapping)

    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    section = data.get("type")
    if section not in ["meta", "level"]:
        return jsonify({"error": "Invalid mapping type"}), 400

    if request.method == "POST":
        mapping[section].append({"original": data["original"], "variable": data["variable"]})

    elif request.method == "PUT":
        for m in mapping[section]:
            if m["original"] == data["original"]:
                m["variable"] = data["variable"]
                break

    elif request.method == "DELETE":
        mapping[section] = [m for m in mapping[section] if m["original"] != data["original"]]

    with open(MAP_FILE, "w") as f:
        json.dump(mapping, f, indent=2)

    return jsonify({"success": True})

@app.route("/api/bufrmap/<site>", methods=["GET", "POST", "PUT", "DELETE"])
@login_required
def api_bufrmap_site(site):
    """
    CRUD API for BUFR mapping configuration per site.
    - GET    → Get mapping for a given site (fallback to default)
    - POST   → Add new mapping entry
    - PUT    → Edit existing mapping entry
    - DELETE → Remove a mapping entry
    """

    import os, json
    from flask import request, jsonify
    CONFIG_DIR = "config"
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # --- Normalized site key ---
    site_key = str(site).lower().strip().replace(" ", "_")
    site_file = os.path.join(CONFIG_DIR, f"bufr_mapping_{site_key}.json")
    default_file = "bufr_mapping_full.json"

    # --- Load default mapping ---
    default_mapping = {
        "meta": [
            {"original": "WMO BLOCK NUMBER", "variable": "wmo_block"},
            {"original": "WMO STATION NUMBER", "variable": "wmo_station"},
            {"original": "004001 YEAR", "variable": "year"},
            {"original": "004002 MONTH", "variable": "month"},
            {"original": "004003 DAY", "variable": "day"},
            {"original": "004004 HOUR", "variable": "hour"},
            {"original": "004005 MINUTE", "variable": "minute"},
            {"original": "004006 SECOND", "variable": "second"},
            {"original": "LATITUDE (HIGH ACCURACY)", "variable": "station_lat"},
            {"original": "LONGITUDE (HIGH ACCURACY)", "variable": "station_lon"},
            {"original": "HEIGHT OF STATION GROUND", "variable": "station_height_m"},
            {"original": "RADIOSONDE SERIAL NUMBER", "variable": "radiosonde_serial_number"},
            {"original": "RADIOSONDE ASCENSION NUMBER", "variable": "radiosonde_ascension_number"},
            {"original": "RADIOSONDE RELEASE NUMBER", "variable": "radiosonde_release_number"},
            {"original": "RADIOSONDE GROUND RECEIVING SYSTEM", "variable": "radiosonde_ground_rx_system"},
            {"original": "RADIOSONDE OPERATING FREQUENCY", "variable": "radiosonde_operating_frequency"},
            {"original": "BALLOON MANUFACTURER", "variable": "balloon_manufacturer"},
            {"original": "WEIGHT OF BALLOON", "variable": "balloon_weight_kg"},
            {"original": "TYPE OF GAS USED IN BALLOON", "variable": "balloon_gas_type"},
            {"original": "TYPE OF PRESSURE SENSOR", "variable": "pressure_sensor_type"},
            {"original": "TYPE OF TEMPERATURE SENSOR", "variable": "temperature_sensor_type"},
            {"original": "TYPE OF HUMIDITY SENSOR", "variable": "humidity_sensor_type"},
            {"original": "SOFTWARE IDENTIFICATION AND VERSION NUMBER", "variable": "software_version"},
            {"original": "REASON FOR TERMINATION", "variable": "reason_for_termination"},
            {"original": "TRACKING TECHNIQUE/STATUS OF SYSTEM USED", "variable": "system_status"}
        ],
        "level": [
            {"original": "PRESSURE", "variable": "pressure_hPa"},
            {"original": "GEOPOTENTIAL HEIGHT", "variable": "height_m"},
            {"original": "TEMPERATURE/AIR TEMPERATURE", "variable": "temp_C"},
            {"original": "DEW-POINT TEMPERATURE", "variable": "dewpoint_C"},
            {"original": "WIND DIRECTION", "variable": "wind_dir_deg"},
            {"original": "WIND SPEED", "variable": "wind_speed_mps"},
            {"original": "LATITUDE DISPLACEMENT", "variable": "lat_disp"},
            {"original": "LONGITUDE DISPLACEMENT", "variable": "lon_disp"},
            {"original": "LONG TIME PERIOD OR DISPLACEMENT", "variable": "time_s"},
            {"original": "EXTENDED VERTICAL SOUNDING SIGNIFICANCE", "variable": "status_flag"}
        ]
    }

    # --- Helper: Load mapping from JSON or fallback ---
    def load_mapping():
        if os.path.exists(site_file):
            try:
                with open(site_file, "r") as f:
                    data = json.load(f)
                    if "meta" in data and "level" in data:
                        return data
            except Exception:
                pass
        # fallback to global
        if os.path.exists(default_file):
            try:
                with open(default_file, "r") as f:
                    data = json.load(f)
                    return data
            except Exception:
                pass
        return default_mapping.copy()

    # --- Helper: Save mapping safely ---
    def save_mapping(data):
        try:
            with open(site_file, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print("❌ Error saving mapping:", e)
            return False

    mapping = load_mapping()

    # =========================================================
    # GET — Return mapping JSON
    # =========================================================
    if request.method == "GET":
        return jsonify(mapping)

    # =========================================================
    # POST — Add new mapping entry
    # =========================================================
    elif request.method == "POST":
        js = request.get_json(force=True)
        section = js.get("type")  # "meta" or "level"
        original = js.get("original", "").strip()
        variable = js.get("variable", "").strip()

        if section not in ("meta", "level"):
            return jsonify({"error": "Invalid mapping type"}), 400
        if not original or not variable:
            return jsonify({"error": "Missing field(s)"}), 400

        # prevent duplicates
        if any(m["original"] == original for m in mapping[section]):
            return jsonify({"error": f"Field '{original}' already exists"}), 400

        mapping[section].append({"original": original, "variable": variable})
        if save_mapping(mapping):
            return jsonify({"success": True})
        return jsonify({"error": "Failed to save mapping"}), 500

    # =========================================================
    # PUT — Update mapping variable
    # =========================================================
    elif request.method == "PUT":
        js = request.get_json(force=True)
        section = js.get("type")
    
        # ✅ Special case: full JSON import (from Load JSON in settings.html)
        if section == "full" and "data" in js:
            data = js["data"]
            if "meta" in data and "level" in data:
                if save_mapping(data):
                    return jsonify({"success": True})
            return jsonify({"error": "Invalid JSON structure"}), 400
    
        # --- Normal single-field update ---
        original = js.get("original", "").strip()
        variable = js.get("variable", "").strip()
        if section not in ("meta", "level"):
            return jsonify({"error": "Invalid mapping type"}), 400
    
        updated = False
        for m in mapping[section]:
            if m["original"] == original:
                m["variable"] = variable
                updated = True
                break
    
        if not updated:
            return jsonify({"error": f"Field '{original}' not found"}), 404
    
        if save_mapping(mapping):
            return jsonify({"success": True})
        return jsonify({"error": "Failed to save mapping"}), 500
    

    # =========================================================
    # DELETE — Remove mapping entry
    # =========================================================
    elif request.method == "DELETE":
        js = request.get_json(force=True)
        section = js.get("type")
        original = js.get("original", "").strip()

        if section not in ("meta", "level"):
            return jsonify({"error": "Invalid mapping type"}), 400

        before = len(mapping[section])
        mapping[section] = [m for m in mapping[section] if m["original"] != original]
        after = len(mapping[section])

        if before == after:
            return jsonify({"error": f"Field '{original}' not found"}), 404

        if save_mapping(mapping):
            return jsonify({"success": True})
        return jsonify({"error": "Failed to save mapping"}), 500

@app.route("/error_analysis")
@login_required
@page_access_required("error_analysis")
def error_analysis_page():
    return render_template("error_analysis.html")

# ===================================================
# 🎈 RAOB / Radiosonde Analysis Page
# ===================================================

@app.route("/analysis")
@login_required
@page_access_required("analysis")
def analysis_page():
    """
    Halaman RAOB Analysis — menampilkan Skew-T & Hodograph
    dengan filter site / tanggal / file.
    """
    return render_template("analysis.html")

@app.route("/api/serial_lookup/<serial>")
@login_required
def api_serial_lookup(serial):
    import json, re, os

    json_path = "list_data_modem.json"
    if not os.path.exists(json_path):
        json_path = "merged_modem_m20.json"

    if not os.path.exists(json_path):
        return jsonify({"error": "❌ JSON database not found"}), 404

    # --- normalisasi serial number ---
    def normalize_serial(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            try:
                return int(val)
            except Exception:
                pass
        s = str(val).strip()
        digits = re.sub(r"\D", "", s)
        if not digits:
            return None
        try:
            return int(digits)
        except Exception:
            return None

    # --- normalisasi input user ---
    target = normalize_serial(serial)
    if target is None:
        return jsonify({"error": "⚠️ Invalid serial input"}), 400

    # --- load json ---
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return jsonify({"error": f"Failed to load JSON: {e}"}), 500

    print(f"\n📦 Loaded {len(data)} entries from {json_path}")
    print(f"🔍 Searching for serial: {target}")

    # --- pencarian ---
    found = None
    for i, row in enumerate(data):
        row_lower = {str(k).strip().lower(): v for k, v in row.items()}
        sn_json = (
            row_lower.get("nomor serial")
            or row_lower.get("nomor seri")
            or row_lower.get("serial")
            or row_lower.get("no seri")
            or row_lower.get("sn")
        )
        sn_json_int = normalize_serial(sn_json)
        if sn_json_int == target:
            found = row_lower
            break

    if not found:
        print("❌ Serial not found.")
        return jsonify({})

    # --- konversi manufactured/out ke tanggal ---
    manufactured = excel_date_to_str(found.get("manufactured"))
    out = excel_date_to_str(found.get("out"))

    result = {
        "nomor_kardus": found.get("nomor kardus", "-"),
        "nomor_seri": normalize_serial(found.get("nomor serial")) or "-",
        "lokasi": found.get("lokasi", "-"),
        "manufactured": manufactured,
        "out": out,
        "source_file": found.get("source_file", "-"),
    }

    print("📤 Result:", result)
    return jsonify(result)

@app.route("/api/db_files/<site>")
@login_required
def api_db_files(site):
    """List semua file dari database berdasarkan site."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT filename, file_date FROM bufr WHERE site=? "
                    "UNION SELECT filename, file_date FROM bfr WHERE site=? "
                    "UNION SELECT filename, file_date FROM bfh WHERE site=? "
                    "UNION SELECT filename, file_date FROM bin WHERE site=?", (site, site, site, site))
        rows = [{"filename": r[0], "file_date": r[1]} for r in cur.fetchall()]
    return jsonify({"files": rows})

@app.route("/api/insert_from_ftp/<site>/<filename>")
@login_required
def api_insert_from_ftp(site, filename):
    """Download dari FTP → decode → simpan ke DB"""
    try:
        download_and_process(site, filename)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/redownload/<site>/<filename>")
@login_required
def api_redownload(site, filename):
    """Re-download file (hapus dari DB dulu)."""
    ftype = filename.split(".")[-1].lower()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"DELETE FROM {ftype} WHERE site=? AND filename=?", (site, filename))
        conn.commit()
    try:
        download_and_process(site, filename)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/my_info")
@login_required
def my_info():
    user = session.get("user")
    info = VALID_USERS.get(user, {})
    return jsonify({
        "user": user,
        "expiry": info.get("expiry", "-")
    })

@app.route("/release")
@login_required
@page_access_required("release")
def release_page():
    return render_template("release_v1_0.html")

# ===== Custom 403 Forbidden Page =====
@app.errorhandler(403)
def forbidden_error(e):
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Access Denied</title>
        <style>
            body {
                background: linear-gradient(to bottom right, #0a2a6b, #004aad, #0078d7);
                font-family: 'Inter', sans-serif;
                color: white;
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0;
            }
            .box {
                background: rgba(255,255,255,0.12);
                padding: 30px 40px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                backdrop-filter: blur(10px);
                max-width: 400px;
                width: 90%;
            }
            h1 {
                font-size: 1.6em;
                margin-bottom: 10px;
                color: #f87171;
            }
            p {
                opacity: 0.9;
                margin-bottom: 20px;
                line-height: 1.5;
            }
            button {
                background: #38bdf8;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 18px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }
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

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8082,debug=True)
