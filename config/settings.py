# rason_backend/config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# === Paths & Core settings ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "rason_data.db"))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))
SECRET_KEY = os.getenv("SECRETKEY", "Unknown")  # asalnya dipakai untuk Fernet & Flask secret

# === Files ===
SITES_FILE = os.getenv("SITES_FILE", os.path.join(BASE_DIR, "sites.json"))
USERS_FILE = os.getenv("USERS_FILE", os.path.join(BASE_DIR, "users.json"))
BUFRCFG_FULL = os.path.join(BASE_DIR, "bufr_mapping_full.json")
MODEM_LIST_JSON = os.path.join(BASE_DIR, "list_data_modem.json")

# === FTP Config ===
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
    }
}
