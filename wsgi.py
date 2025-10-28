import os, sys
from pathlib import Path

# Pastikan project root /app dikenali oleh Python
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Debug print (boleh hapus setelah berhasil)
print("✅ WSGI running from:", os.getcwd())
print("✅ sys.path includes:", sys.path)

# Import create_app
from app import create_app

app = create_app()
