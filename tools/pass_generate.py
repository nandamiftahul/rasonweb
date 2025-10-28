import json
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash

INPUT_FILE = "users.json.backup"         # file lama (plaintext)
OUTPUT_FILE = "users_hashed.json" # file baru (hashed + expiry)

# 🗓️ Tanggal default: 1 tahun dari hari ini (UTC)
default_expiry = (datetime.utcnow() + timedelta(days=365)).strftime("%Y-%m-%d")

with open(INPUT_FILE, "r") as f:
    users = json.load(f)

hashed_users = {}

for user, plain_pw in users.items():
    hashed_pw = generate_password_hash(plain_pw)
    hashed_users[user] = {
        "password": hashed_pw,
        "expiry": default_expiry
    }
    print(f"✅ {user} hashed — expiry set to {default_expiry}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(hashed_users, f, indent=2)

print(f"\n🔐 Done! Saved hashed users with expiry to {OUTPUT_FILE}")
