import json
from werkzeug.security import generate_password_hash

INPUT_FILE = "users.json"
OUTPUT_FILE = "users_hashed.json"

with open(INPUT_FILE, "r") as f:
    users = json.load(f)

hashed_users = {}
for user, plain_pw in users.items():
    hashed_users[user] = generate_password_hash(plain_pw)
    print(f"✅ {user} hashed")

with open(OUTPUT_FILE, "w") as f:
    json.dump(hashed_users, f, indent=2)

print(f"\n🔐 Done! Saved hashed passwords to {OUTPUT_FILE}")
