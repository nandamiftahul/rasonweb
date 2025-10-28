#!/usr/bin/env python3
"""
Simple migrator: load users.json (plaintext fields) and re-save them
using the encryption logic from app.py. This script expects that
app.secret_key or CFG secret is set in environment, or you can pass it.
"""

import os, sys, json
from getpass import getpass
from cryptography.fernet import Fernet
import base64, hashlib

USERS_FILE = "users.json"

def derive_key(secret):
    h = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(h)

def encrypt_value(val, f):
    import json as _json
    raw = _json.dumps(val, ensure_ascii=False).encode("utf-8")
    return f.encrypt(raw).decode("utf-8")

if __name__ == "__main__":
    if not os.path.exists(USERS_FILE):
        print("users.json not found in current dir.")
        sys.exit(1)

    secret = os.environ.get("SECRETKEY") or os.environ.get("FTP_SECRET") or input("Enter secret to derive Fernet key: ")
    key = derive_key(secret)
    f = Fernet(key)

    with open(USERS_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    out = {}
    for u, info in data.items():
        if isinstance(info, dict):
            saved = {}
            for k, v in info.items():
                if k == "password":
                    saved[k] = v
                else:
                    # if already looks encrypted, keep
                    if isinstance(v, str) and v.startswith("gAAAA"):
                        saved[k] = v
                    else:
                        saved[k] = encrypt_value(v, f)
            out[u] = saved
        else:
            out[u] = info

    backup = USERS_FILE + ".bak"
    os.rename(USERS_FILE, backup)
    with open(USERS_FILE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"Migration complete. original backed up to {backup}")
