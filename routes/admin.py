# rason_backend/routes/admin.py
from flask import Blueprint, jsonify, session
from core.auth import login_required, bump_global_session_version, get_global_session_version
from core.auth import ACTIVE_USERS, USER_SESSION_TOKENS

admin = Blueprint("admin", __name__, url_prefix="/admin")

@admin.route("/logout_all", methods=["POST"])
@login_required
def logout_all_users():
    if session.get("user") != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    bump_global_session_version()
    return jsonify({"success": True, "new_version": get_global_session_version()})

@admin.route("/logout_user/<username>", methods=["POST"])
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

@admin.route("/active_users")
@login_required
def api_active_users():
    # Hanya admin yang boleh melihat daftar login aktif
    if session.get("user") != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify({"active": sorted(list(ACTIVE_USERS))})
