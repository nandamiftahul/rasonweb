# rason_backend/routes/pages.py
from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import check_password_hash
import time, os, re
from datetime import datetime
from core.auth import login_required, page_access_required, VALID_USERS, ACTIVE_USERS, USER_SESSION_TOKENS, get_global_session_version, get_user_store
from core.auth import save_users  # if needed
from core.ftp import fetch_all_sites
from config.settings import SECRET_KEY
import secrets

pages = Blueprint("pages", __name__)

@pages.route("/login", methods=["GET","POST"])
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
        session["last_active"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # 🟢 Tambahkan ke daftar user aktif
        ACTIVE_USERS.add(username)
        print(f"✅ User logged in: {username} (active: {list(ACTIVE_USERS)})")

        return redirect(url_for("pages.main_page"))

    # GET method
    return render_template("login.html")

@pages.route("/logout")
@login_required
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
    resp = redirect(url_for("pages.login"))
    resp.set_cookie('session', '', expires=0)
    return resp

@pages.route("/")
@login_required
def index():
    if request.method == "POST":
        f = request.files["rasonfiles"]
        if not f or not f.filename:
            return redirect(url_for("pages.index"))
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
        return redirect(url_for("pages.map_view", t=int(time.time())))

    # GET: tampilkan halaman utama / dashboard
    return render_template("main.html", user=session.get("user"))

@pages.route("/main")
@login_required
@page_access_required("main")
def main_page():
    return render_template("main.html")

@pages.route("/dashboard")
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

@pages.route("/map")
@login_required
@page_access_required("map")
def map_view():
    t = request.args.get("t", str(int(time.time())))
    store = get_user_store()
    total = len(store.get("levels", []))
    user = session.get("user")
    return render_template("map.html", total=total, user=user, t=t)

# halaman lain yang pure render
@pages.route("/underdev")
@login_required
def under_development(): return render_template("under_development.html")

@pages.route("/time_accuracy")
@login_required
@page_access_required("time_accuracy")
def time_accuracy(): return render_template("time_accuracy.html")

@pages.route("/height_reach")
@login_required
@page_access_required("height_reach")
def height_reach(): return render_template("height_reach.html")

@pages.route("/data_availability")
@login_required
@page_access_required("data_availability")
def data_availability_page(): return render_template("data_availability.html")

@pages.route("/raob_doc")
@login_required
@page_access_required("raob_doc")
def raob_doc(): return render_template("raob_doc.html")

@pages.route("/analysis")
@login_required
@page_access_required("analysis")
def analysis_page(): return render_template("analysis.html")

@pages.route("/error_analysis")
@login_required
@page_access_required("error_analysis")
def error_analysis_page():
    return render_template("error_analysis.html")

@pages.route("/trajectory3d")
@login_required
@page_access_required("trajectory3d")
def trajectory3d_page(): return render_template("trajectory3d.html")

@pages.route("/settings")
@login_required
@page_access_required("settings")
def settings_page():
    username = session.get("user")
    return render_template("settings.html", users=VALID_USERS, current_user=username)

@pages.route("/release")
@login_required
@page_access_required("release")
def release_page(): return render_template("release_v1_1.html")

@pages.route("/roadmap")
@login_required
@page_access_required("roadmap")
def roadmap_page():
    return render_template("roadmap.html")

@pages.route("/user_guide")
def user_guide_page():
    return render_template("user_guide.html")

@pages.route("/display")
@login_required
def display_page():
    return render_template("display.html")
