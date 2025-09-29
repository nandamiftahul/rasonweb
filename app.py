#! /usr/bin/python3
import os, re
import json
import ftplib
import subprocess
import pandas as pd
import numpy as np
from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, session
)
from functools import wraps

from dotenv import load_dotenv
from datetime import datetime

# Load local .env file (ignored in production)
load_dotenv()

# --- Load configuration from environment ---
CONFIG = {
    "ftp": {
        "name": os.getenv("FTP_NAME", "Unknown"),
        "host": os.getenv("FTP_HOST", "localhost"),
        "port": int(os.getenv("FTP_PORT", "21")),
        "user": os.getenv("FTP_USER", "anonymous"),
        "password": os.getenv("FTP_PASS", ""),
        "base_path": os.getenv("FTP_BASE_PATH", "/UA"),
        "file_ext": os.getenv("FTP_FILE_EXT", ".bufr,.bfh,.bfr").split(","),
        "limit": int(os.getenv("FTP_LIMIT", "10"))
    }
}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")

# --- Authentication ---
VALID_USERS = {
    "admin": "meteomodem",
    "trial1": "trialpass",
    "trial2": "12345",
    "guest": "guest123"
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in VALID_USERS and VALID_USERS[username] == password:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# --- Global store ---
rason_levels = []   # list of dicts (per level)
metadata = {}       # dict for station metadata

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

# --- Parse BUFR ---
def parse_bufr(decoded_text):
    meta = {}
    levels = []
    current = {}
    station_lat, station_lon = None, None

    for line in decoded_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # --- Metadata ---
        if "WMO BLOCK NUMBER" in line:
            meta["wmo_block"] = int(line.split()[-1])
        elif "WMO STATION NUMBER" in line:
            meta["wmo_station"] = int(line.split()[-1])
        elif line.startswith("004001 YEAR"):
            meta["year"] = int(line.split()[-1])
        elif line.startswith("004002 MONTH"):
            meta["month"] = int(line.split()[-1])
        elif line.startswith("004003 DAY"):
            meta["day"] = int(line.split()[-1])
        elif line.startswith("004004 HOUR"):
            meta["hour"] = int(line.split()[-1])
        elif line.startswith("004005 MINUTE"):
            meta["minute"] = int(line.split()[-1])
        elif line.startswith("004006 SECOND"):
            meta["second"] = int(line.split()[-1])
        elif "LATITUDE (HIGH ACCURACY)" in line and "005001" in line:
            station_lat = float(line.split()[-1])
            meta["station_lat"] = station_lat
        elif "LONGITUDE (HIGH ACCURACY)" in line and "006001" in line:
            station_lon = float(line.split()[-1])
            meta["station_lon"] = station_lon
        elif "HEIGHT OF STATION GROUND" in line:
            meta["station_height_m"] = float(line.split()[-1])

        # --- Level separator ---
        elif line.startswith("# ---") and current:
            levels.append(current)
            current = {}

        # --- Per-level data ---
        elif "PRESSURE" in line and "007004" in line:
            current["pressure_hPa"] = safe_float(line.split()[-1]) / 100.0
        elif "GEOPOTENTIAL HEIGHT" in line and "010009" in line:
            current["height_m"] = safe_float(line.split()[-1])
        elif "TEMPERATURE/AIR TEMPERATURE" in line:
            current["temp_C"] = safe_float(line.split()[-1]) - 273.15
        elif "DEW-POINT TEMPERATURE" in line:
            current["dewpoint_C"] = safe_float(line.split()[-1]) - 273.15
        elif "WIND DIRECTION" in line and "011001" in line:
            current["wind_dir_deg"] = safe_float(line.split()[-1])
        elif "WIND SPEED" in line and "011002" in line:
            current["wind_speed_mps"] = safe_float(line.split()[-1])
        elif "LATITUDE DISPLACEMENT" in line and "005015" in line:
            current["lat_disp"] = safe_float(line.split()[-1])
        elif "LONGITUDE DISPLACEMENT" in line and "006015" in line:
            current["lon_disp"] = safe_float(line.split()[-1])
        elif "LONG TIME PERIOD OR DISPLACEMENT" in line and "004086" in line:
            current["time_s"] = safe_float(line.split()[-1])
        elif "EXTENDED VERTICAL SOUNDING SIGNIFICANCE" in line and "008042" in line:
            current["status_flag"] = line.split()[-1]
        elif "TRACKING TECHNIQUE/STATUS OF SYSTEM USED" in line and "002014" in line:
            meta["system_status"] = line.split()[-1]

    if current:
        levels.append(current)

    df_meta = pd.DataFrame([meta])
    df_levels = pd.DataFrame(levels).replace({None: np.nan})

    # Ascent rate
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

    # --- Combine launch time ---
    if all(k in meta for k in ("year","month","day","hour","minute","second")):
        meta["launch_time"] = (
            f"{meta['year']:04d}-{meta['month']:02d}-{meta['day']:02d} "
            f"{meta['hour']:02d}:{meta['minute']:02d}:{meta['second']:02d} UTC"
        )
        # remove old fields so they don't appear in df_meta
        for k in ("year","month","day","hour","minute","second"):
            meta.pop(k, None)
    
    # --- Convert to DataFrame AFTER cleanup ---
    df_meta = pd.DataFrame([meta])

    return df_meta, df_levels

# --- FTP utility ---
def fetch_all_sites(ext_filter=None, limit=None, with_meta=False):
    result = {}
    cfg = CONFIG["ftp"]
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
                    # filter extensions (case-insensitive)
                    selected = [f for f in all_files if any(f.lower().endswith(ext) for ext in exts)]
                    selected.sort(reverse=True)
                    selected = selected[: (limit or cfg.get("limit", 10))]

                    for fname in selected:
                        item = {
                            "name": fname,
                            "site": site,
                            "file_date": extract_date_from_filename(fname)
                        }
                        if with_meta:
                            try:
                                # download file temporarily
                                local_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
                                with open(local_path, "wb") as f:
                                    ftp.retrbinary(f"RETR " + fname, f.write)
                    
                                decoded = decode_bufr(local_path)
                                df_meta, df_levels = parse_bufr(decoded)
                                if not df_meta.empty:
                                    launch_time = df_meta.iloc[0].get("launch_time")
                                    if launch_time:
                                        item["launch_time"] = launch_time
                    
                                # run flight analysis
                                issues = analyze_flight(df_meta, df_levels)
                                if issues:
                                    item["flight_issues"] = issues
                    
                            except Exception as e:
                                item["launch_time"] = f"Error: {e}"
                                item["flight_issues"] = [f"Error: {e}"]
                        site_files.append(item)
                    
                    ftp.cwd(cfg["base_path"])  # back to root
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
    limit = int(request.args.get("limit") or CONFIG["ftp"].get("limit", 10))
    sites = fetch_all_sites(ext_filter=[ext] if ext else None, limit=limit, with_meta=False)
    return jsonify(sites)

@app.route("/api/sites_with_meta")
@login_required
def api_sites_with_meta():
    ext = request.args.get("ext") or None
    limit = int(request.args.get("limit") or CONFIG["ftp"].get("limit", 10))
    sites = fetch_all_sites(ext_filter=[ext] if ext else None, limit=limit, with_meta=True)
    return jsonify(sites)

def download_and_process(site, filename):
    """Fetch BUFR file from FTP and process into rason_levels + metadata."""
    global rason_levels, metadata
    cfg = CONFIG["ftp"]
    local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(f"{cfg['base_path']}/{site}")
            with open(local_path, "wb") as f:
                ftp.retrbinary(f"RETR {filename}", f.write)

        decoded = decode_bufr(local_path)
        df_meta, df_levels = parse_bufr(decoded)
        
        issues = analyze_flight(df_meta, df_levels)   # ⬅️ add here
        metadata = df_meta.to_dict("records")[0] if not df_meta.empty else {}
        metadata["flight_issues"] = issues            # ⬅️ save into metadata
        rason_levels = df_levels.to_dict("records") if not df_levels.empty else []
        
    except Exception as e:
        print(f"FTP download error: {e}")
        metadata, rason_levels = {}, []

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

# --- Routes ---
@app.route("/dashboard")
@login_required
def dashboard():
    # Default to ".bfr" if ext is missing or empty
    selected_ext = request.args.get("ext")
    if not selected_ext:
        selected_ext = ".bfr"

    limit = request.args.get("limit", type=int, default=2)

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
    try:
        local_path = download_from_ftp(site, filename)
        decoded = decode_bufr(local_path)
        df_meta, _ = parse_bufr(decoded)
        meta = df_meta.to_dict("records")[0] if not df_meta.empty else {}
        return jsonify({"launch_time": meta.get("launch_time", "-")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/load_from_ftp/<site>/<filename>")
@login_required
def load_from_ftp(site, filename):
    download_and_process(site, filename)
    return redirect(url_for("index"))

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    global rason_levels, metadata
    if request.method == "POST":
        f = request.files["rasonfiles"]
        if not f or not f.filename:
            return redirect(url_for("index"))
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
        f.save(filepath)

        decoded = decode_bufr(filepath)
        df_meta, df_levels = parse_bufr(decoded)
        
        issues = analyze_flight(df_meta, df_levels)   # ⬅️ add here
        metadata = df_meta.to_dict("records")[0] if not df_meta.empty else {}
        metadata["flight_issues"] = issues            # ⬅️ save into metadata
        rason_levels = df_levels.to_dict("records") if not df_levels.empty else []


        return redirect(url_for("index"))

    # 👉 Pass user to template
    return render_template("map.html", total=len(rason_levels), user=session.get("user"))

@app.route("/value")
@login_required
def rason_value():
    if not rason_levels:
        return jsonify({"error": "No radiosonde"}), 404
    idx = int(request.args.get("frame", 0)) % len(rason_levels)
    return jsonify(rason_levels[idx])

@app.route("/metadata")
@login_required
def rason_metadata():
    if not metadata:
        return jsonify({"error": "No metadata"}), 404
    return jsonify(metadata)

@app.route("/all_levels")
@login_required
def all_levels_route():
    return jsonify(rason_levels)


if __name__ == "__main__":
    app.run(debug=True)
