#! /usr/bin/python3
import os, re, io, configparser
import json
import ftplib
import subprocess
import pandas as pd
import numpy as np
from flask import (
    Flask, render_template, request, redirect,
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

import metpy.calc as mpcalc
from metpy.units import units
from metpy.plots import SkewT, Hodograph
from scipy.signal import medfilt
from geopy.distance import geodesic

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
        "file_ext": os.getenv("FTP_FILE_EXT", ".bufr,.bfh,.bfr,.bin").split(","),
        "limit": int(os.getenv("FTP_LIMIT", "30"))
    }
}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")

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

USER_FILE = "users.json"

def load_users():
    """Load user data from JSON file or fallback to default."""
    global VALID_USERS
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, "r") as f:
                VALID_USERS = json.load(f)
        except Exception as e:
            print("⚠️ Failed to read users.json:", e)
    else:
        # Default admin user
        VALID_USERS = {"admin": "admin123"}
        save_users()

def save_users():
    """Save user data to JSON file."""
    try:
        with open(USER_FILE, "w") as f:
            json.dump(VALID_USERS, f, indent=2)
    except Exception as e:
        print("⚠️ Failed to save users.json:", e)

# --- Authentication ---
VALID_USERS = {
    "admin": "meteomodem",
    "trial1": "trialpass",
    "trial2": "12345",
    "guest": "guest123",
    "user1": "user123",
    "user2": "user123",
    "user3": "user123",
    "bmkg" : "Bmkg2025$"
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
            return redirect(url_for("main_page"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    clear_user_store()
    session.pop("user", None)
    return redirect(url_for("login"))

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

    def _extract_bytes_field(line: str) -> str:
        """
        Extract b'..' or b"..". Returns stripped string, else last token.
        """
        m = re.search(r"b[\"'](.*?)[\"']", line)
        if m:
            return m.group(1).strip()
        # fallback to entire tail
        return line.split()[-1]

    def _safe_float_tail(line: str):
        try:
            return float(line.split()[-1])
        except Exception:
            return None

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
        # --- Radiosonde/balloon/instrument metadata (add these) ---
        elif "RADIOSONDE SERIAL NUMBER" in line or "001081" in line:
            meta["radiosonde_serial_number"] = _extract_bytes_field(line)
        elif "RADIOSONDE ASCENSION NUMBER" in line or "001082" in line:
            try:
                meta["radiosonde_ascension_number"] = int(line.split()[-1])
            except Exception:
                pass
        elif "RADIOSONDE RELEASE NUMBER" in line or "001083" in line:
            try:
                meta["radiosonde_release_number"] = int(line.split()[-1])
            except Exception:
                pass
        elif "RADIOSONDE GROUND RECEIVING SYSTEM" in line or "002066" in line:
            try:
                meta["radiosonde_ground_rx_system"] = int(line.split()[-1])
            except Exception:
                pass
        elif "RADIOSONDE OPERATING FREQUENCY" in line or "002067" in line:
            v = _safe_float_tail(line)
            if v is not None:
                meta["radiosonde_operating_frequency"] = v  # Hz
        elif "BALLOON MANUFACTURER" in line or "002080" in line:
            try:
                meta["balloon_manufacturer"] = int(line.split()[-1])
            except Exception:
                pass
        elif "WEIGHT OF BALLOON" in line or "002082" in line:
            v = _safe_float_tail(line)
            if v is not None:
                meta["balloon_weight_kg"] = v
        elif "TYPE OF GAS USED IN BALLOON" in line or "002084" in line:
            try:
                meta["balloon_gas_type"] = int(line.split()[-1])
            except Exception:
                pass
        elif "TYPE OF PRESSURE SENSOR" in line or "002095" in line:
            try:
                meta["pressure_sensor_type"] = int(line.split()[-1])
            except Exception:
                pass
        elif "TYPE OF TEMPERATURE SENSOR" in line or "002096" in line:
            try:
                meta["temperature_sensor_type"] = int(line.split()[-1])
            except Exception:
                pass
        elif "TYPE OF HUMIDITY SENSOR" in line or "002097" in line:
            try:
                meta["humidity_sensor_type"] = int(line.split()[-1])
            except Exception:
                pass
        elif "SOFTWARE IDENTIFICATION AND VERSION NUMBER" in line or "025061" in line:
            meta["software_version"] = _extract_bytes_field(line)
        elif "REASON FOR TERMINATION" in line or "008021" in line:
            try:
                meta["reason_for_termination"] = int(line.split()[-1])
            except Exception:
                pass

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

# --- Mappings ---
REASON_MAP = {
    0: "Not specified",
    1: "Balloon burst",
    2: "Battery exhausted",
    3: "Receiver failure",
    4: "Telemetry interrupted",
    5: "Manual termination",
    6: "Other",
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

    _tz = None  # fallback (pakai UTC kalau zoneinfo tidak ada)

    # --- Default window: today 00:00 WIT -> tomorrow 00:00 WIT, converted to UTC (naive) ---
    if start_date is None or end_date is None:
        now_local = datetime.now(_tz) if _tz else datetime.utcnow().replace(tzinfo=timezone.utc)
        today_local_midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow_local_midnight = today_local_midnight + timedelta(days=1)

        def to_utc_naive(dt_local):
            # jadikan UTC naive agar comparable dengan dt hasil strptime (naive) dari filename "UTC"
            if dt_local.tzinfo is None:
                return dt_local  # sudah naive (anggap UTC)
            return dt_local.astimezone(timezone.utc).replace(tzinfo=None)

        if start_date is None:
            start_date = to_utc_naive(today_local_midnight)
        if end_date is None:
            end_date = to_utc_naive(tomorrow_local_midnight)

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
                    print(f"[DEBUG] {site}: total files returned by FTP = {len(all_files)}")
                    selected = [f for f in all_files if any(f.lower().endswith(ext) for ext in exts)]
                    # Attach parsed date to each file
                    items = []
                    for fname in selected:
                        dt_str = extract_date_from_filename(fname)
                        try:
                            # contoh format: "%Y-%m-%d %H:%M:%S UTC"
                            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S UTC")
                            # ⚖️ Filter dengan window [start_date, end_date)
                            if start_date or end_date:
                                if start_date and dt < start_date:
                                    continue
                                if end_date and dt >= end_date:  # end eksklusif
                                    continue
                        except Exception:
                            dt = datetime.min
                        items.append((fname, dt))
                    
                    # Sort by datetime (newest first)
                    items.sort(key=lambda x: x[1], reverse=True)
                    
                    # Only keep the last N (limit)
                    if limit: items = items[:limit]
                    
                    # Build back list of names
                    selected = [fname for fname, dt in items]
                    
                    for fname in selected:
                        item = {
                            "name": fname,
                            "site": site,
                            "file_date": extract_date_from_filename(fname)
                        }
                        if with_meta:
                            try:
                                local_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
                                with open(local_path, "wb") as f:
                                    ftp.retrbinary(f"RETR " + fname, f.write)

                                decoded = decode_bufr(local_path)
                                df_meta, df_levels = parse_bufr(decoded)

                                if not df_meta.empty:
                                    meta_row = df_meta.iloc[0]

                                    # Launch time
                                    launch_time = meta_row.get("launch_time")
                                    if launch_time:
                                        item["launch_time"] = launch_time

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

                                # Flight issues
                                issues = analyze_flight(df_meta, df_levels)
                                if issues:
                                    item["flight_issues"] = issues

                                # --- Extra derived flight metadata ---
                                try:
                                    if not df_levels.empty:
                                        # End pressure
                                        end_pressure = df_levels["pressure_hPa"].dropna().min()
                                        if pd.notna(end_pressure):
                                            item["end_pressure"] = f"{end_pressure:.1f} hPa"

                                        # Max height
                                        max_height = df_levels["height_m"].dropna().max()
                                        if pd.notna(max_height):
                                            item["max_height"] = f"{max_height:.0f} m"

                                        # End time
                                        if "time_s" in df_levels and not df_meta.empty:
                                            launch_time = pd.to_datetime(meta_row.get("launch_time"))
                                            if pd.notna(launch_time):
                                                end_time = launch_time + pd.to_timedelta(df_levels["time_s"].max(), unit="s")
                                                item["end_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")

                                        # Distance from station
                                        if {"latitude", "longitude"} <= set(df_levels.columns) and not df_meta.empty:
                                            last_lat = df_levels["latitude"].dropna().iloc[-1]
                                            last_lon = df_levels["longitude"].dropna().iloc[-1]
                                            st_lat = meta_row.get("station_lat")
                                            st_lon = meta_row.get("station_lon")
                                            if st_lat and st_lon and pd.notna(last_lat) and pd.notna(last_lon):
                                                dist = geodesic((st_lat, st_lon), (last_lat, last_lon)).km
                                                item["end_distance"] = f"{dist:.1f} km"

                                        # Avg ascent rate
                                        if "height_m" in df_levels and "time_s" in df_levels:
                                            elapsed = df_levels["time_s"].max() - df_levels["time_s"].min()
                                            if elapsed > 0 and pd.notna(max_height):
                                                ascent_rate = max_height / elapsed
                                                item["avg_ascent_rate"] = f"{ascent_rate:.2f} m/s"
                                except Exception as e:
                                    print("Extra metadata calc failed in fetch_all_sites:", e)

                            except Exception as e:
                                item["launch_time"] = f"Error: {e}"
                                item["flight_issues"] = [f"Error: {e}"]

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
    sites = fetch_all_sites(with_meta=True, limit=1)
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
    Membaca status balon dari file status.ini di FTP tiap site.
    Mengembalikan JSON: site, status, dan waktu update terakhir.
    """
    cfg = CONFIG["ftp"]
    sites = ["aceh", "tarakan", "sorong", "cilacap", "pangkalanbun", "ranai"]
    results = []

    
    #print("start FTP")
    try:
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            #print("connected")
            for site in sites:
                status = "offline"
                update = "-"
                try:
                    # ⚙️ Jangan pakai .lower(), karena nama folder FTP case-sensitive
                    ftp.cwd(f"{cfg['base_path']}/{site}")
                    f = io.BytesIO()
                    ftp.retrbinary("RETR status.ini", f.write)
                    f.seek(0)
                    parser = configparser.ConfigParser()
                    parser.read_string("[root]\n" + f.read().decode())
                    status = parser.get("root", "status", fallback="offline")
                    update_str = parser.get("root", "update", fallback="-")
                    if update_str.isdigit() and len(update_str) == 14:
                        dt = datetime.strptime(update_str, "%Y%m%d%H%M%S")
                        update = dt.strftime("%Y-%m-%d %H:%M UTC")
                    else:
                        update = update_str
                except Exception as e:
                    print(f"⚠️ Gagal baca {site}: {e}")
                #print({"site": site, "status": status, "update": update})
                results.append({"site": site, "status": status, "update": update})
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
    """Fetch BUFR from FTP and save into THIS user's store."""
    cfg = CONFIG["ftp"]
    local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        # --- Download from FTP ---
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(f"{cfg['base_path']}/{site}")
            with open(local_path, "wb") as f:
                ftp.retrbinary(f"RETR " + filename, f.write)

        # --- Decode & parse ---
        decoded = decode_bufr(local_path)
        df_meta, df_levels = parse_bufr(decoded)
        issues = analyze_flight(df_meta, df_levels)

        # --- User store ---
        store = get_user_store()
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
                    3: "Receiver failure",
                    4: "Telemetry interrupted",
                    5: "Manual termination",
                    6: "Other",
                }
                meta["reason_for_termination"] = f"{code} – {reason_map.get(code, 'Unknown')}"
            except Exception:
                pass

        # Sensor/Balloon type code tables.
        # We handle BOTH possible key styles that might come from parse_bufr.
        sensor_maps = {
            "pressure": {0:"Unknown",1:"Aneroid",2:"Capacitive",3:"Other"},
            "temperature": {0:"Unknown",1:"Thermistor",2:"Platinum",3:"Other"},
            "humidity": {0:"Unknown",1:"Hair",2:"Capacitive",3:"Carbon",4:"Other"},
            "balloon": {0:"Unknown",1:"Latex",2:"Polyethylene",3:"Other"},
            "balloon_gas": {0:"Unknown",1:"Hydrogen",2:"Helium"},
            "balloon_manufacturer": {0:"Unknown",1:"Totex",2:"Kaysam",3:"Other"},
        }
        # key aliases seen in different decoders
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

                # Avg ascent rate = max_height / elapsed_seconds
                if "height_m" in df_levels and "time_s" in df_levels and df_levels["time_s"].notna().any():
                    max_h = df_levels["height_m"].dropna().max()
                    elapsed = df_levels["time_s"].max() - df_levels["time_s"].min()
                    if pd.notna(max_h) and elapsed and elapsed > 0:
                        meta["avg_ascent_rate"] = f"{(max_h/elapsed):.2f} m/s"
        except Exception as e:
            print("Extra metadata calc failed in download_and_process:", e)

        # --- Save & exit ---
        meta["flight_issues"] = issues
        store["levels"] = df_levels.to_dict("records") if not df_levels.empty else []

    except Exception as e:
        print(f"FTP download error: {e}")
        store = get_user_store()
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
    try:
        local_path = download_from_ftp(site, filename)
        decoded = decode_bufr(local_path)
        df_meta, _ = parse_bufr(decoded)
        meta = df_meta.to_dict("records")[0] if not df_meta.empty else {}
        return jsonify({
            "launch_time": meta.get("launch_time", "-"),
            "radiosonde_serial_number": meta.get("radiosonde_serial_number", "-")
        })
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

        # 🔧 Tambahkan analisis issues di sini
        issues = analyze_flight(df_meta, df_levels)

        store = get_user_store()
        store["metadata"] = df_meta.to_dict("records")[0] if not df_meta.empty else {}

        # --- Normalize radiosonde frequency (Hz → MHz with 3 decimals) ---
        if "radiosonde_operating_frequency" in store["metadata"]:
            try:
                hz_val = float(store["metadata"]["radiosonde_operating_frequency"])
                mhz_val = hz_val / 1e6
                store["metadata"]["radiosonde_operating_frequency"] = f"{mhz_val:.3f} MHz"
            except Exception:
                pass

        store["metadata"]["flight_issues"] = issues
        store["levels"] = df_levels.to_dict("records") if not df_levels.empty else []

        return redirect(url_for("index"))

    store = get_user_store()
    return render_template("map.html", total=len(store["levels"]), user=session.get("user"))

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
    if not store["metadata"]:
        return jsonify({"error": "No metadata"}), 404
    return jsonify(store["metadata"])

@app.route("/all_levels")
@login_required
def all_levels_route():
    store = get_user_store()
    return jsonify(store["levels"])

@app.route("/download_wmo/<site>/<filename>")
@login_required
def download_wmo(site, filename):
    try:
        local_path = download_from_ftp(site, filename)
        decoded = decode_bufr(local_path)
        df_meta, df_levels = parse_bufr(decoded)
        wmo_text = generate_wmo_temp(df_meta, df_levels)

        return Response(
            wmo_text,
            mimetype="text/plain",
            headers={"Content-Disposition": f"attachment;filename={filename}.wmo.txt"}
        )
    except Exception as e:
        return f"Error generating WMO: {e}", 500

@app.route("/raob/<site>/<filename>")
@login_required
def raob_analysis(site, filename):
    try:
        # --- Fetch & decode BUFR ---
        local_path = download_from_ftp(site, filename)
        decoded = decode_bufr(local_path)
        df_meta, df_levels = parse_bufr(decoded)

        if df_levels.empty:
            return "No levels found", 500

        # --- Metadata harus diambil lebih awal ---
        meta = df_meta.to_dict("records")[0] if not df_meta.empty else {}

        # --- Clean profile ---
        df = df_levels.dropna(subset=["pressure_hPa"]).copy()
        df = df.sort_values("pressure_hPa", ascending=False)
        df["pressure_hPa"] = medfilt(df["pressure_hPa"].values, kernel_size=3)

        if "temp_C" in df:
            df["temp_C"] = pd.Series(df["temp_C"]).interpolate(limit_direction="both")
            df["temp_C"] = medfilt(df["temp_C"].values, kernel_size=3)

        if "dewpoint_C" in df:
            df["dewpoint_C"] = pd.Series(df["dewpoint_C"]).interpolate(limit_direction="both")
            df["dewpoint_C"] = medfilt(df["dewpoint_C"].values, kernel_size=3)

        df = df.drop_duplicates(subset=["pressure_hPa"])
        df = df.sort_values("pressure_hPa", ascending=False).reset_index(drop=True)

        # --- Thermo profile ---
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

        # --- Thermodynamic indices ---
        lcl_pressure, lcl_temp = mpcalc.lcl(p[0], T[0], Td[0])
        parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to("degC")
        cape, cin = mpcalc.cape_cin(p, T, Td, parcel_prof)
        li = mpcalc.lifted_index(p, T, parcel_prof)
        ki = mpcalc.k_index(p, T, Td)

        # --- Wind shear ---
        shear_0_1km_mag = np.nan * units("knot")
        shear_0_6km_mag = np.nan * units("knot")
        if (u is not None) and (v is not None):
            try:
                sh_u_1km, sh_v_1km = mpcalc.bulk_shear(p_w, u, v, depth=1000 * units.meter)
                sh_u_6km, sh_v_6km = mpcalc.bulk_shear(p_w, u, v, depth=6000 * units.meter)
                shear_0_1km_mag = mpcalc.wind_speed(sh_u_1km, sh_v_1km).to("knot")
                shear_0_6km_mag = mpcalc.wind_speed(sh_u_6km, sh_v_6km).to("knot")
            except Exception:
                pass

        # --- SRH ---
        srh_0_1km = srh_0_3km = np.nan * units("m^2/s^2")
        if (u is not None) and (v is not None) and (hgt is not None):
            try:
                sort_idx = np.argsort(hgt)
                hgt_sorted = hgt[sort_idx]
                u_sorted = u[sort_idx]
                v_sorted = v[sort_idx]
                rmotion, _, _ = mpcalc.bunkers_storm_motion(p_w, u_sorted, v_sorted, hgt_sorted)
                storm_u, storm_v = rmotion
                srh_0_1km, _, _ = mpcalc.storm_relative_helicity(
                    hgt_sorted, u_sorted, v_sorted,
                    depth=1000*units.m, bottom=0*units.m,
                    storm_u=storm_u, storm_v=storm_v)
                srh_0_3km, _, _ = mpcalc.storm_relative_helicity(
                    hgt_sorted, u_sorted, v_sorted,
                    depth=3000*units.m, bottom=0*units.m,
                    storm_u=storm_u, storm_v=storm_v)
            except Exception as e:
                print("SRH calculation failed:", e)

        # --- Freezing Level ---
        freezing_level = "-"
        try:
            idx = np.where(np.diff(np.sign(T.m)))[0]
            if idx.size > 0:
                freezing_level = f"{thermo.iloc[idx[0]]['pressure_hPa']:.0f} hPa"
        except Exception:
            pass

        # --- Tropopause ---
        tropopause_level = "-"
        try:
            if {"height_m", "temp_C", "pressure_hPa"} <= set(thermo.columns):
                T_vals = thermo["temp_C"].values
                Z_vals = thermo["height_m"].values
                P_vals = thermo["pressure_hPa"].values
                mask = P_vals < 500
                T_vals, Z_vals, P_vals = T_vals[mask], Z_vals[mask], P_vals[mask]
                if len(T_vals) > 5:
                    lapse_rate = np.gradient(T_vals, Z_vals) * 1000.0
                    for i in range(len(Z_vals)):
                        if lapse_rate[i] <= 2.0:
                            z_top = Z_vals[i] + 2000.0
                            mask2 = (Z_vals >= Z_vals[i]) & (Z_vals <= z_top)
                            if np.any(mask2) and np.mean(lapse_rate[mask2]) <= 2.0:
                                tropopause_level = f"{P_vals[i]:.0f} hPa"
                                break
        except Exception as e:
            print("Tropopause calc failed:", e)

        # --- Skew-T plot ---
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
        skew.ax.set_xlabel("Temperature (°C)")
        skew.ax.set_ylabel("Pressure (hPa)")
        skew.ax.set_title(f"{meta.get('wmo_station','')}  {meta.get('launch_time','')}",
                          fontsize=10, fontweight="bold", color="#222")
        buf1 = BytesIO()
        plt.savefig(buf1, format="png", bbox_inches="tight")
        buf1.seek(0)
        skewt_img = base64.b64encode(buf1.read()).decode("utf-8")
        plt.close(fig1)

        # --- Hodograph (Cartesian style + compass labels) ---
        if (u is not None) and (v is not None):
            fig2, ax = plt.subplots(figsize=(6, 6))
            hodo = Hodograph(ax, component_range=60.0)
            hodo.add_grid(increment=10)
        
            # 🔄 Rotasi 180° (berlawanan arah jarum jam)
            u_rot = -u.to("m/s")
            v_rot = -v.to("m/s")
        
            # Plot garis utama
            hodo.plot(u_rot, v_rot, color="#007bff", linewidth=2, label="Wind profile")
        
            # Scatter dengan warna tinggi
            if hgt is not None:
                mask = ~np.isnan(hgt.m)
                ax.scatter(
                    u_rot[mask].to("m/s"),
                    v_rot[mask].to("m/s"),
                    c=hgt[mask].m / 1000.0,
                    cmap="viridis",
                    s=30,
                    edgecolors="black",
                    linewidths=0.3,
                    label="Height (km)"
                )
        
            # Background & grid
            ax.set_facecolor("#f9f9f9")
            ax.grid(True, linestyle="--", color="gray", alpha=0.5)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("U wind (m/s)")
            ax.set_ylabel("V wind (m/s)")
            ax.set_title("Hodograph", fontsize=10, fontweight="bold", color="#004085")
            ax.legend(fontsize=8, loc="upper left")
        
            # 🔠 Tambahkan label arah mata angin
            lim = 60  # disesuaikan dengan component_range
            offset = lim * 0.9
            ax.text(0,  offset, "N", fontsize=10, fontweight="bold", ha="center", va="bottom", color="#222")
            ax.text(0, -offset, "S", fontsize=10, fontweight="bold", ha="center", va="top", color="#222")
            ax.text( offset, 0, "E", fontsize=10, fontweight="bold", ha="left", va="center", color="#222")
            ax.text(-offset, 0, "W", fontsize=10, fontweight="bold", ha="right", va="center", color="#222")
        
            # Simpan ke buffer
            buf2 = BytesIO()
            plt.savefig(buf2, format="png", bbox_inches="tight")
            buf2.seek(0)
            hodo_img = base64.b64encode(buf2.read()).decode("utf-8")
            plt.close(fig2)
        else:
            hodo_img = None
        
        

        # --- Indices table ---
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
            "Tropopause": tropopause_level,
        }

        # --- Auto Weather Analysis ---
        analysis_text = generate_weather_analysis(df)

        # --- Render Template ---
        return render_template(
            "raob.html",
            meta=meta,
            indices=indices,
            skewt_img=skewt_img,
            hodo_img=hodo_img,
            analysis_text=analysis_text
        )

    except Exception as e:
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
def main_page():
    return render_template("main.html")

@app.route("/map")
@login_required
def map():
    return render_template("map.html")

@app.route("/underdev")
@login_required
def under_development():
    return render_template("under_development.html")

@app.route("/time_accuracy")
@login_required
def time_accuracy():
    """
    Halaman grafik time accuracy (Launch→100hPa dan Launch→30hPa)
    """
    return render_template("time_accuracy.html")
    
@app.route("/api/time_accuracy/<site>")
@login_required
def api_time_accuracy(site):
    """
    Ambil data time accuracy dari file .bfr bulan berjalan untuk site tertentu.
    - Patokan waktu: dari nama file (bukan metadata)
    - Jika hanya 100 hPa tercapai, tetap disertakan
    - Jam disimpan sebagai '00Z' atau '12Z'
    """
    from datetime import datetime, timedelta
    import pandas as pd
    import re

    now = datetime.utcnow()
    start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_date = (start_date + timedelta(days=32)).replace(day=1)

    def extract_datetime_from_filename(fname: str):
        """
        Contoh nama: T2097502A202510050000.BFR -> datetime(2025,10,5,0,0)
        """
        match = re.search(r"(\d{10,14})", fname)
        if not match:
            return None
        dt_str = match.group(1)
        try:
            if len(dt_str) == 10:
                return datetime.strptime(dt_str, "%Y%m%d%H")
            elif len(dt_str) == 12:
                return datetime.strptime(dt_str, "%Y%m%d%H%M")
            elif len(dt_str) == 14:
                return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
        except Exception:
            return None
        return None

    data = []
    try:
        all_sites = fetch_all_sites(
            ext_filter=[".bfr"],
            with_meta=False,
            start_date=start_date,
            end_date=end_date
        )

        site_keys = {s.lower(): s for s in all_sites.keys()}
        site_key = site.lower()

        if site_key not in site_keys:
            print(f"⚠️ Site {site} not found in FTP")
            return jsonify({"site": site, "data": []})

        true_site = site_keys[site_key]
        for f in all_sites[true_site]:
            fname = f["name"]
            try:
                # --- Ambil waktu dari nama file (bukan metadata) ---
                file_dt = extract_datetime_from_filename(fname)
                if not file_dt:
                    print(f"⚠️ Skip (no datetime): {fname}")
                    continue

                date_str = file_dt.strftime("%Y-%m-%d")
                hour_label = f"{file_dt.hour:02d}Z"
                if hour_label not in ["00Z", "12Z"]:
                    # Pastikan hanya dua siklus utama
                    hour_label = "00Z" if file_dt.hour < 6 else "12Z"

                # --- Baca isi BUFR ---
                local_path = download_from_ftp(true_site, fname)
                decoded = decode_bufr(local_path)
                df_meta, df_levels = parse_bufr(decoded)
                if df_levels.empty:
                    continue

                # --- Cari waktu ke 100 dan 30 hPa ---
                t100 = df_levels.loc[df_levels["pressure_hPa"] <= 100, "time_s"].min()
                t30  = df_levels.loc[df_levels["pressure_hPa"] <= 30,  "time_s"].min()

                # --- Simpan meskipun cuma 100 hPa ---
                if pd.notna(t100):
                    data.append({
                        "filename": fname,
                        "date": date_str,
                        "hour": hour_label,
                        "AB": round(t100 / 60.0, 1),
                        "CD": round(t30 / 60.0, 1) if pd.notna(t30) else None
                    })

            except Exception as e:
                print(f"⚠️ Error parsing {fname}:", e)

    except Exception as e:
        print("❌ Time accuracy fetch failed:", e)

    # --- Urutkan: tanggal + jam (00Z dulu, baru 12Z) ---
    def sort_key(x):
        return (x["date"], 0 if x["hour"] == "00Z" else 1)

    data = sorted(data, key=sort_key)
    return jsonify({"site": site, "data": data})

@app.route("/height_reach")
@login_required
def height_reach():
    return render_template("height_reach.html")


@app.route("/api/height_reach/<site>")
@login_required
def api_height_reach(site):
    """Ambil data ketinggian maksimum balon (.bfr) per hari untuk satu bulan berjalan."""
    from datetime import datetime, timedelta
    import pandas as pd, re

    now = datetime.utcnow()
    start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_date = (start_date + timedelta(days=32)).replace(day=1)

    def extract_datetime_from_filename(fname):
        """Ambil waktu dari nama file (contoh: T2097502A202510050000.BFR)."""
        m = re.search(r"(\d{10,14})", fname)
        if not m: return None
        s = m.group(1)
        for fmt in ["%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d%H"]:
            try: return datetime.strptime(s, fmt)
            except: pass
        return None

    data = []
    try:
        all_sites = fetch_all_sites(ext_filter=[".bfr"], with_meta=False,
                                    start_date=start_date, end_date=end_date)
        site_keys = {s.lower(): s for s in all_sites.keys()}
        if site.lower() not in site_keys:
            return jsonify({"site": site, "data": []})
        true_site = site_keys[site.lower()]

        for f in all_sites[true_site]:
            fname = f["name"]
            dt = extract_datetime_from_filename(fname)
            if not dt:
                continue
            hour_label = "00Z" if dt.hour < 6 else "12Z"

            try:
                local_path = download_from_ftp(true_site, fname)
                decoded = decode_bufr(local_path)
                _, df_levels = parse_bufr(decoded)
                if df_levels.empty or "height_m" not in df_levels.columns:
                    continue

                max_height = df_levels["height_m"].max()
                if pd.notna(max_height) and max_height > 0:
                    data.append({
                        "filename": fname,
                        "date": dt.strftime("%Y-%m-%d"),
                        "hour": hour_label,
                        "max_height": round(float(max_height), 0)
                    })
            except Exception as e:
                print("⚠️ Skip file:", fname, e)

    except Exception as e:
        print("❌ Height reach fetch failed:", e)

    data = sorted(data, key=lambda x: (x["date"], 0 if x["hour"] == "00Z" else 1))
    return jsonify({"site": site, "data": data})

@app.route("/settings")
@login_required
def settings_page():
    # Hanya admin
    if session.get("user") != "admin":
        return "Access denied. Admin only.", 403
    return render_template("settings.html", users=VALID_USERS)


@app.route("/api/users", methods=["GET", "POST", "DELETE"])
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

        VALID_USERS[username] = password
        save_users()
        print(f"✅ Added new user: {username}")
        return jsonify({"success": True, "users": VALID_USERS})

    # --- DELETE: remove user
    if request.method == "DELETE":
        data = request.get_json()
        username = data.get("username")
        if username not in VALID_USERS:
            return jsonify({"error": "User not found"}), 404
        if username == "admin":
            return jsonify({"error": "Cannot delete admin"}), 400

        del VALID_USERS[username]
        save_users()
        print(f"🗑️ Deleted user: {username}")
        return jsonify({"success": True, "users": VALID_USERS})

@app.route("/raob_doc")
@login_required
def raob_doc():
    return render_template("raob_doc.html")

    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8082,debug=True)
