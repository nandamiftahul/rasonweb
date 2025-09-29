#! /usr/bin/python3
import os, re
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
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # safe for server
import matplotlib.pyplot as plt
from io import BytesIO
import base64

import metpy.calc as mpcalc
from metpy.units import units
from metpy.plots import SkewT, Hodograph
from scipy.signal import medfilt


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
            return redirect(url_for("dashboard"))
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
    """Fetch BUFR from FTP and save into THIS user's store."""
    cfg = CONFIG["ftp"]
    local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(f"{cfg['base_path']}/{site}")
            with open(local_path, "wb") as f:
                ftp.retrbinary(f"RETR " + filename, f.write)

        decoded = decode_bufr(local_path)
        df_meta, df_levels = parse_bufr(decoded)
        issues = analyze_flight(df_meta, df_levels)

        store = get_user_store()
        store["metadata"] = df_meta.to_dict("records")[0] if not df_meta.empty else {}
        store["metadata"]["flight_issues"] = issues
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

        issues = analyze_flight(df_meta, df_levels)
        store = get_user_store()
        store["metadata"] = df_meta.to_dict("records")[0] if not df_meta.empty else {}
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
            ws = wind["wind_speed_mps"].values * (units.meter/units.second)
            wdir = wind["wind_dir_deg"].values * units.degree
            u, v = mpcalc.wind_components(ws, wdir)

            # Heights: use BUFR height if present, otherwise estimate
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
                rmotion, lmotion, mean_wind = mpcalc.bunkers_storm_motion(p_w, u, v, hgt)
                storm_u, storm_v = rmotion
                srh_0_1km, _, _ = mpcalc.storm_relative_helicity(
                    hgt, u, v, depth=1000*units.m,
                    storm_u=storm_u, storm_v=storm_v
                )
                srh_0_3km, _, _ = mpcalc.storm_relative_helicity(
                    hgt, u, v, depth=3000*units.m,
                    storm_u=storm_u, storm_v=storm_v
                )
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

        # --- Tropopause (WMO: lapse rate < 2°C/km) ---
        tropopause_level = "-"
        try:
            if "height_m" in thermo:
                lapse_rate = mpcalc.lapse_rate(
                    thermo["temp_C"].values * units.degC,
                    thermo["height_m"].values * units.meter
                ).to("degC/km")
                bad = np.where(lapse_rate > -2 * units("degC/km"))[0]
                if bad.size > 0:
                    tropopause_level = f"{thermo.iloc[bad[0]]['pressure_hPa']:.0f} hPa"
        except Exception:
            pass

        # --- Skew-T plot ---
        fig1 = plt.figure(figsize=(6, 6))
        skew = SkewT(fig1, rotation=45)
        skew.plot(p, T, "r")
        skew.plot(p, Td, "g")
        if (u is not None) and (v is not None):
            skew.plot_barbs(p_w, u.to("m/s"), v.to("m/s"))
        skew.plot(p, parcel_prof, "k")
        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-40, 40)
        skew.ax.set_title(f"Skew-T  {df_meta.iloc[0].get('launch_time', '')}")
        buf1 = BytesIO()
        plt.savefig(buf1, format="png", bbox_inches="tight")
        buf1.seek(0)
        skewt_img = base64.b64encode(buf1.read()).decode("utf-8")
        plt.close(fig1)

        # --- Hodograph ---
        if (u is not None) and (v is not None):
            fig2 = plt.figure(figsize=(6, 6))
            ax = fig2.add_subplot(1, 1, 1)
            hodo = Hodograph(ax, component_range=60.0)
            hodo.add_grid(increment=20)
            hodo.plot(u.to("m/s"), v.to("m/s"), color="b")
            ax.set_title("Hodograph")
            buf2 = BytesIO()
            plt.savefig(buf2, format="png", bbox_inches="tight")
            buf2.seek(0)
            hodo_img = base64.b64encode(buf2.read()).decode("utf-8")
            plt.close(fig2)
        else:
            hodo_img = None

        # --- Helper for clean indices ---
        def scalar_str(x, fmt=".1f"):
            try:
                val = np.atleast_1d(x.m)[0]
                return format(val, fmt) if np.isfinite(val) else "-"
            except Exception:
                return "-"

        # --- Indices table ---
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

        return render_template(
            "raob.html",
            meta=df_meta.to_dict("records")[0],
            indices=indices,
            skewt_img=skewt_img,
            hodo_img=hodo_img
        )

    except Exception as e:
        return f"RAOB error: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
