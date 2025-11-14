# rason_backend/core/utils.py
import os, re, json as _json, base64 as _base64, hashlib as _hashlib
import numpy as np
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from geopy.distance import geodesic

from config.settings import SECRET_KEY, SITES_FILE


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a 32-byte Fernet key from SECRET_KEY string."""
    if not secret:
        raise ValueError("Secret key empty.")
    h = _hashlib.sha256(secret.encode("utf-8")).digest()
    return _base64.urlsafe_b64encode(h)

def _get_fernet():
    """Return global Fernet instance derived from config SECRET_KEY."""
    return Fernet(_derive_fernet_key(SECRET_KEY))

def encrypt_value(val):
    """Encrypt any serializable object to a Fernet token."""
    try:
        f = _get_fernet()
        raw = _json.dumps(val, ensure_ascii=False).encode("utf-8")
        return f.encrypt(raw).decode("utf-8")
    except Exception as e:
        print(f"encrypt_value failed: {e}")
        return val

def decrypt_value(token_str):
    """Decrypt token back to original JSON value."""
    if token_str is None:
        return None
    if not isinstance(token_str, str):
        return token_str
    if not token_str.startswith("gAAAA"):
        try:
            return _json.loads(token_str)
        except Exception:
            return token_str
    try:
        f = _get_fernet()
        raw = f.decrypt(token_str.encode("utf-8"))
        return _json.loads(raw.decode("utf-8"))
    except Exception as e:
        print(f"decrypt_value failed: {e}")
        return token_str

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

def excel_date_to_str(val):
    """Konversi nilai Excel float (misal 45680.0) menjadi YYYY-MM-DD."""
    try:
        if isinstance(val, (int, float)) and val > 30000:
            base = datetime(1899, 12, 30)
            return (base + timedelta(days=int(val))).strftime("%Y-%m-%d")
        if isinstance(val, str):
            return val.strip()
    except Exception:
        pass
    return "-"

def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan

# Ambil tanggal dari nama file (atau fallback ke waktu modifikasi)
def parse_log_date(filename):
    m = re.search(r"(\d{1,2})[ _-]?([A-Za-z]{3})[ _-]?(\d{2})", filename, re.IGNORECASE)
    if m:
        day, mon, yy = m.groups()
        try:
            return datetime.strptime(f"{day} {mon} 20{yy}", "%d %b %Y")
        except Exception:
            pass
    # fallback ke tanggal modifikasi di FTP
    try:
        modified = ftp.sendcmd(f"MDTM {filename}")
        return datetime.strptime(modified[4:], "%Y%m%d%H%M%S")
    except Exception:
        return datetime.min

# --- helper: ambil angka saja, aman untuk "12345", "12345.0", "SN-12345", dll.
def parse_serial_to_int(val):
    if val is None:
        return None
    s = str(val).strip()
    # kalau float "12345.0" → "12345"
    try:
        f = float(s)
        return int(f)
    except:
        pass
    # jika format campur huruf, ambil digitnya
    digits = re.sub(r"\D", "", s)
    return int(digits) if digits.isdigit() else None

REASON_MAP = {
    0:"Not specified",1:"Balloon burst",2:"Battery exhausted",3:"Ascent Stop",
    4:"Telemetry interrupted",5:"Manual termination",6:"Other",11:"Temperature KO",
}

COMBINED_REASON_MAP = {
    "BUR-FRZ-100": {
        "primary": "Balloon Burst",
        "secondary": "RH Freeze-out",
        "level": "<100 hPa",
        "meaning": "Burst terjadi setelah sensor freeze-out, terjadi pada supercooled dry layer."
    },
    "BUR-CLD-100": {
        "primary": "Balloon Burst",
        "secondary": "Cloud/Rain Layer",
        "level": "<100 hPa",
        "meaning": "Balon pecah akibat turbulensi atau drag saat melewati lapisan awan/hujan."
    },
    "BUR-CNB-100": {
        "primary": "Balloon Burst",
        "secondary": "Deep Convection",
        "level": "<100 hPa",
        "meaning": "Balon pecah saat melewati cumulonimbus atau updraft kuat."
    },
    "BUR-SHR-100": {
        "primary": "Balloon Burst",
        "secondary": "Strong Shear",
        "level": "<100 hPa",
        "meaning": "Shear vertikal tinggi menyebabkan balon pecah sebelum mencapai puncak."
    },
    "BUR-DSH-100": {
        "primary": "Balloon Burst",
        "secondary": "Directional Shear",
        "level": "<100 hPa",
        "meaning": "Perubahan arah angin ekstrem memicu stress pada balon."
    },
    "BUR-PRS-UNK": {
        "primary": "Balloon Burst",
        "secondary": "Pressure Reversal",
        "level": "Unknown",
        "meaning": "Pecah terjadi setelah anomali tekanan, kemungkinan sensor error atau turbulensi."
    },

    "ASC-CLD-UNK": {
        "primary": "Ascent Stop",
        "secondary": "Cloud/Rain Layer",
        "level": "N/A",
        "meaning": "Balon tertahan hujan/drag sehingga berhenti naik."
    },
    "ASC-CNB-UNK": {
        "primary": "Ascent Stop",
        "secondary": "Deep Convection",
        "level": "N/A",
        "meaning": "Downburst atau turbulensi cumulonimbus menghentikan kenaikan balon."
    },
    "ASC-FRZ-UNK": {
        "primary": "Ascent Stop",
        "secondary": "RH Freeze-out",
        "level": "N/A",
        "meaning": "Freeze-out menyebabkan data stagnan lalu ascent stop."
    },
    "ASC-ASN-UNK": {
        "primary": "Ascent Stop",
        "secondary": "Slow Ascent",
        "level": "N/A",
        "meaning": "Laju naik melemah bertahap akibat densitas udara, cuaca, atau gas kurang."
    },
    "ASC-SHR-UNK": {
        "primary": "Ascent Stop",
        "secondary": "Strong Shear",
        "level": "N/A",
        "meaning": "Shear layer menghambat kenaikan sehingga balon berhenti."
    },

    "TEL-CLD-UNK": {
        "primary": "Telemetry Interrupted",
        "secondary": "Cloud/Rain Layer",
        "level": "N/A",
        "meaning": "Hujan menyebabkan sinyal lemah/terhalang."
    },
    "TEL-CNB-UNK": {
        "primary": "Telemetry Interrupted",
        "secondary": "Deep Convection",
        "level": "N/A",
        "meaning": "Sinyal hilang saat balon masuk cumulonimbus."
    },
    "TEL-GPS-UNK": {
        "primary": "Telemetry Interrupted",
        "secondary": "GPS Fail",
        "level": "N/A",
        "meaning": "GPS hilang dan sistem tracking gagal mengikuti balon."
    },
    "TEL-SHR-UNK": {
        "primary": "Telemetry Interrupted",
        "secondary": "Strong Shear",
        "level": "N/A",
        "meaning": "Shear tinggi membuat balon drift cepat keluar coverage antena."
    },

    "TMP-FRZ-UNK": {
        "primary": "Temperature KO",
        "secondary": "RH Freeze-out",
        "level": "N/A",
        "meaning": "Sensor T rusak akibat pembentukan es."
    },
    "TMP-SAT-UNK": {
        "primary": "Temperature KO",
        "secondary": "RH Saturation",
        "level": "N/A",
        "meaning": "Sensor T dan RH jenuh sehingga data KO."
    },
    "TMP-CPT-UNK": {
        "primary": "Temperature KO",
        "secondary": "Cold Point Tropopause",
        "level": "N/A",
        "meaning": "Sensor tidak mampu membaca temperatur ekstrem di tropopause."
    },

    "BAT-CPT-UNK": {
        "primary": "Battery Exhausted",
        "secondary": "Cold Point Tropopause",
        "level": "N/A",
        "meaning": "Tegangan drop karena suhu ekstrem di tropopause."
    },
    "BAT-FRZ-UNK": {
        "primary": "Battery Exhausted",
        "secondary": "RH Freeze-out",
        "level": "N/A",
        "meaning": "Es dan kondensasi mempercepat konsumsi power."
    },

    "OTH-CLD-UNK": {
        "primary": "Other",
        "secondary": "Cloud Layer",
        "level": "N/A",
        "meaning": "Gangguan operasi umum terkait lapisan awan."
    },
    "OTH-SHR-UNK": {
        "primary": "Other",
        "secondary": "Strong Shear",
        "level": "N/A",
        "meaning": "Shear ekstrem menyebabkan gangguan aerodinamis."
    },
    "UNK-SHR-UNK": {
        "primary": "Not Specified",
        "secondary": "Strong Shear",
        "level": "N/A",
        "meaning": "Kasus umum tanpa reason eksplisit, namun shear terdeteksi."
    }
}


SENSOR_MAPS = {
    "pressure": {0:"Unknown",1:"Aneroid",2:"Capacitive",3:"Other"},
    "temperature": {0:"Unknown",1:"Thermistor",2:"Platinum",3:"Other"},
    "humidity": {0:"Unknown",1:"Hair",2:"Capacitive",3:"Carbon",4:"Other"},
    "balloon": {0:"Unknown",1:"Latex",2:"Polyethylene",3:"Other"},
    "balloon_gas": {0:"Unknown",1:"Hydrogen",2:"Helium"},
    "balloon_manufacturer": {0:"Unknown",1:"Totex",2:"Kaysam",3:"Other"},
}

# =============================
# 🔧 PRIMARY & SECONDARY NORMALIZER
# =============================

def normalize_primary(reason_raw: str) -> str:
    """Convert '1 – Balloon burst' → 'balloon burst' """
    if not reason_raw:
        return ""
    return reason_raw.split("–")[-1].strip().lower()


def infer_secondary_from_qc(issues_text: str) -> str:
    """Infer secondary category only from QC flight_issues."""
    issues = issues_text.lower()

    if "ascent stop" in issues:
        return "slow ascent"            # maps to ASC-ASN-UNK

    if "not reaching 30" in issues or "not reaching 100" in issues:
        return "strong shear"           # maps to *-SHR-* groups

    if "gps fail" in issues:
        return "gps fail"

    if "temp ko" in issues:
        return "rh freeze-out"

    return ""


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

SITE_LIST = []
# core/utils.py
def load_sites():
    """Return list of sites from sites.json (support alias)."""
    try:
        with open(SITES_FILE, "r", encoding="utf-8") as f:
            data = _json.load(f)
        sites_raw = data.get("sites", data) if isinstance(data, dict) else data
        sites = []
        for s in sites_raw:
            if isinstance(s, str):
                sites.append({
                    "name": s.lower(),
                    "alias": s.title(),  # fallback
                    "lat": 0.0,
                    "lon": 0.0,
                    "utc_offset": 7
                })
            elif isinstance(s, dict):
                sites.append({
                    "name": s.get("name", "").lower(),
                    "alias": s.get("alias", s.get("name", "").title()),
                    "lat": float(s.get("lat", 0)),
                    "lon": float(s.get("lon", 0)),
                    "utc_offset": int(s.get("utc_offset", 7))
                })
        print(f"✅ Loaded {len(sites)} sites (with alias).")
        return sites
    except Exception as e:
        print(f"[WARN] load_sites() failed: {e}")
        return []

def save_sites(sites):
    """Write list of sites into sites.json ({'sites':[...]})"""
    try:
        with open(SITES_FILE, "w", encoding="utf-8") as f:
            _json.dump({"sites": sites}, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(sites)} sites to {SITES_FILE}")
    except Exception as e:
        print(f"[WARN] save_sites() failed: {e}")

def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan

def load_modem_lookup(json_path="list_data_modem.json"):
    """Convert modem list JSON to dict {serial_int: manufactured_str}."""
    from config.settings import BASE_DIR
    import os
    path = os.path.join(BASE_DIR, json_path)
    if not os.path.exists(path):
        print(f"[WARN] Modem list not found: {path}")
        return {}
    data = _json.load(open(path, "r", encoding="utf-8"))
    lookup = {}
    for row in data:
        low = {str(k).strip().lower(): v for k, v in row.items()}
        serial_raw = low.get("nomor serial") or low.get("serial") or low.get("no seri") or low.get("sn")
        manufactured_raw = low.get("manufactured") or low.get("tanggal manufactured") or low.get("tgl manufactured") or low.get("mfg")
        sn_int = parse_serial_to_int(serial_raw)
        manufactured_str = excel_date_to_str(manufactured_raw)
        if sn_int:
            lookup[sn_int] = manufactured_str
    return lookup

MODEM_LOOKUP = load_modem_lookup("list_data_modem.json")
