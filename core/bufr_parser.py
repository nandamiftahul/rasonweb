# rason_backend/core/bufr_parser.py
import subprocess, re, json, os
import pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta
from metpy.units import units
from config.settings import BASE_DIR, BUFRCFG_FULL
from .utils import REASON_MAP, SENSOR_MAPS

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

# --- Parse BUFR (dynamic + site-specific + fallback) ---
def parse_bufr(decoded_text, site="default"):
    import os, json, re
    import pandas as pd, numpy as np
    from datetime import datetime, timezone

    meta = {}
    levels = []
    current = {}
    station_lat, station_lon = None, None

    # --- Helper functions ---
    def _extract_bytes_field(line: str) -> str:
        """Extract b'..' or b"..". Returns stripped string, else last token."""
        m = re.search(r"b[\"'](.*?)[\"']", line)
        if m:
            return m.group(1).strip()
        return line.split()[-1]

    def _safe_float_tail(line: str):
        try:
            return float(line.split()[-1])
        except Exception:
            return None

    # ==================================================================
    # 🔹 1. Load site-specific mapping (priority)
    # ==================================================================
    CONFIG_DIR = "config"
    site_key = str(site).lower().strip().replace(" ", "_")
    site_file = os.path.join(CONFIG_DIR, f"bufr_mapping_{site_key}.json")
    default_file = "bufr_mapping_full.json"

    mapping = None

    # Try site-specific mapping first
    if os.path.exists(site_file):
        try:
            with open(site_file) as f:
                mapping = json.load(f)
            print(f"🗺️ Using site mapping: {site_file}")
        except Exception as e:
            print(f"⚠️ Failed to read mapping for site '{site_key}': {e}")

    # Fallback to global bufr_mapping_full.json
    if not mapping and os.path.exists(default_file):
        try:
            with open(default_file) as f:
                mapping = json.load(f)
            print(f"ℹ️ Using global mapping: {default_file}")
        except Exception:
            mapping = None

    # ==================================================================
    # 🔹 2. Default fallback mapping (original unchanged)
    # ==================================================================
    default_mapping = {
        "meta": [
            {"original": "WMO BLOCK NUMBER", "variable": "wmo_block"},
            {"original": "WMO STATION NUMBER", "variable": "wmo_station"},
            {"original": "004001 YEAR", "variable": "year"},
            {"original": "004002 MONTH", "variable": "month"},
            {"original": "004003 DAY", "variable": "day"},
            {"original": "004004 HOUR", "variable": "hour"},
            {"original": "004005 MINUTE", "variable": "minute"},
            {"original": "004006 SECOND", "variable": "second"},
            {"original": "LATITUDE (HIGH ACCURACY)", "variable": "station_lat"},
            {"original": "LONGITUDE (HIGH ACCURACY)", "variable": "station_lon"},
            {"original": "HEIGHT OF STATION GROUND", "variable": "station_height_m"},
            {"original": "RADIOSONDE SERIAL NUMBER", "variable": "radiosonde_serial_number"},
            {"original": "RADIOSONDE ASCENSION NUMBER", "variable": "radiosonde_ascension_number"},
            {"original": "RADIOSONDE RELEASE NUMBER", "variable": "radiosonde_release_number"},
            {"original": "RADIOSONDE GROUND RECEIVING SYSTEM", "variable": "radiosonde_ground_rx_system"},
            {"original": "RADIOSONDE OPERATING FREQUENCY", "variable": "radiosonde_operating_frequency"},
            {"original": "BALLOON MANUFACTURER", "variable": "balloon_manufacturer"},
            {"original": "WEIGHT OF BALLOON", "variable": "balloon_weight_kg"},
            {"original": "TYPE OF GAS USED IN BALLOON", "variable": "balloon_gas_type"},
            {"original": "TYPE OF PRESSURE SENSOR", "variable": "pressure_sensor_type"},
            {"original": "TYPE OF TEMPERATURE SENSOR", "variable": "temperature_sensor_type"},
            {"original": "TYPE OF HUMIDITY SENSOR", "variable": "humidity_sensor_type"},
            {"original": "SOFTWARE IDENTIFICATION AND VERSION NUMBER", "variable": "software_version"},
            {"original": "REASON FOR TERMINATION", "variable": "reason_for_termination"},
            {"original": "TRACKING TECHNIQUE/STATUS OF SYSTEM USED", "variable": "system_status"}
        ],
        "level": [
            {"original": "PRESSURE", "variable": "pressure_hPa"},
            {"original": "GEOPOTENTIAL HEIGHT", "variable": "height_m"},
            {"original": "TEMPERATURE/AIR TEMPERATURE", "variable": "temp_C"},
            {"original": "DEW-POINT TEMPERATURE", "variable": "dewpoint_C"},
            {"original": "WIND DIRECTION", "variable": "wind_dir_deg"},
            {"original": "WIND SPEED", "variable": "wind_speed_mps"},
            {"original": "LATITUDE DISPLACEMENT", "variable": "lat_disp"},
            {"original": "LONGITUDE DISPLACEMENT", "variable": "lon_disp"},
            {"original": "LONG TIME PERIOD OR DISPLACEMENT", "variable": "time_s"},
            {"original": "EXTENDED VERTICAL SOUNDING SIGNIFICANCE", "variable": "status_flag"}
        ]
    }

    # ==================================================================
    # 🔹 3. Choose active mapping (same as before)
    # ==================================================================
    active_map = mapping if (mapping and "meta" in mapping and "level" in mapping) else default_mapping
    #print(active_map["meta"])
    # ==================================================================
    # 🔹 4. Original parsing logic (unchanged)
    # ==================================================================
    for line in decoded_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # ========== METADATA ==========
        matched_meta = False
        for item in active_map["meta"]:
            if item["original"] in line:
                key = item["variable"]
                val = _safe_float_tail(line)

                if "LATITUDE" in item["original"] and "005001" in line:
                    try:
                        station_lat = float(line.split()[-1])
                        meta[key] = station_lat
                    except Exception:
                        pass
                    matched_meta = True
                    break

                elif "LONGITUDE" in item["original"] and "006001" in line:
                    try:
                        station_lon = float(line.split()[-1])
                        meta[key] = station_lon
                    except Exception:
                        pass
                    matched_meta = True
                    break

                elif "SERIAL" in item["original"] or "VERSION" in item["original"]:
                    meta[key] = _extract_bytes_field(line)
                    matched_meta = True
                    break

                elif val is not None:
                    meta[key] = val
                    matched_meta = True
                    break

        if matched_meta:
            continue

        # --- Level separator ---
        if line.startswith("# ---") and current:
            levels.append(current)
            current = {}
            continue

        # ========== PER-LEVEL DATA ==========
        matched_level = False
        for lv in active_map["level"]:
            if lv["original"] in line:
                key = lv["variable"]
                val = _safe_float_tail(line)
                #print(key,val)
                if "TEMPERATURE" in lv["original"]:
                    try: val = val - 273.15
                    except Exception: pass
                elif "DEW-POINT" in lv["original"]:
                    try: val = val - 273.15
                    except Exception: pass
                elif key == "pressure_hPa" and val is not None:
                    val = val / 100.0

                current[key] = val
                matched_level = True
                break
   
        if matched_level:
            continue
    #print(meta) 
    #print(current)    
    if current:
        levels.append(current)

    df_meta = pd.DataFrame([meta])
    df_levels = pd.DataFrame(levels).replace({None: np.nan})

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

    if all(k in meta for k in ("year", "month", "day", "hour", "minute", "second")):
        try:
            dt_utc = datetime(
                int(meta["year"]), int(meta["month"]), int(meta["day"]),
                int(meta["hour"]), int(meta["minute"]), int(meta["second"]),
                tzinfo=timezone.utc
            )
            meta["launch_time"] = dt_utc.isoformat().replace("+00:00", "Z")
        except Exception:
            meta["launch_time"] = "-"
        for k in ("year", "month", "day", "hour", "minute", "second"):
            meta.pop(k, None)

    return pd.DataFrame([meta]), df_levels

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
