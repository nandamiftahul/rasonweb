# rason_backend/core/ftp.py
import os, re, io, ftplib
from datetime import datetime, timedelta, timezone
import pandas as pd
from geopy.distance import geodesic

from config.settings import CONFIG, UPLOAD_FOLDER
from .bufr_parser import decode_bufr, parse_bufr
from .db import db_get, db_insert
from .utils import extract_date_from_filename, REASON_MAP, SENSOR_MAPS, load_sites, analyze_flight

def download_from_ftp(site, filename):
    """Fetch file from FTP and return local path only (no processing)."""
    cfg = CONFIG["ftp"]
    local_path = os.path.join(UPLOAD_FOLDER, filename)
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

def fetch_all_sites(ext_filter=None, limit=None, with_meta=False,
                    start_date=None, end_date=None):
    """
    Fetch list of radiosonde files from FTP or (if cached) from local SQLite DB.
    Untuk file hari ini dan kemarin, data selalu diambil langsung dari FTP (bypass cache).
    """
    cfg = CONFIG["ftp"]
    _tz = None  # fallback (pakai UTC kalau zoneinfo tidak ada)
    #print("=========================================>",start_date,end_date)
    # --- Default window: today 00:00 UTC -> tomorrow 00:00 UTC ---
    if start_date is None or end_date is None:
        now_local = datetime.now(_tz) if _tz else datetime.utcnow().replace(tzinfo=timezone.utc)
        today_local_midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow_local_midnight = today_local_midnight + timedelta(days=1)

        def to_utc_naive(dt_local):
            if dt_local.tzinfo is None:
                return dt_local
            return dt_local.astimezone(timezone.utc).replace(tzinfo=None)

        if start_date is None:
            start_date = to_utc_naive(today_local_midnight)
        if end_date is None:
            end_date = to_utc_naive(tomorrow_local_midnight)

    result = {}
    
    exts = [e.lower() for e in (ext_filter or cfg.get("file_ext", [".bufr"]))]

    try:
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(cfg["base_path"])
            sites_cfg = load_sites()
            cfg_sites = load_sites()
            ftp_sites = ftp.nlst()
            sites = [s["name"] for s in cfg_sites if s["name"] in ftp_sites]
            
            print(f"[INFO] Using {len(sites)} sites from sites.json: {sites}")
        
            for site in sites:
                site_files = []
                try:
                    ftp.cwd(f"{cfg['base_path']}/{site}")
                    all_files = ftp.nlst()
                    print(f"[DEBUG] {site}: total files returned by FTP = {len(all_files)}")

                    selected = [f for f in all_files if any(f.lower().endswith(ext) for ext in exts)]
                    items = []

                    for fname in selected:
                        dt_str = extract_date_from_filename(fname)
                        try:
                            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S UTC")
                            if start_date or end_date:
                                if start_date and dt < start_date:
                                    continue
                                if end_date and dt >= end_date:
                                    continue
                        except Exception:
                            dt = datetime.min
                        items.append((fname, dt))

                    # Sort newest first
                    items.sort(key=lambda x: x[1], reverse=True)
                    if limit:
                        items = items[:limit]
                    selected = [fname for fname, _ in items]

                    for fname in selected:
                        item = {
                            "name": fname,
                            "site": site,
                            "file_date": extract_date_from_filename(fname)
                        }

                        # ==========================================================
                        # 🔹 Jika with_meta=True, gunakan DB cache (kecuali hari ini & kemarin)
                        # ==========================================================
                        if with_meta:
                            try:
                                ftype = fname.split(".")[-1].lower()
                                if ftype not in ["bufr", "bfr", "bfh", "bin"]:
                                    ftype = "bufr"

                                # --- Tentukan apakah file hari ini atau kemarin ---
                                use_cache = True
                                try:
                                    file_dt = datetime.strptime(extract_date_from_filename(fname), "%Y-%m-%d %H:%M:%S UTC")
                                    now_utc = datetime.utcnow()
                                    yesterday_utc = now_utc - timedelta(days=1)
                                    if file_dt.date() in (now_utc.date(), yesterday_utc.date()):
                                        use_cache = False
                                except Exception:
                                    pass

                                # --- Cek di database hanya jika bukan hari ini/kemarin ---
                                cached = db_get(ftype, site, fname) if use_cache else None
                                if cached:
                                    df_meta, df_levels = cached
                                    print(f"[DB] ✅ fetch_all cache hit for {fname}")
                                else:
                                    # --- Ambil baru dari FTP ---
                                    local_path = os.path.join(UPLOAD_FOLDER, fname)
                                    with open(local_path, "wb") as f:
                                        ftp.retrbinary(f"RETR " + fname, f.write)

                                    decoded = decode_bufr(local_path)
                                    df_meta, df_levels = parse_bufr(decoded, site=site)

                                    db_insert(ftype, site, fname,
                                              extract_date_from_filename(fname),
                                              df_meta, df_levels)
                                    print(f"[DB] 💾 cached {fname}")

                                # --- Ambil metadata dari DataFrame ---
                                if not df_meta.empty:
                                    meta_row = df_meta.iloc[0]

                                    # ✅ Launch time (auto-handle any format)
                                    raw_lt = meta_row.get("launch_time")
                                    lt = None
                                    if raw_lt is not None and not pd.isna(raw_lt):
                                        try:
                                            if isinstance(raw_lt, (datetime, pd.Timestamp)):
                                                lt = raw_lt.astimezone(timezone.utc) if raw_lt.tzinfo else raw_lt.replace(tzinfo=timezone.utc)
                                            elif isinstance(raw_lt, (int, float)):
                                                lt = pd.to_datetime(raw_lt, utc=True, errors="coerce")
                                            elif isinstance(raw_lt, str):
                                                clean_lt = raw_lt.strip().replace("Z", "").replace(" UTC", "")
                                                lt = pd.to_datetime(clean_lt, utc=True, errors="coerce")
                                        except Exception as e:
                                            print(f"[WARN] launch_time parse failed for {fname}: {e}")

                                    if lt is not None and pd.notna(lt):
                                        item["launch_time"] = lt.strftime("%Y-%m-%d %H:%M:%S UTC")
                                    else:
                                        item["launch_time"] = "-"

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

                                # --- Flight issues ---
                                issues = analyze_flight(df_meta, df_levels)
                                if issues:
                                    item["flight_issues"] = issues

                                # --- Derived flight metadata ---
                                try:
                                    if not df_levels.empty:
                                        # End pressure
                                        end_pressure = df_levels["pressure_hPa"].dropna().min()
                                        if pd.notna(end_pressure):
                                            item["end_pressure"] = round(float(end_pressure), 1)

                                        # Max height
                                        max_height = df_levels["height_m"].dropna().max()
                                        if pd.notna(max_height):
                                            item["max_height"] = round(float(max_height), 0)

                                        # End time
                                        if "time_s" in df_levels and not df_meta.empty:
                                            if "launch_time" in item and item["launch_time"] != "-":
                                                try:
                                                    lt = pd.to_datetime(item["launch_time"].replace(" UTC", ""), utc=True)
                                                    end_time = lt + pd.to_timedelta(df_levels["time_s"].max(), unit="s")
                                                    item["end_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                                                except Exception as e:
                                                    print(f"[WARN] end_time calc failed for {fname}: {e}")

                                        # Distance from station
                                        if {"latitude", "longitude"} <= set(df_levels.columns) and not df_meta.empty:
                                            last_lat = df_levels["latitude"].dropna().iloc[-1]
                                            last_lon = df_levels["longitude"].dropna().iloc[-1]
                                            st_lat = meta_row.get("station_lat")
                                            st_lon = meta_row.get("station_lon")
                                            if st_lat and st_lon and pd.notna(last_lat) and pd.notna(last_lon):
                                                dist = geodesic((st_lat, st_lon), (last_lat, last_lon)).km
                                                item["end_distance"] = round(float(dist), 1)

                                        # Avg ascent rate
                                        if "height_m" in df_levels and "time_s" in df_levels:
                                            elapsed = df_levels["time_s"].max() - df_levels["time_s"].min()
                                            if elapsed > 0 and pd.notna(max_height):
                                                ascent_rate = max_height / elapsed
                                                item["avg_ascent_rate"] = round(float(ascent_rate), 2)
                                except Exception as e:
                                    print("Extra metadata calc failed in fetch_all_sites:", e)

                            except Exception as e:
                                item["launch_time"] = f"Error: {e}"
                                item["flight_issues"] = [f"Error: {e}"]

                        # ==========================================================
                        # ✅ Append hasil per file
                        # ==========================================================
                        site_files.append(item)

                    ftp.cwd(cfg["base_path"])
                    result[site] = site_files

                except Exception as e:
                    result[site] = [{"name": f"Error: {e}", "site": site, "file_date": "-"}]

    except Exception as e:
        result["GLOBAL"] = [{"name": f"FTP Error: {e}", "site": "GLOBAL", "file_date": "-"}]

    return result

def combine_t_p_files(site, base_name):
    """
    Cari file T*.X dan P*.X dari FTP dengan timestamp sama (contoh: 2025100900).
    Gabungkan konten keduanya jadi satu TXT file sementara untuk proses WMO.
    """
    cfg = CONFIG["ftp"]
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    combined_path = os.path.join(UPLOAD_FOLDER, f"{base_name}_TP.txt")

    pattern = re.search(r"(\d{10})", base_name)
    if not pattern:
        raise ValueError("No valid timestamp found in filename")
    timestamp = pattern.group(1)

    try:
        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(f"{cfg['base_path']}/{site}")
            all_files = ftp.nlst()

            # Cari file T dan P yang cocok
            t_files = [f for f in all_files if re.search(fr"T\d+.*{timestamp}.*\.[xX]$", f)]
            p_files = [f for f in all_files if re.search(fr"P\d+.*{timestamp}.*\.[xX]$", f)]

            if not t_files or not p_files:
                raise FileNotFoundError(f"T or P file not found for timestamp {timestamp}")

            t_file = t_files[0]
            p_file = p_files[0]

            # Unduh dua file
            t_data = io.BytesIO()
            ftp.retrbinary(f"RETR {t_file}", t_data.write)
            p_data = io.BytesIO()
            ftp.retrbinary(f"RETR {p_file}", p_data.write)

            # Gabungkan kontennya
            with open(combined_path, "wb") as out:
                out.write(t_data.getvalue())
                out.write(b"\n")
                out.write(p_data.getvalue())

            print(f"✅ Combined {t_file} + {p_file} → {combined_path}")
            return combined_path

    except Exception as e:
        print("❌ combine_t_p_files failed:", e)
        raise

def download_and_process(site, filename):
    """
    Fetch BUFR/BFR/BFH/BIN data either from local SQLite DB (if cached)
    or from FTP (then decode via pybufrkit and save into DB).
    Return results into current user's session store.
    """
    from core.utils import analyze_flight, extract_date_from_filename, REASON_MAP, SENSOR_MAPS
    from core.auth import get_user_store
    import pandas as pd
    from geopy.distance import geodesic

    ftype = filename.split(".")[-1].lower()
    if ftype not in ["bufr", "bfr", "bfh", "bin"]:
        ftype = "bufr"

    cfg = CONFIG["ftp"]
    local_path = os.path.join(UPLOAD_FOLDER, filename)
    store = get_user_store()

    try:
        cached = db_get(ftype, site, filename)
        if cached:
            print(f"[DB] ✅ Loaded {filename} from cache ({ftype})")
            df_meta, df_levels = cached
            issues = analyze_flight(df_meta, df_levels)
            store["metadata"] = df_meta.to_dict("records")[0] if not df_meta.empty else {}
            store["metadata"]["flight_issues"] = issues
            store["levels"] = df_levels.to_dict("records") if not df_levels.empty else []
            return

        with ftplib.FTP() as ftp:
            ftp.connect(cfg["host"], cfg.get("port", 21))
            ftp.login(cfg["user"], cfg["password"])
            ftp.cwd(f"{cfg['base_path']}/{site}")
            with open(local_path, "wb") as f:
                ftp.retrbinary(f"RETR " + filename, f.write)
        print(f"[FTP] ✅ Downloaded {filename}")

        decoded = decode_bufr(local_path)
        df_meta, df_levels = parse_bufr(decoded, site=site)
        issues = analyze_flight(df_meta, df_levels)
        db_insert(ftype, site, filename, extract_date_from_filename(filename), df_meta, df_levels)
        print(f"[DB] 💾 Cached {filename}")

        meta = df_meta.to_dict("records")[0] if not df_meta.empty else {}
        meta["flight_issues"] = issues
        store["metadata"] = meta
        store["levels"] = df_levels.to_dict("records") if not df_levels.empty else []
    except Exception as e:
        print(f"[ERROR] download_and_process failed for {filename}: {e}")
        store["metadata"], store["levels"] = {}, []
