# rason_backend/core/db.py
import io
import sqlite3
import pandas as pd
from . import utils
from config.settings import DB_PATH

def db_init():
    """Initialize 4 main tables for BUFR/BFR/BFH/BIN caching."""
    with sqlite3.connect(DB_PATH) as conn:
        for t in ["bufr", "bfr", "bfh", "bin"]:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site TEXT,
                    filename TEXT UNIQUE,
                    filetype TEXT,
                    file_date TEXT,
                    meta_json TEXT,
                    levels_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

def db_get(filetype, site, filename):
    """Return (df_meta, df_levels) if exists in DB; else None."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"SELECT meta_json, levels_json FROM {filetype} WHERE site=? AND filename=?",
            (site, filename)
        ).fetchone()
    if not row:
        return None
    try:
        meta_df = pd.read_json(io.StringIO(row[0]))
        levels_df = pd.read_json(io.StringIO(row[1]))
        return meta_df, levels_df
    except Exception as e:
        print(f"[DB] decode error {filename}: {e}")
        return None

def db_insert(filetype, site, filename, file_date, df_meta, df_levels):
    """Insert parsed BUFR into DB."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(f"""
                INSERT OR REPLACE INTO {filetype}
                (site, filename, filetype, file_date, meta_json, levels_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                site,
                filename,
                filetype,
                file_date,
                df_meta.to_json(),
                df_levels.to_json()
            ))
            conn.commit()
        print(f"[DB] ✅ inserted {filename}")
    except Exception as e:
        print(f"[DB] insert error {filename}: {e}")
