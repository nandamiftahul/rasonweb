RASON MONITORING BACKEND
========================

Author     : Terrindo · BMKG Project
Build Date : October 2025
Framework  : Flask (Python 3.12)
Database   : SQLite (rason_data.db)
Deployment : Gunicorn / Nginx / Railway / Render

--------------------------------------------------
📁 PROJECT STRUCTURE
--------------------------------------------------

rason_backend/
│
├── app.py                # Entry point (menjalankan Flask)
│
├── config/
│   ├── __init__.py
│   └── settings.py       # Konfigurasi global, load .env, FTP, secret key, paths
│
├── core/
│   ├── __init__.py
│   ├── db.py             # Semua fungsi database SQLite
│   ├── utils.py          # Fungsi umum: enkripsi, analisis, parser, helper
│   ├── ftp.py            # Semua operasi FTP (fetch_all_sites, download, dll)
│   ├── bufr_parser.py    # Decode & parsing BUFR/BFR/BIN files
│   └── auth.py           # Login/logout, session management
│
├── routes/
│   ├── __init__.py
│   ├── api.py            # Semua @app.route("/api/...")
│   ├── pages.py          # Semua halaman HTML (render_template)
│   └── admin.py          # Endpoint admin-only (user mgmt, logout all, dll)
│
├── static/
│   └── ...               # CSS, JS, images
│
├── templates/
│   └── ...               # HTML pages
│
├── sites.json
├── users.json
├── bufr_mapping_full.json
├── list_data_modem.json
└── requirements.txt


--------------------------------------------------
⚙️ REQUIREMENTS
--------------------------------------------------
Python >= 3.10
Flask >= 3.0
pandas >= 2.0
numpy >= 1.25
ftplib (builtin)
metpy
geopy
cryptography
python-dotenv
matplotlib
scipy
werkzeug

Install all dependencies:
> pip install -r requirements.txt

--------------------------------------------------
🚀 RUNNING THE APP
--------------------------------------------------
1️⃣ Activate virtual environment:
   > source venv/bin/activate

2️⃣ Run Flask (development mode):
   > python app.py

3️⃣ Or run via Gunicorn (production mode):
   > gunicorn -w 3 -b 0.0.0.0:8082 app:app

Then open your browser:
   http://localhost:8082

--------------------------------------------------
🔐 LOGIN & SECURITY
--------------------------------------------------
- User credentials stored in `users.json`
- Passwords hashed via Werkzeug
- Session timeout: 30 minutes
- Encrypted expiry & page access stored using Fernet
- Only admin can modify or delete users

--------------------------------------------------
🗄️ DATABASE (SQLite)
--------------------------------------------------
File: rason_data.db
Tables:
- bufr / bfr / bfh / bin
  (id, site, filename, filetype, file_date, meta_json, levels_json, created_at)

--------------------------------------------------
🌐 FTP CONNECTION
--------------------------------------------------
All FTP credentials and base path loaded from `.env`:
Example:
    FTP_HOST=192.168.1.100
    FTP_USER=anonymous
    FTP_PASS=
    FTP_BASE_PATH=/UA
    FTP_LIMIT=30
    SECRETKEY=my_super_secret_key

--------------------------------------------------
🧠 MAIN FEATURES
--------------------------------------------------
- User login / subscription expiry
- Radiosonde data decoding (BUFR, BFR, BFH, BIN)
- FTP data caching (SQLite)
- Time Accuracy Analysis
- Height Reach Analysis
- RAOB & Skew-T Plot
- 3D Trajectory Viewer (Deck.gl)
- Error Analysis (Physics-based QC)
- Data Availability calendar
- Settings (User & BUFR mapping)
- Admin Controls (Logout all, update expiry, allowed pages)

--------------------------------------------------
📦 DEPLOYMENT NOTES
--------------------------------------------------
For Gunicorn + Nginx:
- Gunicorn service runs app:app
- Static files served by Nginx under `/static`
- Ensure uploads/ writable by web user
- Environment variables loaded from `.env`

--------------------------------------------------
🧩 VERSIONING
--------------------------------------------------
v1.0.0 — Initial production build  
v1.0.1 — Login/session/expiry upgrade  
v1.0.2 — Database/FTP comparison + settings refactor  
v1.0.3 — (planned) Geo-tracking login, global session map

--------------------------------------------------
🧑‍💻 DEVELOPED BY
Terrindo · BMKG · Meteomodem Collaboration
Powered by Flask · Chart.js · Leaflet · Deck.gl · MetPy

--------------------------------------------------
📄 LICENSE
--------------------------------------------------
This software is proprietary and intended for internal BMKG use only.
Redistribution or commercial use without permission is prohibited.
