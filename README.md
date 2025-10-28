RASON MONITORING (RASONWEB)
=====================================

Version: Refactor Branch (v1.1.0)
Author: Terrindo · BMKG Project
Category: Radiosonde Data Visualization & Analysis Platform
Environment: Python 3.12, Flask, Gunicorn, Nginx


📘 DESCRIPTION
-------------------------------------
Rason Monitoring (RasonWeb) adalah platform web berbasis Flask
yang digunakan untuk memonitor, menganalisis, dan memvisualisasikan
data radiosonde (BUFR/BFR/BFH/BIN) dari stasiun BMKG.

Sistem ini melakukan decoding, analisis fisika mendalam (Deep Physics),
dan menampilkan hasilnya dalam bentuk grafik interaktif, tabel, serta peta 2D dan 3D.

Refactor branch ini merupakan versi perombakan total yang memisahkan struktur
backend menjadi modul-modul yang lebih rapi dan scalable.


📁 DIRECTORY STRUCTURE
-------------------------------------
.
├── app.py                     → Flask app entry point
├── wsgi.py                    → Entry point for Gunicorn (production)
├── Procfile                   → Render/Railway deployment config
├── requirements.txt            → Python dependencies
├── README.md / README.txt      → Project documentation
├── __init__.py                 → App initialization marker
│
├── routes/                     → All Flask Blueprints
│   ├── api.py                 → REST API endpoints
│   ├── pages.py               → Web page routes
│   ├── admin.py               → User management routes
│   └── __init__.py
│
├── core/                       → Backend logic modules
│   ├── auth.py                → User authentication / token control
│   ├── bufr_parser.py         → Meteomodem BUFR/BFR/BIN decoder
│   ├── db.py                  → SQLite caching for radiosonde metadata
│   ├── ftp.py                 → FTP downloader / fetcher
│   ├── utils.py               → Helper functions (hashing, formatting, etc.)
│   └── __init__.py
│
├── config/                     → Configuration sets per site
│   ├── bufr_mapping_default.json
│   ├── bufr_mapping_*.json     → Per-site BUFR mapping (Aceh, Ranai, Tarakan, etc.)
│   ├── settings.py             → Global configuration (FTP, paths, etc.)
│   └── __init__.py
│
├── templates/                  → HTML templates (Jinja2)
│   ├── main.html              → Main menu / dashboard
│   ├── data_availability.html → Calendar-based availability page
│   ├── error_analysis.html    → QC/physics analysis viewer
│   ├── height_reach.html      → Maximum height graph page
│   ├── time_accuracy.html     → Time accuracy comparison
│   ├── trajectory3d.html      → 3D flight visualization (deck.gl)
│   ├── settings.html          → User & BUFR config management
│   ├── under_development.html → Placeholder for WIP pages
│   ├── login.html             → Login page
│   └── others...              → RAOB docs, release notes, etc.
│
├── static/                     → Static frontend assets
│   ├── libs/                 → JS/CSS libraries (Chart.js, Leaflet, MapLibre, jQuery)
│   ├── images/               → Logos (TBR2, BMKG, Meteomodem)
│   ├── geojson/              → Map boundaries of Indonesia
│   └── style.css, fonts/, etc
│
├── uploads/                    → Uploaded or cached radiosonde files
│   └── *.bfr / *.bin files
│
├── tools/                      → Utility scripts
│   ├── encrypt_users.py       → Hashing all passwords in users.json
│   ├── pass_generate.py       → Password generator
│   ├── fetch_libs.sh          → Auto-download frontend libraries
│   ├── merged_data_modem.py   → Data merging utility
│   └── app.py.backup          → Previous app version
│
├── users.json                  → User accounts, hashed passwords, expiry, allowed_pages
├── sites.json                  → List of all radiosonde sites
├── bufr_mapping_full.json      → Complete BUFR mapping table
└── list_data_modem.json        → List of known data sources


⚙️ INSTALLATION
-------------------------------------
1. Clone the repository:
   git clone https://github.com/nandamiftahul/rasonweb.git
   cd rasonweb

2. Create virtual environment:
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

3. Run development server:
   python app.py
   (or)
   python -c "from app import create_app; app=create_app(); app.run(host='0.0.0.0', port=8082)"

4. Production (Gunicorn):
   gunicorn -w 2 -b 0.0.0.0:8080 wsgi:app

   Example systemd service:
   -------------------------------------
   [Unit]
   Description=Gunicorn instance for Rason Web
   After=network.target

   [Service]
   User=tbr
   Group=tbr
   WorkingDirectory=/home/tbr/app_core/rason_web
   Environment="PATH=/home/tbr/venv/py312_core/bin"
   ExecStart=/home/tbr/venv/py312_core/bin/gunicorn \
             --workers 1 \
             --threads 2 \
             --bind unix:/home/tbr/app_core/rason_web/run/rason_web_gunicorn.sock \
             wsgi:app
   Restart=always

   [Install]
   WantedBy=multi-user.target


🧭 FEATURES
-------------------------------------
- Decode BUFR/BFR/BFH/BIN files (Meteomodem M10/M20)
- Store decoded metadata to SQLite cache
- Deep physics analysis: cold point tropopause, lapse rate, freeze-out RH, pressure anomaly,
  ascent deceleration, wind shear, directional change, burst detection, etc.
- Chart.js graphs: T/RH/Wind vs Height
- MapLibre & Deck.gl: 3D trajectory visualization
- Leaflet-based site maps (2D)
- User management with hashed passwords, expiry control, and per-page access restriction
- Admin force logout and single-session protection
- Time Accuracy & Data Availability API endpoints
- Responsive UI with collapsible sidebar, footer, and unified gradient theme
- Automatic FTP data retrieval and caching


🔐 USER MANAGEMENT
-------------------------------------
users.json structure:
[
  {
    "username": "admin",
    "password": "<hashed>",
    "expiry": "2026-10-01",
    "pages": ["*"]
  },
  {
    "username": "guest",
    "password": "<hashed>",
    "expiry": "2025-12-31",
    "pages": ["main", "map", "data_availability"]
  }
]

Features:
- Hashing handled by `core/auth.py`
- Encryption script: `tools/encrypt_users.py`
- Expiry check on login (auto-block expired)
- Single active session enforcement
- Admin override access


🌐 FRONTEND OVERVIEW
-------------------------------------
- Gradient Theme: #0a2a6b → #004aad → #0078d7
- Font: Inter
- Sidebar: collapsible with localStorage persistence
- Header box: adaptive + digital clock
- Charts: Chart.js v4 + plugins (zoom, datalabels)
- Tables: DataTables.js with responsive layout
- Maps: Leaflet / MapLibre + GeoJSON overlays (Indonesia provinces)
- Footer: “Powered by Terrindo · BMKG · Meteomodem”


🧪 APIs SUMMARY
-------------------------------------
/api/time_accuracy/<site>
  → Returns AA/BB/CC/DD time deviation dataset
/api/data_availability
  → Returns daily 00Z/12Z availability matrix
/upload_bufr
  → Decode BUFR/BFR/BIN and return metadata & level data


🧱 BACKEND MODULES
-------------------------------------
core/auth.py         → Token + password management
core/db.py           → SQLite caching layer
core/ftp.py          → FTP file fetching
core/bufr_parser.py  → Main BUFR decoder (pybufrkit / pyart)
core/utils.py        → Utility helpers

routes/api.py        → JSON API endpoints
routes/pages.py      → Web routes
routes/admin.py      → Admin-only routes


🚀 DEPLOYMENT (Render / Railway)
-------------------------------------
Procfile:
web: gunicorn -w 2 -b 0.0.0.0:8080 wsgi:app

Environment variables:
FLASK_ENV=production
SECRET_KEY=<your_secret_key>
UPLOAD_FOLDER=./uploads

Deploy by connecting GitHub repo and using this branch (refactor).


🧾 CHANGELOG (v1.1.0 Refactor)
-------------------------------------
• Backend refactored into modular packages (core/, routes/, config/)
• Separated BUFR mapping per site (Aceh, Ranai, Tarakan, etc.)
• Added error handling in FTP and BUFR parsing
• Reorganized templates & static libs
• Improved login security and single-session enforcement
• Updated Time Accuracy and Height Reach pages with Chart.js v4
• Optimized Gunicorn worker/thread model
• Added under_development.html placeholder for new features
• UI enhancements with responsive sidebar & header

---

📄 LICENSE
-------------------------------------
Terrindo © 2025


💬 CONTACT
-------------------------------------
PT. Terrindo Media Raya · BMKG Project Collaboration
For internal use within BMKG & Terrindo technical team.

