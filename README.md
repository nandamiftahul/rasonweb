# 🎈 RASON MONITORING WEB
A **Flask-based web application** for decoding, caching, and visualizing **Radiosonde (Upper Air Sounding)** data from multiple observation sites across Indonesia.  
Developed by **Terrindo x BMKG** — this system integrates automated FTP data ingestion, SQLite caching, and interactive web visualization for radiosonde operations.

---

## 🧩 Overview
**Rason Monitoring** (Radiosonde Monitoring Web) provides a centralized interface to:
- Monitor radiosonde data collection in near real-time.
- Analyze key performance metrics (launch time accuracy, balloon burst height, data completeness).
- Visualize trajectories, metadata, and derived weather parameters.
- Support operational and research teams for daily upper-air soundings.

---

## ✨ Core Features
### 📊 **Data Dashboard**
- Summary of latest radiosonde observations per site.
- Interactive tables with sorting & filtering (DataTables.js).
- Quick access to decoded BUFR/BFR/BFH/BIN metadata.

### 🗺️ **Balloon Tracking & 3D Trajectory**
- Real-time and historical tracking on 2D/3D map (MapLibre GL + Deck.gl).
- Optional Himawari satellite overlay for environmental context.
- Tooltip showing position, altitude, and wind vector.

### ⏱️ **Time Accuracy Analysis**
- Comparison of message group arrival times (AA, BB, CC, DD).
- Monthly performance chart using **Chart.js**.
- Automatically fetches and caches data from FTP / local SQLite DB.

### 📈 **Height Reach Graph**
- Displays balloon burst height per day.
- Graph + table with max/min summary.
- Sortable and exportable for reporting.

### 📅 **Data Availability Table**
- Checks completeness of daily observation data on the server.
- Grouped by site, date, and time (00Z / 12Z).

### 🧮 **Data Analysis**
- Vertical profile decoding (pressure, temperature, humidity, wind).
- Calculation of derived indices: CAPE, CIN, LCL, KI, LI, etc. (MetPy).
- Tropopause and freezing level detection.

### ⚙️ **Settings Panel**
- Switch active database (multi-site or test database).
- Manage data paths and cache refresh.
- Database creation and schema validation handled automatically.

### 🔒 **User Authentication**
- Secure login/logout using Flask-Login.
- Session-based access for all API endpoints.

---

## 🧠 Backend Architecture
- **Language**: Python 3.10+
- **Framework**: Flask + Gunicorn
- **Database**: SQLite (automatic caching for each file type: `.bufr`, `.bfr`, `.bfh`, `.bin`)
- **File Decoder**: PyBufrKit / Eccodes (with fallback JSON mode)
- **Task Flow**:
  1. Check cache (SQLite)
  2. If not found → Fetch from FTP
  3. Decode & insert metadata
  4. Serve via `/api/...` endpoints

---

## 💻 Frontend Stack
- **HTML/CSS/JS (vanilla)** with unified **Inter font** and responsive layout.
- **Chart.js 4.x** for interactive charts.
- **MapLibre GL JS** for mapping.
- **DataTables.js** for data visualization.
- **Sidebar layout** (collapsible, glassmorphism style).
- Responsive design — optimized for desktop & mobile.

---

## 🌍 Deployment Options

### 🚀 Local (Nginx + Gunicorn)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app locally:

```bash
gunicorn --bind 0.0.0.0:8082 app:app
```

3. Configure Nginx for reverse proxy (HTTP or HTTPS):

```nginx
server {
    listen 8082 ssl;
    ssl_certificate /etc/ssl/certs/rason_selfsigned.crt;
    ssl_certificate_key /etc/ssl/private/rason_selfsigned.key;
    location / {
        proxy_pass http://127.0.0.1:8082;
        include proxy_params;
    }
}
```

---

### ☁️ Render

1. Create a free Render account.  
2. New Web Service → Connect this repository.  
3. Set build and start commands:

```bash
Build command: pip install -r requirements.txt
Start command: gunicorn app:app
```

4. Add environment variables if needed:  
   - `FLASK_ENV=production`  
   - `PORT=8080`

---

### ⚡ Railway

1. Create a free Railway account.  
2. New Project → Deploy from GitHub.  
3. Railway auto-detects Flask; set the start command if needed:

```bash
gunicorn app:app
```

4. For custom domain or HTTPS, use Railway settings.

---

## 📦 Folder Structure
```
rason_monitoring/
│
├── app.py                     # Main Flask app
├── static/
│   ├── libs/                  # Local JS/CSS libraries (Chart.js, MapLibre, DataTables, etc.)
│   ├── images/                # Logos and icons
│   └── css/                   # Custom styles
│
├── templates/                 # All HTML pages (dashboard, tracking, charts, etc.)
│   ├── main.html
│   ├── height_reach.html
│   ├── time_accuracy.html
│   ├── data_availability.html
│   ├── raob_doc.html
│   └── login.html
│
├── db/
│   ├── rason_data.db          # Default SQLite database
│   └── active_db.txt          # Pointer to currently active DB
│
├── utils/
│   ├── ftp_fetch.py           # FTP downloader
│   ├── decode_bufr.py         # BUFR decoding
│   └── cache_manager.py       # SQLite cache logic
│
├── requirements.txt
└── README.md
```

---

## 🧾 API Endpoints (Examples)
| Endpoint | Description |
|-----------|--------------|
| `/api/time_accuracy/<site>` | Monthly time accuracy stats |
| `/api/height_reach/<site>` | Daily balloon burst height |
| `/api/data_availability/<site>` | Data completeness table |
| `/api/filter` | Filter files by site/date/hour/type |
| `/api/raob/<site>` | Retrieve decoded RAOB data |

---

## 🧰 Technologies Used
| Category | Tools / Libraries |
|-----------|------------------|
| **Backend** | Flask, Gunicorn, SQLite, PyBufrKit, MetPy |
| **Frontend** | HTML5, CSS3, JS, Chart.js, MapLibre GL, DataTables |
| **Deployment** | Nginx, Render, Railway |
| **Design** | Inter Font, Responsive Layout, Gradient UI |
| **Security** | Flask-Login session auth, HTTPS support |

---

## 📅 Version History
| Version    | Date        | Description                                             |
|----------  |------       |-------------                                            |
| **v1.0.0** |   Oct 2025  | Initial production release of Radiosonde Monitoring Web |
| **v1.1.0** | *(planned)* | Integration of FTP auto-sync                            |

---

## 👥 Credits
Developed by **Nanda Miftahul Khoyri**  
for **Terrindo & BMKG Upper-Air Division**  
2025 © All Rights Reserved.

---

## 🧭 Motto
> “Monitoring the Sky, One Balloon at a Time.”
