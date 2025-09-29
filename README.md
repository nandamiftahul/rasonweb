# Radiosonde Viewer 🌐

A Flask-based web app to decode and visualize radiosonde BUFR data.

## Features
- Upload radiosonde BUFR/ BFR/ BFH files.
- Interactive map with balloon tracking.
- Metadata & weather charts.
- Dashboard view (last files from FTP).
- Login authentication.

## 🚀 Deploy on Render / Railway

### Render
1. Create a free Render account.
2. New Web Service → Connect this repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`

### Railway
1. Create a free Railway account.
2. New Project → Deploy from GitHub.
3. Railway auto-detects Flask. If needed, set:
   - Start command: `gunicorn app:app`

## ⚠️ Note
- Free hosting may block `pybufrkit` if compilation fails.  
- If so, preprocess BUFR files offline and upload JSON/CSV for demo.
