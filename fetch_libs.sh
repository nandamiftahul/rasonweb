#!/bin/bash
# ===========================================================
# fetch_libs.sh — Download local frontend libraries
# Works offline for Flask / HTML dashboards
# Includes: jQuery, DataTables, Leaflet, Inter font, Bootstrap 5
# ===========================================================

set -e

echo "📦 Preparing directory structure..."
mkdir -p static/libs/{jquery,datatables,leaflet,fonts/inter,bootstrap,bootstrap-icons}

# -----------------------------------------------
# 1. jQuery
# -----------------------------------------------
echo "[1/6] Downloading jQuery..."
wget -q -O static/libs/jquery/jquery-3.6.0.min.js https://code.jquery.com/jquery-3.6.0.min.js

# -----------------------------------------------
# 2. DataTables
# -----------------------------------------------
echo "[2/6] Downloading DataTables..."
wget -q -O static/libs/datatables/datatables.min.css https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css
wget -q -O static/libs/datatables/datatables.min.js https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js

# -----------------------------------------------
# 3. Leaflet
# -----------------------------------------------
echo "[3/6] Downloading Leaflet..."
wget -q -O static/libs/leaflet/leaflet.css https://unpkg.com/leaflet/dist/leaflet.css
wget -q -O static/libs/leaflet/leaflet.js https://unpkg.com/leaflet/dist/leaflet.js

# -----------------------------------------------
# 4. Inter Font (Google Fonts)
# -----------------------------------------------
echo "[4/6] Downloading Inter font CSS..."
wget -q -O static/libs/fonts/inter/inter.css "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap"

echo "🔍 Extracting font URLs and downloading..."
grep -o 'https://[^)]*\.woff2' static/libs/fonts/inter/inter.css | while read -r url; do
    fname=$(basename "$url")
    wget -q -O "static/libs/fonts/inter/$fname" "$url"
done

# Ganti URL agar menjadi relatif lokal
sed -i 's|https://[^)]*/||g' static/libs/fonts/inter/inter.css

# -----------------------------------------------
# 5. Bootstrap 5
# -----------------------------------------------
echo "[5/6] Downloading Bootstrap 5..."
wget -q -O static/libs/bootstrap/bootstrap.min.css https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css
wget -q -O static/libs/bootstrap/bootstrap.bundle.min.js https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js

# -----------------------------------------------
# 6. Bootstrap Icons
# -----------------------------------------------
echo "[6/6] Downloading Bootstrap Icons..."
wget -q -O static/libs/bootstrap-icons/bootstrap-icons.css https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css

# Extract URLs to fonts and download locally
grep -o 'https://[^)]*\.woff2' static/libs/bootstrap-icons/bootstrap-icons.css | while read -r url; do
    fname=$(basename "$url")
    wget -q -O "static/libs/bootstrap-icons/$fname" "$url"
done

# Update CSS to use local font path
sed -i 's|https://[^)]*/||g' static/libs/bootstrap-icons/bootstrap-icons.css

# -----------------------------------------------
# 7. Summary
# -----------------------------------------------
echo
echo "✅ All libraries downloaded successfully!"
echo
echo "📁 Folder structure:"
echo "static/libs/"
echo " ├── jquery/jquery-3.6.0.min.js"
echo " ├── datatables/{datatables.min.css, datatables.min.js}"
echo " ├── leaflet/{leaflet.css, leaflet.js}"
echo " ├── fonts/inter/{inter.css, *.woff2}"
echo " ├── bootstrap/{bootstrap.min.css, bootstrap.bundle.min.js}"
echo " └── bootstrap-icons/{bootstrap-icons.css, *.woff2}"
echo
echo "💡 Use in your Flask HTML templates:"
echo
echo '<link rel="stylesheet" href="{{ url_for("static", filename="libs/fonts/inter/inter.css") }}">'
echo '<link rel="stylesheet" href="{{ url_for("static", filename="libs/bootstrap/bootstrap.min.css") }}">'
echo '<link rel="stylesheet" href="{{ url_for("static", filename="libs/bootstrap-icons/bootstrap-icons.css") }}">'
echo '<link rel="stylesheet" href="{{ url_for("static", filename="libs/datatables/datatables.min.css") }}">'
echo '<link rel="stylesheet" href="{{ url_for("static", filename="libs/leaflet/leaflet.css") }}">'
echo '<script src="{{ url_for("static", filename="libs/jquery/jquery-3.6.0.min.js") }}"></script>'
echo '<script src="{{ url_for("static", filename="libs/bootstrap/bootstrap.bundle.min.js") }}"></script>'
echo '<script src="{{ url_for("static", filename="libs/datatables/datatables.min.js") }}"></script>'
echo '<script src="{{ url_for("static", filename="libs/leaflet/leaflet.js") }}"></script>'
echo
echo "✨ Done — all frontend dependencies are now available offline!"
