import pandas as pd
import glob
import os
import json

# === 1. Baca data lama (jika ada) ===
json_path = "../list_data_modem.json"
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        old_data = json.load(f)
    df_old = pd.DataFrame(old_data)
    print(f"📂 Data lama terbaca: {len(df_old)} baris")
else:
    df_old = pd.DataFrame()
    print("⚠️ Tidak ada file JSON lama, mulai baru.")

# === 2. Baca data Excel baru ===
files = glob.glob("doc/List SN*.xlsx")
merged = []

for f in files:
    print(f"\n📘 Membaca file: {f}")
    try:
        xl = pd.ExcelFile(f)
        for sheet in xl.sheet_names:
            print(f"   📄 Sheet: {sheet}")
            try:
                df = xl.parse(sheet)
                if df.empty:
                    continue

                df.columns = [str(c).strip().lower() for c in df.columns]
                cols = [c for c in df.columns if any(k in c for k in ['kardus', 'seri', 'lokasi', 'manufactured', 'out'])]
                if len(cols) >= 3:
                    sub = df[cols].copy()
                    for c in sub.columns:
                        if 'kardus' in c:
                            sub[c] = sub[c].ffill()
                    sub["source_file"] = f"{os.path.basename(f)} :: {sheet}"
                    merged.append(sub)
            except Exception as e:
                print(f"⚠️ Gagal baca sheet '{sheet}': {e}")
    except Exception as e:
        print(f"⚠️ Error buka file {f}: {e}")

if not merged:
    print("\n❌ Tidak ada data baru ditemukan.")
    exit()

df_new = pd.concat(merged, ignore_index=True)
print(f"📦 Data baru: {len(df_new)} baris")

# === 3. Gabungkan lama + baru ===
if not df_old.empty:
    combined = pd.concat([df_old, df_new], ignore_index=True)
    # hilangkan duplikat berdasar kolom 'seri' jika ada
    seri_cols = [c for c in combined.columns if 'seri' in c]
    if seri_cols:
        combined.drop_duplicates(subset=seri_cols[0], inplace=True, keep="last")
else:
    combined = df_new

# === 4. Simpan kembali ===
combined.to_json(json_path, orient="records", indent=2, force_ascii=False)
print(f"\n✅ Data berhasil diperbarui → {json_path}")
print("   Total baris akhir:", len(combined))
