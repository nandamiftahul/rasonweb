import pandas as pd
import glob
import os

# === 📂 1. Atur folder dan file ===
files = glob.glob("doc/List SN*.xlsx")  # semua file Excel di folder 'doc/'
merged = []

# === 📄 2. Loop semua file dan sheet ===
for f in files:
    print(f"\n📘 Membaca file: {f}")
    try:
        xl = pd.ExcelFile(f)
        for sheet in xl.sheet_names:
            print(f"   📄 Sheet: {sheet}")
            try:
                df = xl.parse(sheet)
                if df.empty:
                    print("   ⚠️ Sheet kosong, dilewati.")
                    continue

                df.columns = [str(c).strip().lower() for c in df.columns]

                # ambil kolom yang relevan
                cols = [c for c in df.columns if any(k in c for k in ['kardus', 'seri', 'lokasi', 'manufactured', 'out'])]
                if len(cols) >= 3:
                    sub = df[cols].copy()

                    # isi kosong di kolom kardus → isi nilai sebelumnya
                    for c in sub.columns:
                        if 'kardus' in c:
                            sub[c] = sub[c].ffill()

                    sub["source_file"] = f"{os.path.basename(f)} :: {sheet}"
                    merged.append(sub)
                else:
                    print("   ⚠️ Tidak ditemukan kolom relevan, dilewati.")
            except Exception as e:
                print(f"   ⚠️ Gagal membaca sheet '{sheet}': {e}")
    except Exception as e:
        print(f"⚠️ Error membuka file {f}: {e}")

# === 💾 3. Gabungkan dan simpan ===
if merged:
    merged_df = pd.concat(merged, ignore_index=True)
    merged_df.to_json("merged_modem_m20.json", orient="records", indent=2, force_ascii=False)
    print("\n✅ Berhasil digabung ke merged_modem_m20.json")
    print("   Total data:", len(merged_df))
else:
    print("\n❌ Tidak ditemukan kolom yang sesuai di file manapun.")
