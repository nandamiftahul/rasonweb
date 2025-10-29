import subprocess
import json
import re
import sys
from pathlib import Path

# === Variable list tetap (tidak boleh diubah) ===
VARIABLES_META = [
    "wmo_block", "wmo_station", "year", "month", "day",
    "hour", "minute", "second", "station_lat", "station_lon",
    "station_height_m", "radiosonde_serial_number",
    "radiosonde_ascension_number", "radiosonde_release_number",
    "radiosonde_ground_rx_system", "radiosonde_operating_frequency",
    "balloon_manufacturer", "balloon_weight_kg", "balloon_gas_type",
    "pressure_sensor_type", "temperature_sensor_type",
    "humidity_sensor_type", "software_version",
    "reason_for_termination", "system_status"
]

VARIABLES_LEVEL = [
    "pressure_hPa", "height_m", "temp_C", "dewpoint_C",
    "wind_dir_deg", "wind_speed_mps", "lat_disp", "lon_disp",
    "time_s", "status_flag"
]

# === Kunci pencarian heuristik antar variable dan kata di BUFR ===
KEYWORDS = {
    "wmo_block": "WMO BLOCK NUMBER",
    "wmo_station": "WMO STATION NUMBER",
    "year": "YEAR",
    "month": "MONTH",
    "day": "DAY",
    "hour": "HOUR",
    "minute": "MINUTE",
    "second": "SECOND",
    "station_lat": "LATITUDE",
    "station_lon": "LONGITUDE",
    "station_height_m": "HEIGHT OF STATION GROUND",
    "radiosonde_serial_number": "RADIOSONDE SERIAL NUMBER",
    "radiosonde_ascension_number": "RADIOSONDE ASCENSION NUMBER",
    "radiosonde_release_number": "RADIOSONDE RELEASE NUMBER",
    "radiosonde_ground_rx_system": "RADIOSONDE GROUND RECEIVING SYSTEM",
    "radiosonde_operating_frequency": "RADIOSONDE OPERATING FREQUENCY",
    "balloon_manufacturer": "BALLOON MANUFACTURER",
    "balloon_weight_kg": "WEIGHT OF BALLOON",
    "balloon_gas_type": "TYPE OF GAS USED IN BALLOON",
    "pressure_sensor_type": "TYPE OF PRESSURE SENSOR",
    "temperature_sensor_type": "TYPE OF TEMPERATURE SENSOR",
    "humidity_sensor_type": "TYPE OF HUMIDITY SENSOR",
    "software_version": "SOFTWARE IDENTIFICATION AND VERSION NUMBER",
    "reason_for_termination": "REASON FOR TERMINATION",
    "system_status": "TRACKING TECHNIQUE/STATUS OF SYSTEM USED",
    "pressure_hPa": "PRESSURE",
    "height_m": "GEOPOTENTIAL HEIGHT",
    "temp_C": "TEMPERATURE/AIR TEMPERATURE",
    "dewpoint_C": "DEW-POINT TEMPERATURE",
    "wind_dir_deg": "WIND DIRECTION",
    "wind_speed_mps": "WIND SPEED",
    "lat_disp": "LATITUDE DISPLACEMENT",
    "lon_disp": "LONGITUDE DISPLACEMENT",
    "time_s": "LONG TIME PERIOD OR DISPLACEMENT",
    "status_flag": "EXTENDED VERTICAL SOUNDING SIGNIFICANCE"
}

def decode_bufr(file_path):
    """Run pybufrkit decode -a and return text"""
    result = subprocess.run(
        ["pybufrkit", "decode", "-a", str(file_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout

def extract_descriptors(decoded_text):
    """Extract descriptors → list of tuples (code, name)"""
    pattern = re.compile(r"^\s*(\d{6})\s+([A-Z0-9/() \-]+)\s")
    found = []
    for line in decoded_text.splitlines():
        m = pattern.match(line)
        if m:
            found.append((m.group(1).strip(), m.group(2).strip().upper()))
    return found

def to_wmo_code(num_code):
    """Convert 004001 → 0-04-001"""
    if len(num_code) != 6 or not num_code.isdigit():
        return None
    return f"{num_code[0]}-{int(num_code[1:3]):02d}-{int(num_code[3:]):03d}"

def match_variables(var_list, descriptors):
    """Cari descriptor cocok untuk setiap variable"""
    output = []
    for var in var_list:
        key = KEYWORDS.get(var, "").upper()
        match = next(((code, name) for code, name in descriptors if key in name), (None, None))
        code, name = match

        # hilangkan angka di depan 'original'
        clean_name = name if not name else re.sub(r"^\d{6}\s*", "", name).strip()

        output.append({
            "variable": var,
            "code": to_wmo_code(code) if code else None,
            "original": clean_name if clean_name else None
        })
    return output


def main():
    if len(sys.argv) < 2:
        print("Usage: python bufr_autocode_finder.py <file.bfr>")
        sys.exit(1)

    bufr_file = Path(sys.argv[1])
    decoded_text = decode_bufr(bufr_file)
    descriptors = extract_descriptors(decoded_text)

    meta_out = match_variables(VARIABLES_META, descriptors)
    level_out = match_variables(VARIABLES_LEVEL, descriptors)

    result = {"meta": meta_out, "level": level_out}

    output_file = bufr_file.with_suffix(".autocode.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Mapping saved to {output_file}")
    print(f"Found {len(descriptors)} descriptors in file.")


if __name__ == "__main__":
    main()
