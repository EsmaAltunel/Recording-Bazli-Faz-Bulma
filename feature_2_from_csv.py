"""
feature_2_from_csv.py

Senkronizasyonu bilinmeyen kişilerin FULL RECORDING ham Trigno (IMU/EMG) CSV
verilerinden doğrudan (joblib kullanmadan) 50 ms pencere bazlı feature dataset oluşturur.
Desteklenen Zıplama Tipleri: CMJ, SJ, DJ

Bu dosyada Phase LABEL YOKTUR (feature_2_from_joblib.py ile birebir aynı mantık).

Input:
    ham_sensor_verileri/*.csv (Force ve MVC dosyaları hariç ham Trigno sensör CSV'leri)

Output:
    unlabeled_full_recording_features_csvden.csv
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
CSV_DIR = "ham_sensor_verileri"
OUTPUT_CSV = "unlabeled_full_recording_features_csvden.csv"
WINDOW_SIZE = 0.05  # 50 ms

LOCATION_TO_SENSOR_ID = {
    "right vastus medialis": 1,
    "right rectus femoris": 2,
    "right biceps femoris": 3,
    "right gastrocnemius": 4,
    "left vastus medialis": 5,
    "left rectus femoris": 6,
    "left biceps femoris": 7,
    "left gastrocnemius": 8,
}


# ============================================================
# 1. WINDOW FEATURE EXTRACTION
# ============================================================
def get_window_features(data_array, time_array, start_t, end_t, prefix):
    """
    Belirli bir 50 ms zaman penceresindeki sinyalden:
    mean, std, rms, jerk çıkarır.
    """
    mask = (time_array >= start_t) & (time_array < end_t)
    window_data = data_array[mask]

    if len(window_data) > 0:
        return {
            f"{prefix}_mean": float(np.mean(window_data)),
            f"{prefix}_std": float(np.std(window_data)),
            f"{prefix}_rms": float(np.sqrt(np.mean(window_data ** 2))),
            f"{prefix}_jerk": float(np.mean(np.abs(np.diff(window_data)))) if len(window_data) > 1 else 0.0,
        }
    return {f"{prefix}_{stat}": 0.0 for stat in ["mean", "std", "rms", "jerk"]}


# ============================================================
# 2. FILENAME & TRIGNO CSV PARSER
# ============================================================
def parse_filename_metadata(filepath):
    """
    Dosya adından (örn: AYNUR_CMJ_1_1_0.csv) Athlete, JumpType ve Trial bilgilerini çıkarır.
    Force ve MVC dosyalarını otomatik atlar.
    """
    basename = os.path.basename(filepath)
    upper_name = basename.upper()

    if "FORCE" in upper_name or "MVC" in upper_name:
        return None

    match = re.match(r"^(.+?)_(CMJ|SJ|DJ)_(\d+)(?:_.*)?\.csv$", basename, re.IGNORECASE)
    if not match:
        return None

    raw_athlete = match.group(1).replace("_", " ").strip()
    athlete = raw_athlete.title()
    jump_type = match.group(2).upper()
    trial = int(match.group(3))

    return athlete, jump_type, trial


def load_trigno_csv(filepath):
    """
    Trigno Discover ham CSV dosyasını okur ve her sensör (S1..S8) için
    IMU ve EMG zaman/sinyal dizilerini döndürür.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        header_lines = []
        for _ in range(8):
            line = f.readline()
            if not line:
                break
            header_lines.append(line)

    if len(header_lines) < 8:
        return {}

    muscle_row = [col.strip() for col in header_lines[3].split(";")]
    channel_row = [col.strip() for col in header_lines[5].split(";")]

    # Her sensör bloğu 14 sütundan oluşur (8 sensör x 14 = 112 sütun)
    sensor_blocks = []
    usecols = []

    for base_col in range(0, len(channel_row), 14):
        if base_col + 13 >= len(channel_row):
            break

        ch_name = channel_row[base_col]
        if not ch_name:
            continue

        # Sensör numarasını "EMG <N> Time Series (s)" başlığından çıkar
        m_emg = re.search(r"EMG\s*(\d+)", ch_name, re.IGNORECASE)
        s_id = int(m_emg.group(1)) if m_emg else None

        # Başlıkta numara yoksa 4. satırdaki kas isminden bul
        if s_id is None and base_col < len(muscle_row):
            muscle_info = re.sub(r"\(.*?\)", "", muscle_row[base_col]).strip().lower()
            s_id = LOCATION_TO_SENSOR_ID.get(muscle_info)

        if s_id is None:
            continue

        # İlgili sensör için okunacak sütun indeksleri:
        # base+0: EMG Time, base+1: EMG (mV)
        # base+2: IMU Time, base+3: ACC_X, base+5: ACC_Y, base+7: ACC_Z
        # base+9: GYRO_X, base+11: GYRO_Y, base+13: GYRO_Z
        cols = [
            base_col,
            base_col + 1,
            base_col + 2,
            base_col + 3,
            base_col + 5,
            base_col + 7,
            base_col + 9,
            base_col + 11,
            base_col + 13,
        ]
        sensor_blocks.append((s_id, cols))
        usecols.extend(cols)

    if not sensor_blocks:
        return {}

    raw_df = pd.read_csv(
        filepath,
        sep=";",
        skiprows=8,
        header=None,
        decimal=",",
        usecols=usecols,
        skipinitialspace=True,
        na_values=["", " "],
        dtype=float,
        engine="c",
    )

    sensors = {}
    for s_id, cols in sensor_blocks:
        c_emg_t, c_emg_v, c_imu_t, c_ax, c_ay, c_az, c_gx, c_gy, c_gz = cols

        # EMG verisi (NaN olmayan satırlar)
        t_emg_raw = raw_df[c_emg_t].to_numpy(dtype=float)
        emg_raw = raw_df[c_emg_v].to_numpy(dtype=float)
        valid_emg = ~(np.isnan(t_emg_raw) | np.isnan(emg_raw))
        t_emg = t_emg_raw[valid_emg]
        emg_amp = emg_raw[valid_emg]

        # IMU verisi (NaN olmayan satırlar)
        t_imu_raw = raw_df[c_imu_t].to_numpy(dtype=float)
        ax_raw = raw_df[c_ax].to_numpy(dtype=float)
        ay_raw = raw_df[c_ay].to_numpy(dtype=float)
        az_raw = raw_df[c_az].to_numpy(dtype=float)
        gx_raw = raw_df[c_gx].to_numpy(dtype=float)
        gy_raw = raw_df[c_gy].to_numpy(dtype=float)
        gz_raw = raw_df[c_gz].to_numpy(dtype=float)

        valid_imu = ~(
            np.isnan(t_imu_raw)
            | np.isnan(ax_raw)
            | np.isnan(ay_raw)
            | np.isnan(az_raw)
            | np.isnan(gx_raw)
            | np.isnan(gy_raw)
            | np.isnan(gz_raw)
        )

        t_imu = t_imu_raw[valid_imu]
        acc_x = ax_raw[valid_imu]
        acc_y = ay_raw[valid_imu]
        acc_z = az_raw[valid_imu]
        gyro_x = gx_raw[valid_imu]
        gyro_y = gy_raw[valid_imu]
        gyro_z = gz_raw[valid_imu]

        acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        gyro_mag = np.sqrt(gyro_x ** 2 + gyro_y ** 2 + gyro_z ** 2)

        sensors[s_id] = {
            "imu": {
                "time": t_imu,
                "acc_x": acc_x,
                "acc_y": acc_y,
                "acc_z": acc_z,
                "acc_mag": acc_mag,
                "gyro_mag": gyro_mag,
            } if len(t_imu) > 0 else None,
            "emg": {
                "time": t_emg,
                "amplitude_mV": emg_amp,
            } if len(t_emg) > 0 else None,
        }

    return sensors


# ============================================================
# 3. MAIN
# ============================================================
def main():
    print("=" * 70)
    print("UNLABELED FULL RECORDING FEATURE EXTRACTION FROM CSV (CMJ + SJ + DJ)")
    print("=" * 70)

    csv_paths = sorted(glob.glob(os.path.join(CSV_DIR, "**", "*.csv"), recursive=True))
    recording_files = []
    for path in csv_paths:
        meta = parse_filename_metadata(path)
        if meta is not None:
            recording_files.append((path, *meta))

    print(f"\nFound {len(recording_files)} raw sensor CSV recordings in '{CSV_DIR}/'.")

    all_rows = []

    for filepath, athlete, jump_type, trial in recording_files:
        trial_name = f"{jump_type}_{trial}"
        sensors = load_trigno_csv(filepath)

        if not sensors:
            print(f"No valid Trigno sensor columns: {athlete} {trial_name}")
            continue

        # Referans IMU sensor
        valid_sensor = next(
            (s for s in sensors.values() if s["imu"] is not None and len(s["imu"]["time"]) > 1),
            None,
        )
        if valid_sensor is None:
            print(f"No IMU data: {athlete} {trial_name}")
            continue

        t_ref = valid_sensor["imu"]["time"]
        recording_start = float(t_ref[0])
        recording_end = float(t_ref[-1])
        print(f"Processing {athlete:20s} {trial_name:6s} | {recording_start:.2f} - {recording_end:.2f} s")

        # Sensör sırasını (S4, S7, ... veya S1..S8) koruyarak 50 ms pencereleri tara
        curr_t = recording_start
        while curr_t + WINDOW_SIZE <= recording_end:
            next_t = curr_t + WINDOW_SIZE
            mid_t = curr_t + WINDOW_SIZE / 2.0
            row_features = {}

            for s_id, sensor in sensors.items():
                # IMU
                if sensor["imu"] is not None:
                    imu = sensor["imu"]
                    t_imu = imu["time"]
                    row_features.update(get_window_features(imu["acc_mag"], t_imu, curr_t, next_t, f"S{s_id}_ACC_MAG"))
                    row_features.update(get_window_features(imu["acc_x"], t_imu, curr_t, next_t, f"S{s_id}_ACC_X"))
                    row_features.update(get_window_features(imu["acc_y"], t_imu, curr_t, next_t, f"S{s_id}_ACC_Y"))
                    row_features.update(get_window_features(imu["acc_z"], t_imu, curr_t, next_t, f"S{s_id}_ACC_Z"))
                    row_features.update(get_window_features(imu["gyro_mag"], t_imu, curr_t, next_t, f"S{s_id}_GYRO_MAG"))

                # EMG
                if sensor["emg"] is not None:
                    emg = sensor["emg"]
                    row_features.update(get_window_features(emg["amplitude_mV"], emg["time"], curr_t, next_t, f"S{s_id}_EMG"))

            # META ONLY — NO PHASE
            row_features.update({
                "Meta_Athlete": athlete,
                "Meta_JumpType": jump_type,
                "Meta_Trial": trial,
                "Meta_Time": mid_t,
            })

            all_rows.append(row_features)
            curr_t = next_t

    # SAVE
    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 70)
    print("UNLABELED CSV FEATURE EXTRACTION COMPLETED")
    print("=" * 70)
    print(f"Total windows: {len(final_df)}")
    print(f"Total columns: {len(final_df.columns)}")
    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
