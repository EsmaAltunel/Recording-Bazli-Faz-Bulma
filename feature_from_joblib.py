"""
feature_from_joblib.py

Senkronize (etiketli) kişilerin FULL RECORDING IMU/EMG verilerinden
50 ms pencere bazlı feature dataset oluşturur.
Desteklenen Zıplama Tipleri: CMJ, SJ, DJ

Phase labels (6 States):
    0 -> Unweighting (CMJ)
    1 -> Propulsion (CMJ, SJ, DJ)
    2 -> Flight (CMJ, SJ, DJ)
    3 -> Landing (CMJ, SJ, DJ)
    4 -> Absorption (DJ)
    5 -> Background / waiting

Output:
    full_recording_ml_training_data.csv
"""

import os
import glob
import warnings
import joblib
import numpy as np
import pandas as pd

from classes import EMGData, IMUData, ForcePlate, Sensor, Recording, Player

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
WINDOW_SIZE = 0.05  # 50 ms
PHASE_MAP = {
    "Unweighting": 0,
    "Propulsion": 1,
    "Flight": 2,
    "Landing": 3,
    "Absorption": 4,
    "waiting": 5,
}
PHASE_NAMES = {
    0: "Unweighting",
    1: "Propulsion",
    2: "Flight",
    3: "Landing",
    4: "Absorption",
    5: "Background",
}


# ============================================================
# 1. WINDOW FEATURE EXTRACTION
# ============================================================
def get_window_features(data_array, time_array, start_t, end_t, prefix):
    """
    Belirli bir 50 ms zaman penceresindeki sinyalden:
    mean, std, rms, jerk çıkarır.
    """
    data_array = np.asarray(data_array, dtype=float)
    time_array = np.asarray(time_array, dtype=float)

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
# 2. FULL RECORDING PHASE LABEL
# ============================================================
def get_phase_label(mid_t, recording_phases):
    """
    Hazır full-recording phase CSV'sinden pencerenin orta noktasına göre label döndürür.
    """
    match = recording_phases[
        (recording_phases["Start_Time_sn"] <= mid_t) & (mid_t < recording_phases["End_Time_sn"])
    ]
    if match.empty:
        return PHASE_MAP["waiting"]

    phase_name = str(match.iloc[0]["Phase"]).strip()
    return PHASE_MAP.get(phase_name, PHASE_MAP["waiting"])


# ============================================================
# 3. MAIN
# ============================================================
def main():
    print("=" * 70)
    print("FULL RECORDING FEATURE EXTRACTION (CMJ + SJ + DJ)")
    print("=" * 70)

    # 1. Full-recording phase labels
    phase_csv = "imu_faz_zamanlari_tum_recording.csv"
    print(f"\n1. Full-recording phase labels loading from: {phase_csv}")
    phase_df = pd.read_csv(phase_csv)
    print(f"Phase rows loaded: {len(phase_df)}")

    # 2. Raw joblib players
    print("\n2. Raw recording joblibs loading...")
    joblib_paths = glob.glob("joblibler/*.joblib")
    players = {}
    for path in joblib_paths:
        player = joblib.load(path)
        players[player.name.lower()] = player
    print(f"{len(players)} players loaded.")

    # 3. Feature extraction
    all_rows = []

    for player in players.values():
        athlete = player.name

        for rec in player.recordings:
            trial_type = str(rec.trial_type).upper()

            if "MVC" in trial_type:
                continue

            parts = str(rec.trial_type).rsplit("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue

            jump_type = parts[0].upper()
            trial = int(parts[1])
            if jump_type not in ["CMJ", "SJ", "DJ"]:
                continue

            trial_name = f"{jump_type}_{trial}"

            # Bu recording'in full phase timeline'ı
            recording_phases = phase_df[
                (phase_df["Athlete"].str.lower() == athlete.lower())
                & (phase_df["Trial_Name"].str.upper() == trial_name.upper())
            ].copy()

            if recording_phases.empty:
                print(f"No phase info: {athlete} {trial_name}")
                continue

            recording_phases = recording_phases.sort_values("Start_Time_sn")

            # Referans IMU sensor
            valid_sensor = next(
                (s for s in rec.sensors.values() if s.imu_data is not None and len(s.imu_data.time) > 1),
                None,
            )
            if valid_sensor is None:
                print(f"No IMU data: {athlete} {trial_name}")
                continue

            t_ref = np.asarray(valid_sensor.imu_data.time, dtype=float)
            recording_start = float(t_ref[0])

            # Son jump'ın bittiği yerde kes (sonrasındaki gereksiz waiting kısmını alma)
            actual_jumps = recording_phases[recording_phases["Jump"] > 0]
            last_jump_end = float(actual_jumps["End_Time_sn"].max()) if not actual_jumps.empty else float(recording_phases["End_Time_sn"].max())
            recording_end = min(float(t_ref[-1]), last_jump_end)

            print(f"Processing {athlete:20s} {trial_name:6s} | {recording_start:.2f} - {recording_end:.2f} s (Last Jump: {last_jump_end:.2f} s)")

            # 50 ms windows
            curr_t = recording_start
            while curr_t + WINDOW_SIZE <= recording_end:
                next_t = curr_t + WINDOW_SIZE
                mid_t = curr_t + WINDOW_SIZE / 2.0
                row_features = {}

                # ALL SENSORS
                for s_id, sensor in rec.sensors.items():
                    # IMU
                    if sensor.imu_data is not None:
                        t_imu = np.asarray(sensor.imu_data.time, dtype=float)
                        acc_x = np.asarray(sensor.imu_data.acc_x, dtype=float)
                        acc_y = np.asarray(sensor.imu_data.acc_y, dtype=float)
                        acc_z = np.asarray(sensor.imu_data.acc_z, dtype=float)
                        gyro_x = np.asarray(sensor.imu_data.gyro_x, dtype=float)
                        gyro_y = np.asarray(sensor.imu_data.gyro_y, dtype=float)
                        gyro_z = np.asarray(sensor.imu_data.gyro_z, dtype=float)

                        acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
                        gyro_mag = np.sqrt(gyro_x ** 2 + gyro_y ** 2 + gyro_z ** 2)

                        row_features.update(get_window_features(acc_mag, t_imu, curr_t, next_t, f"S{s_id}_ACC_MAG"))
                        row_features.update(get_window_features(acc_x, t_imu, curr_t, next_t, f"S{s_id}_ACC_X"))
                        row_features.update(get_window_features(acc_y, t_imu, curr_t, next_t, f"S{s_id}_ACC_Y"))
                        row_features.update(get_window_features(acc_z, t_imu, curr_t, next_t, f"S{s_id}_ACC_Z"))
                        row_features.update(get_window_features(gyro_mag, t_imu, curr_t, next_t, f"S{s_id}_GYRO_MAG"))

                    # EMG
                    if sensor.emg_data is not None:
                        t_emg = np.asarray(sensor.emg_data.time, dtype=float)
                        emg_amp = np.asarray(sensor.emg_data.amplitude_mV, dtype=float)
                        row_features.update(get_window_features(emg_amp, t_emg, curr_t, next_t, f"S{s_id}_EMG"))

                # LABEL
                phase = get_phase_label(mid_t, recording_phases)
                row_features.update({
                    "Phase": phase,
                    "Meta_Athlete": athlete,
                    "Meta_JumpType": jump_type,
                    "Meta_Trial": trial,
                    "Meta_Time": mid_t,
                })

                all_rows.append(row_features)
                curr_t = next_t

    # SAVE
    final_df = pd.DataFrame(all_rows)
    final_df.to_csv("full_recording_ml_training_data.csv", index=False)

    print("\n" + "=" * 70)
    print("FULL RECORDING FEATURE EXTRACTION COMPLETED")
    print("=" * 70)
    print(f"Total windows: {len(final_df)}")
    print(f"Total columns: {len(final_df.columns)}")

    print("\nPhase Distribution:")
    for phase_id, count in final_df["Phase"].value_counts().sort_index().items():
        print(f"  {PHASE_NAMES.get(phase_id, 'Unknown'):12s} ({phase_id}): {count}")

    print("\nSaved to: full_recording_ml_training_data.csv")


if __name__ == "__main__":
    main()