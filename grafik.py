"""
recording_bazli_landing_median_sync.py

Her Athlete + Jump_Type + Trial = 1 recording.

O recording içindeki bütün jump'ların Landing Start offsetlerini alır:
    offset_i = Pred_Landing_Start_i - Force_Landing_Start_i

Sonra:
    recording_offset = median(offset_i)

hesaplar ve TÜM recording IMU zamanına tek bir shift uygular:
    t_imu_sync = t_imu - recording_offset

Her recording için tek grafik oluşturur:
    Üst: Force full recording
    Alt: Median offset uygulanmış IMU full recording

Force ve predicted (Propulsion, Flight, Landing) eventleri grafik üzerinde gösterilir.
"""

import os
import glob
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classes import EMGData, IMUData, ForcePlate, Sensor, Recording, Player

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
OFFSET_CSV = "force_faz_zamanlari_tam_birlesik.csv"
PRED_CSV = "predicted_phase_segments.csv"
JOBLIB_DIR = "joblibler_2"
OUTPUT_DIR = "Recording_Bazli_Landing_Median_Sync"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD PLAYERS & SENSORS
# ============================================================
def load_players():
    players = {}
    paths = glob.glob(os.path.join(JOBLIB_DIR, "*.joblib"))
    for path in paths:
        player = joblib.load(path)
        players[player.name.lower().strip()] = player
    print(f"{len(players)} players loaded.")
    return players


def find_recording(player, jump_type, trial):
    trial_name = f"{jump_type}_{int(trial)}"
    for rec in player.recordings:
        if str(rec.trial_type).upper() == trial_name.upper():
            return rec
    return None


def get_reference_imu(rec):
    for s_id, sensor in rec.sensors.items():
        if sensor.imu_data is not None and len(sensor.imu_data.time) > 1:
            return s_id, sensor
    return None, None


def get_acc_magnitude(sensor):
    imu = sensor.imu_data
    t = np.asarray(imu.time, dtype=float)
    acc_x = np.asarray(imu.acc_x, dtype=float)
    acc_y = np.asarray(imu.acc_y, dtype=float)
    acc_z = np.asarray(imu.acc_z, dtype=float)
    acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    return t, acc_mag


def get_force(rec):
    fs = float(rec.force_plate.frequency)
    left = np.asarray(rec.force_plate.total_left, dtype=float)
    right = np.asarray(rec.force_plate.total_right, dtype=float)
    total_force = (left + right) * 9.81
    t_force = np.arange(len(total_force), dtype=float) / fs
    return t_force, total_force


# ============================================================
# EVENT LINES
# ============================================================
def draw_force_events(ax, rec_df):
    first_abs, first_prop, first_flight, first_landing = True, True, True, True

    for _, row in rec_df.iterrows():
        # Absorption (DJ)
        val = row.get("Absorption_Start_sn", np.nan)
        if pd.isna(val):
            val = row.get("Touchdown_Start_sn", np.nan)
        if not pd.isna(val):
            ax.axvline(val, linestyle="--", color="purple", alpha=0.8, label="Force Absorption" if first_abs else None)
            first_abs = False

        # Propulsion
        val = row.get("Propulsion_Start_sn", np.nan)
        if not pd.isna(val):
            ax.axvline(val, linestyle="--", color="green", alpha=0.8, label="Force Propulsion" if first_prop else None)
            first_prop = False

        # Flight
        val = row.get("Flight_Start_sn", np.nan)
        if not pd.isna(val):
            ax.axvline(val, linestyle=":", color="red", alpha=0.8, label="Force Flight" if first_flight else None)
            first_flight = False

        # Landing
        val = row.get("Landing_Start_sn", np.nan)
        if not pd.isna(val):
            ax.axvline(val, linestyle="-.", color="orange", alpha=0.9, label="Force Landing" if first_landing else None)
            first_landing = False


def draw_predicted_events(ax, rec_df, recording_offset):
    first_abs, first_prop, first_flight, first_landing = True, True, True, True

    for _, row in rec_df.iterrows():
        # Absorption (DJ)
        val = row.get("Pred_Absorption_Start", np.nan)
        if not pd.isna(val):
            ax.axvline(val - recording_offset, linestyle="--", color="purple", alpha=0.8, label="IMU Pred Absorption" if first_abs else None)
            first_abs = False

        # Propulsion
        val = row.get("Pred_Propulsion_Start", np.nan)
        if not pd.isna(val):
            ax.axvline(val - recording_offset, linestyle="--", color="green", alpha=0.8, label="IMU Pred Propulsion" if first_prop else None)
            first_prop = False

        # Flight
        val = row.get("Pred_Flight_Start", np.nan)
        if pd.isna(val):
            val = row.get("Pred_Propulsion_End", np.nan)
        if not pd.isna(val):
            ax.axvline(val - recording_offset, linestyle=":", color="red", alpha=0.8, label="IMU Pred Flight" if first_flight else None)
            first_flight = False

        # Landing
        val = row.get("Pred_Landing_Start", np.nan)
        if not pd.isna(val):
            ax.axvline(val - recording_offset, linestyle="-.", color="orange", alpha=0.9, label="IMU Pred Landing" if first_landing else None)
            first_landing = False


# ============================================================
# PLOT FULL RECORDING
# ============================================================
def plot_recording(athlete, jump_type, trial, rec, rec_df, recording_offset):
    sensor_id, sensor = get_reference_imu(rec)
    if sensor is None:
        print("  No IMU.")
        return

    t_force, force = get_force(rec)
    t_imu, acc_mag = get_acc_magnitude(sensor)
    t_imu_sync = t_imu - recording_offset

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

    # Upper: Force Full Recording
    axes[0].plot(t_force, force, linewidth=1, color="tab:blue")
    draw_force_events(axes[0], rec_df)
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title(f"{athlete} | {jump_type}_{trial} | FULL FORCE RECORDING")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    # Lower: Synced IMU Full Recording
    axes[1].plot(t_imu_sync, acc_mag, linewidth=1, color="tab:orange")
    draw_predicted_events(axes[1], rec_df, recording_offset)
    axes[1].set_ylabel("|Acceleration|")
    axes[1].set_xlabel("Synchronized Time (s)")
    axes[1].set_title(
        f"FULL IMU RECORDING AFTER LANDING MEDIAN SYNC | Sensor {sensor_id} | Offset = {recording_offset:.4f} s"
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    # Common X Axis
    min_t = min(np.min(t_force), np.min(t_imu_sync))
    max_t = max(np.max(t_force), np.max(t_imu_sync))
    axes[0].set_xlim(min_t, max_t)
    axes[1].set_xlim(min_t, max_t)

    fig.suptitle(
        f"{athlete} | {jump_type}_{trial} | Recording-level Landing Median Synchronization",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()

    safe_name = athlete.replace(" ", "_").replace("/", "_")
    filename = f"{safe_name}_{jump_type}_{trial}_LANDING_MEDIAN_SYNC.png"
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def load_data():
    gt_df = pd.read_csv(OFFSET_CSV) if os.path.exists(OFFSET_CSV) else None
    pred_df = pd.read_csv(PRED_CSV) if os.path.exists(PRED_CSV) else None

    if gt_df is not None and pred_df is not None:
        if "Offset_Landing_s" in gt_df.columns:
            df = gt_df
        elif "Pred_Landing_Start" in gt_df.columns and "Landing_Start_sn" in gt_df.columns:
            df = gt_df.copy()
            df["Offset_Landing_s"] = df["Pred_Landing_Start"] - df["Landing_Start_sn"]
        else:
            df = pd.merge(pred_df, gt_df, on=["Athlete", "Jump_Type", "Trial", "Jump"], how="inner")
            df["Offset_Landing_s"] = df["Pred_Landing_Start"] - df["Landing_Start_sn"]
    elif gt_df is not None:
        df = gt_df
        if "Offset_Landing_s" not in df.columns and "Pred_Landing_Start" in df.columns and "Landing_Start_sn" in df.columns:
            df["Offset_Landing_s"] = df["Pred_Landing_Start"] - df["Landing_Start_sn"]
    elif pred_df is not None:
        df = pred_df
    else:
        raise FileNotFoundError("Gerekli veri dosyaları bulunamadı.")

    if "Absorption_Start_sn" not in df.columns and "Touchdown_Start_sn" in df.columns:
        df["Absorption_Start_sn"] = df["Touchdown_Start_sn"]

    return df


def main():
    print("=" * 80)
    print("RECORDING-BASED LANDING MEDIAN SYNCHRONIZATION")
    print("=" * 80)

    df = load_data()
    players = load_players()

    group_cols = ["Athlete", "Jump_Type", "Trial"]
    processed = 0

    for (athlete, jump_type, trial), rec_df in df.groupby(group_cols):
        athlete = str(athlete)
        jump_type = str(jump_type).upper()
        trial = int(trial)

        if "Offset_Landing_s" not in rec_df.columns:
            print(f"No Offset_Landing_s column for: {athlete} {jump_type}_{trial}")
            continue

        landing_offsets = rec_df["Offset_Landing_s"].dropna().to_numpy(dtype=float)
        if len(landing_offsets) == 0:
            print(f"No Landing offsets: {athlete} {jump_type}_{trial}")
            continue

        recording_offset = float(np.median(landing_offsets))
        print(f"\n{athlete:20s} {jump_type}_{trial}")
        print(f"  Jump Landing offsets: {np.round(landing_offsets, 4)}")
        print(f"  -> MEDIAN OFFSET: {recording_offset:.4f} s")

        player = players.get(athlete.lower().strip())
        if player is None:
            print("  Player not found.")
            continue

        rec = find_recording(player, jump_type, trial)
        if rec is None:
            print("  Recording not found.")
            continue

        if rec.force_plate is None:
            print("  No force plate.")
            continue

        plot_recording(athlete, jump_type, trial, rec, rec_df, recording_offset)
        processed += 1

    print("\n" + "=" * 80)
    print(f"Completed recordings: {processed}")
    print(f"Graphs saved to: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()