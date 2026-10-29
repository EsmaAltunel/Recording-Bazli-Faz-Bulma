"""
phase.py

predicted_phases.csv içindeki full-recording model tahminlerini
tarar ve modelin kendi bulduğu jump/faz sınırlarını çıkarır.
Desteklenen Zıplama Tipleri: CMJ, SJ, DJ

FORCE KULLANILMAZ.
OFFSET KULLANILMAZ.
GROUND TRUTH KULLANILMAZ.

Input:
    predicted_phases.csv

Output:
    predicted_phase_segments.csv
"""

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = "predicted_phases.csv"
OUTPUT_PATH = "predicted_phase_segments.csv"

WINDOW_SIZE_S = 0.05
HALF_WINDOW = WINDOW_SIZE_S / 2.0

PHASE_UNWEIGHTING = 0
PHASE_PROPULSION = 1
PHASE_FLIGHT = 2
PHASE_LANDING = 3
PHASE_ABSORPTION = 4
PHASE_WAITING = 5


# ============================================================
# YARDIMCI: CONTIGUOUS PHASE BLOKLARINI BUL
# ============================================================
def find_contiguous_blocks(phases, target_phase):
    """
    target_phase için kesintisiz blokların
    (start_index, end_index) çiftlerini döndürür.
    """
    blocks = []
    start_idx = None

    for i, phase in enumerate(phases):
        if phase == target_phase:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                blocks.append((start_idx, i - 1))
                start_idx = None

    if start_idx is not None:
        blocks.append((start_idx, len(phases) - 1))

    return blocks


# ============================================================
# FULL RECORDING'DEN JUMP / PHASE ÇIKAR
# ============================================================
def extract_phase_segments(pred_df):
    results = []
    group_cols = ["Meta_Athlete", "Meta_JumpType", "Meta_Trial"]

    for (athlete, jump_type, trial), group in pred_df.groupby(group_cols):
        group = group.sort_values("Meta_Time").reset_index(drop=True)
        times = group["Meta_Time"].to_numpy(dtype=float)
        phases = group["Phase_Pred"].to_numpy(dtype=int)

        # Her Flight bloğunu bir jump anchor olarak kabul et
        flight_blocks = find_contiguous_blocks(phases, PHASE_FLIGHT)
        print(f"{athlete:20s} {jump_type}_{int(trial)} | Detected jumps: {len(flight_blocks)}")

        jump_number = 1
        for flight_start_idx, flight_end_idx in flight_blocks:
            # FLIGHT
            pred_flight_start = times[flight_start_idx] - HALF_WINDOW
            pred_flight_end = times[flight_end_idx] + HALF_WINDOW

            # PROPULSION: Flight'ın hemen öncesindeki Propulsion bloğu
            p_end = flight_start_idx - 1
            pred_propulsion_start = np.nan
            pred_propulsion_end = np.nan

            if p_end >= 0 and phases[p_end] == PHASE_PROPULSION:
                p_start = p_end
                while p_start >= 0 and phases[p_start] == PHASE_PROPULSION:
                    p_start -= 1
                p_start += 1

                pred_propulsion_start = times[p_start] - HALF_WINDOW
                pred_propulsion_end = times[p_end] + HALF_WINDOW
            else:
                p_start = flight_start_idx

            # UNWEIGHTING: Propulsion'ın hemen öncesindeki Unweighting (Sadece CMJ)
            pred_unweighting_start = np.nan
            pred_unweighting_end = np.nan

            if jump_type.upper() == "CMJ":
                u_end = p_start - 1
                if u_end >= 0 and phases[u_end] == PHASE_UNWEIGHTING:
                    u_start = u_end
                    while u_start >= 0 and phases[u_start] == PHASE_UNWEIGHTING:
                        u_start -= 1
                    u_start += 1

                    pred_unweighting_start = times[u_start] - HALF_WINDOW
                    pred_unweighting_end = times[u_end] + HALF_WINDOW

            # ABSORPTION: Propulsion'ın hemen öncesindeki Absorption (Sadece DJ)
            pred_absorption_start = np.nan
            pred_absorption_end = np.nan

            if jump_type.upper() == "DJ":
                a_end = p_start - 1
                if a_end >= 0 and phases[a_end] == PHASE_ABSORPTION:
                    a_start = a_end
                    while a_start >= 0 and phases[a_start] == PHASE_ABSORPTION:
                        a_start -= 1
                    a_start += 1

                    pred_absorption_start = times[a_start] - HALF_WINDOW
                    pred_absorption_end = times[a_end] + HALF_WINDOW

            # LANDING: Flight'ın hemen sonrasındaki Landing bloğu
            pred_landing_start = np.nan
            pred_landing_end = np.nan

            l_start = flight_end_idx + 1
            if l_start < len(phases) and phases[l_start] == PHASE_LANDING:
                l_end = l_start
                while l_end < len(phases) and phases[l_end] == PHASE_LANDING:
                    l_end += 1
                l_end -= 1

                pred_landing_start = times[l_start] - HALF_WINDOW
                pred_landing_end = times[l_end] + HALF_WINDOW

            results.append({
                "Athlete": athlete,
                "Jump_Type": jump_type,
                "Trial": int(trial),
                "Jump": jump_number,
                "Pred_Absorption_Start": pred_absorption_start,
                "Pred_Absorption_End": pred_absorption_end,
                "Pred_Unweighting_Start": pred_unweighting_start,
                "Pred_Unweighting_End": pred_unweighting_end,
                "Pred_Propulsion_Start": pred_propulsion_start,
                "Pred_Propulsion_End": pred_propulsion_end,
                "Pred_Flight_Start": pred_flight_start,
                "Pred_Flight_End": pred_flight_end,
                "Pred_Landing_Start": pred_landing_start,
                "Pred_Landing_End": pred_landing_end,
            })
            jump_number += 1

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("MODEL PREDICTED PHASE SEGMENTS (CMJ + SJ + DJ)")
    print("=" * 80)

    print("\nPrediction timeline loading...")
    pred_df = pd.read_csv(INPUT_PATH)
    print(f"Total predicted windows: {len(pred_df)}")

    print("\nScanning full recordings...")
    segments_df = extract_phase_segments(pred_df)

    # Sıralama
    segments_df = segments_df.sort_values(
        ["Athlete", "Jump_Type", "Trial", "Jump"]
    ).reset_index(drop=True)

    segments_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 80)
    print("PREDICTED PHASE SEGMENTS COMPLETED")
    print("=" * 80)
    print(f"Total detected jumps: {len(segments_df)}")
    print(f"\nSaved to: {OUTPUT_PATH}")

    print("\nDetected jumps per recording:")
    print(
        segments_df.groupby(["Athlete", "Jump_Type", "Trial"])
        .size()
        .rename("N_Detected_Jumps")
        .to_string()
    )


if __name__ == "__main__":
    main()