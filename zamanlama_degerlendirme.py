"""
zamanlama_degerlendirme.py
===========================
Full-recording timing accuracy evaluation script.
Desteklenen Zıplama Tipleri: CMJ, SJ, DJ

Converts model phase predictions (6-state: Unweighting, Propulsion, Flight, Landing, Absorption, Background)
over complete recording timelines into predicted jump event timestamps (start/end) and compares 
them against ground truth force plate timing annotations.

Inputs:
    - full_recording_ml_training_data.csv
    - imu_faz_zamanlari_tum_recording.csv

Outputs:
    - zamanlama_degerlendirme_sonuclari.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

WINDOW_SIZE_S = 0.05  # 50 ms sliding window step
N_STATES = 6
PHASE_UNWEIGHTING = 0
PHASE_PROPULSION = 1
PHASE_FLIGHT = 2
PHASE_LANDING = 3
PHASE_ABSORPTION = 4
PHASE_BACKGROUND = 5


def estimate_empirical_transition_matrix(data_df):
    """
    Estimates 6-state HMM transition matrix and initial state probabilities 
    exclusively from training fold recordings.
    """
    transition_counts = np.zeros((N_STATES, N_STATES), dtype=float)
    start_counts = np.zeros(N_STATES, dtype=float)

    group_cols = ["Meta_Athlete", "Meta_JumpType", "Meta_Trial"]
    for _, group in data_df.groupby(group_cols, dropna=False):
        group = group.sort_values("Meta_Time")
        phases = group["Phase"].astype(int).values

        if len(phases) > 0:
            start_counts[phases[0]] += 1

        for i in range(len(phases) - 1):
            curr_state = phases[i]
            next_state = phases[i + 1]
            if 0 <= curr_state < N_STATES and 0 <= next_state < N_STATES:
                transition_counts[curr_state, next_state] += 1

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    empirical_matrix = np.divide(
        transition_counts,
        row_sums,
        out=np.zeros_like(transition_counts),
        where=row_sums != 0,
    )

    for state in range(N_STATES):
        if empirical_matrix[state].sum() == 0:
            empirical_matrix[state, state] = 1.0

    start_sum = start_counts.sum()
    start_probs = start_counts / start_sum if start_sum > 0 else np.ones(N_STATES) / N_STATES
    return empirical_matrix, start_probs


def apply_viterbi_filter(probabilities, trans_matrix, start_probs):
    """
    Decodes full recording sequence into 6-state phase timeline using Viterbi algorithm.
    """
    n_samples, n_states = probabilities.shape
    if n_samples == 0:
        return np.array([], dtype=int)

    log_probs = np.log(probabilities + 1e-10)
    log_trans = np.log(trans_matrix + 1e-10)
    log_start = np.log(start_probs + 1e-10)

    viterbi_table = np.zeros((n_samples, n_states))
    backpointer = np.zeros((n_samples, n_states), dtype=int)
    viterbi_table[0, :] = log_probs[0, :] + log_start

    for t in range(1, n_samples):
        for s in range(n_states):
            trans_probs = viterbi_table[t - 1, :] + log_trans[:, s]
            best_prev = np.argmax(trans_probs)
            backpointer[t, s] = best_prev
            viterbi_table[t, s] = trans_probs[best_prev] + log_probs[t, s]

    smoothed_path = np.zeros(n_samples, dtype=int)
    smoothed_path[-1] = np.argmax(viterbi_table[-1, :])

    for t in range(n_samples - 2, -1, -1):
        smoothed_path[t] = backpointer[t + 1, smoothed_path[t + 1]]

    return smoothed_path


def run_loso_full_recording(df, features):
    """
    Performs Specific LOSO cross-validation over full recordings for CMJ, SJ, and DJ.
    """
    jump_types = df["Meta_JumpType"].unique()
    athletes = df["Meta_Athlete"].unique()
    all_preds = []

    for j_type in jump_types:
        print(f"  -> Running LOSO for {j_type} full recordings...")
        type_df = df[df["Meta_JumpType"] == j_type].reset_index(drop=True)

        for test_athlete in athletes:
            train_data = type_df[type_df["Meta_Athlete"] != test_athlete].copy()
            test_data = type_df[type_df["Meta_Athlete"] == test_athlete].copy().reset_index(drop=True)

            if test_data.empty or train_data.empty:
                continue

            trans_matrix, start_probs = estimate_empirical_transition_matrix(train_data)

            le = LabelEncoder()
            y_train = le.fit_transform(train_data["Phase"])

            model = CatBoostClassifier(iterations=500, depth=6, verbose=0, random_seed=42)
            model.fit(train_data[features], y_train)

            test_data["Temp_Index"] = np.arange(len(test_data))
            test_data["Phase_Pred"] = -1

            # Decode per full recording
            group_cols = ["Meta_Trial"]
            for _, group in test_data.groupby(group_cols, dropna=False):
                if group.empty:
                    continue

                group = group.sort_values("Meta_Time")
                g_indices = group["Temp_Index"].values
                g_X = group[features]

                raw_probs = model.predict_proba(g_X)
                probs = np.zeros((len(g_X), N_STATES))
                for idx, cls in enumerate(le.classes_):
                    probs[:, int(cls)] = raw_probs[:, idx]

                smoothed_g_preds = apply_viterbi_filter(probs, trans_matrix, start_probs)
                test_data.loc[g_indices, "Phase_Pred"] = smoothed_g_preds

            preds = test_data["Phase_Pred"].values
            result_chunk = test_data[
                ["Meta_Athlete", "Meta_JumpType", "Meta_Trial", "Meta_Time"]
            ].copy()
            result_chunk["Phase_pred"] = preds
            all_preds.append(result_chunk)

    return pd.concat(all_preds, ignore_index=True)


def extract_predicted_segments(preds_df):
    """
    Scans full recording predictions to detect jump cycles and 
    extract phase segment boundaries completely independent of ground truth.
    Anchors on 'Flight' (Phase 2).
    """
    rows = []
    half_window = WINDOW_SIZE_S / 2.0
    group_cols = ["Meta_Athlete", "Meta_JumpType", "Meta_Trial"]

    for (athlete, jump_type, trial), group in preds_df.groupby(group_cols):
        group = group.sort_values("Meta_Time").reset_index(drop=True)
        times = group["Meta_Time"].to_numpy()
        phases = group["Phase_pred"].to_numpy()

        # 1. Contiguous Flight blocks
        in_flight = False
        flight_starts, flight_ends = [], []

        for i in range(len(phases)):
            if phases[i] == PHASE_FLIGHT and not in_flight:
                flight_starts.append(i)
                in_flight = True
            elif phases[i] != PHASE_FLIGHT and in_flight:
                flight_ends.append(i - 1)
                in_flight = False

        if in_flight:
            flight_ends.append(len(phases) - 1)

        # 2. Segment boundaries around each Flight block
        jump_count = 1
        for f_start, f_end in zip(flight_starts, flight_ends):
            # Propulsion
            p_end = f_start - 1
            p_start = p_end
            while p_start >= 0 and phases[p_start] == PHASE_PROPULSION:
                p_start -= 1
            p_start += 1

            if p_end < 0 or p_start > p_end or phases[p_end] != PHASE_PROPULSION:
                prop_start_t, prop_end_t = np.nan, np.nan
            else:
                prop_start_t = times[p_start] - half_window
                prop_end_t = times[p_end] + half_window

            # Unweighting (CMJ)
            pred_unw_start, pred_unw_end = np.nan, np.nan
            if jump_type.upper() == "CMJ":
                u_end = p_start - 1
                u_start = u_end
                while u_start >= 0 and phases[u_start] == PHASE_UNWEIGHTING:
                    u_start -= 1
                u_start += 1

                if u_end >= 0 and u_start <= u_end and phases[u_end] == PHASE_UNWEIGHTING:
                    pred_unw_start = times[u_start] - half_window
                    pred_unw_end = times[u_end] + half_window

            # Absorption (DJ)
            pred_abs_start, pred_abs_end = np.nan, np.nan
            if jump_type.upper() == "DJ":
                a_end = p_start - 1
                a_start = a_end
                while a_start >= 0 and phases[a_start] == PHASE_ABSORPTION:
                    a_start -= 1
                a_start += 1

                if a_end >= 0 and a_start <= a_end and phases[a_end] == PHASE_ABSORPTION:
                    pred_abs_start = times[a_start] - half_window
                    pred_abs_end = times[a_end] + half_window

            # Landing
            l_start = f_end + 1
            l_end = l_start
            while l_end < len(phases) and phases[l_end] == PHASE_LANDING:
                l_end += 1
            l_end -= 1

            if l_start >= len(phases) or l_start > l_end or phases[l_start] != PHASE_LANDING:
                land_start_t, land_end_t = np.nan, np.nan
            else:
                land_start_t = times[l_start] - half_window
                land_end_t = times[l_end] + half_window

            rows.append({
                "Athlete": athlete,
                "Jump_Type": jump_type,
                "Trial": trial,
                "Jump": jump_count,
                "Pred_Absorption_Start": pred_abs_start,
                "Pred_Absorption_End": pred_abs_end,
                "Pred_Unweighting_Start": pred_unw_start,
                "Pred_Unweighting_End": pred_unw_end,
                "Pred_Propulsion_Start": prop_start_t,
                "Pred_Propulsion_End": prop_end_t,
                "Pred_Landing_Start": land_start_t,
                "Pred_Landing_End": land_end_t,
            })
            jump_count += 1

    return pd.DataFrame(rows)


def prepare_ground_truth_wide(gt_long):
    """
    Converts user's long-format ground truth CSV into wide-format columns.
    Filters out 'waiting' phases (Jump == 0).
    """
    gt_long = gt_long[gt_long["Jump"] > 0]
    gt_wide_records = []

    for (athlete, j_type, trial, jump), group in gt_long.groupby(["Athlete", "Jump_Type", "Trial", "Jump"]):
        record = {
            "Athlete": athlete,
            "Jump_Type": j_type,
            "Trial": trial,
            "Jump": jump,
            "Absorption_Start_sn": np.nan,
            "Unweighting_Start_sn": np.nan,
            "Propulsion_Start_sn": np.nan,
            "Flight_Start_sn": np.nan,
            "Landing_Start_sn": np.nan,
            "Jump_End_sn": np.nan,
        }
        for _, row in group.iterrows():
            phase = row["Phase"]
            if phase == "Absorption":
                record["Absorption_Start_sn"] = row["Start_Time_sn"]
            elif phase == "Unweighting":
                record["Unweighting_Start_sn"] = row["Start_Time_sn"]
            elif phase == "Propulsion":
                record["Propulsion_Start_sn"] = row["Start_Time_sn"]
            elif phase == "Flight":
                record["Flight_Start_sn"] = row["Start_Time_sn"]
            elif phase == "Landing":
                record["Landing_Start_sn"] = row["Start_Time_sn"]
                record["Jump_End_sn"] = row["End_Time_sn"]

        gt_wide_records.append(record)

    return pd.DataFrame(gt_wide_records)


def main():
    print("Full-recording training data loading...")
    df = pd.read_csv("full_recording_ml_training_data.csv")

    gt_csv = "imu_faz_zamanlari_tum_recording.csv"
    print(f"Ground truth data loading from {gt_csv} and reshaping to wide format...")
    gt_raw = pd.read_csv(gt_csv)
    ground_truth = prepare_ground_truth_wide(gt_raw)

    features = [c for c in df.columns if not c.startswith("Meta_") and c != "Phase"]
    print(f"Number of features used: {len(features)}")

    print("Collecting full-recording predictions via LOSO + 6-state HMM Viterbi...")
    preds_df = run_loso_full_recording(df, features)

    print("Extracting predicted phase segment boundaries (start/end)...")
    segments_df = extract_predicted_segments(preds_df)

    print("Merging with ground truth force plate timestamps...")
    segments_df["Trial"] = segments_df["Trial"].astype(int)
    segments_df["Jump"] = segments_df["Jump"].astype(int)
    ground_truth["Trial"] = ground_truth["Trial"].astype(int)
    ground_truth["Jump"] = ground_truth["Jump"].astype(int)

    merged = pd.merge(
        segments_df,
        ground_truth,
        on=["Athlete", "Jump_Type", "Trial", "Jump"],
        how="left",
        indicator=True,
    )

    unmatched = merged[merged["_merge"] != "both"]
    if len(unmatched) > 0:
        print(f"\nWARNING: {len(unmatched)} jumps could not be matched with ground truth:")
        print(unmatched[["Athlete", "Jump_Type", "Trial", "Jump"]].to_string(index=False))
    merged = merged[merged["_merge"] == "both"].drop(columns=["_merge"])

    # Absolute timing errors in milliseconds
    merged["Err_Absorption_Start_ms"] = (
        (merged["Pred_Absorption_Start"] - merged["Absorption_Start_sn"]).abs() * 1000
        if "Absorption_Start_sn" in merged.columns else np.nan
    )
    merged["Err_Unweighting_Start_ms"] = (
        (merged["Pred_Unweighting_Start"] - merged["Unweighting_Start_sn"]).abs() * 1000
        if "Unweighting_Start_sn" in merged.columns else np.nan
    )
    merged["Err_Propulsion_Start_ms"] = (
        (merged["Pred_Propulsion_Start"] - merged["Propulsion_Start_sn"]).abs() * 1000
        if "Propulsion_Start_sn" in merged.columns else np.nan
    )
    merged["Err_Propulsion_End_ms"] = (
        (merged["Pred_Propulsion_End"] - merged["Flight_Start_sn"]).abs() * 1000
        if "Flight_Start_sn" in merged.columns else np.nan
    )
    merged["Err_Landing_Start_ms"] = (
        (merged["Pred_Landing_Start"] - merged["Landing_Start_sn"]).abs() * 1000
        if "Landing_Start_sn" in merged.columns else np.nan
    )
    merged["Err_Landing_End_ms"] = (
        (merged["Pred_Landing_End"] - merged["Jump_End_sn"]).abs() * 1000
        if "Jump_End_sn" in merged.columns else np.nan
    )

    merged.to_csv("zamanlama_degerlendirme_sonuclari.csv", index=False)

    print("\n" + "=" * 70)
    print("   MISSING SEGMENT RATES (Model predicted no instances for phase)")
    print("=" * 70)
    total = len(merged)

    # Absorption check (DJ)
    dj_sub = merged[merged["Jump_Type"] == "DJ"]
    missing_abs = dj_sub["Pred_Absorption_Start"].isna().sum()
    n_dj = len(dj_sub)
    print(f"  Absorption (DJ): {missing_abs}/{n_dj} jumps missing ({100 * missing_abs / n_dj if n_dj else 0:.1f}%)")

    # Unweighting check (CMJ)
    cmj_sub = merged[merged["Jump_Type"] == "CMJ"]
    missing_unw = cmj_sub["Pred_Unweighting_Start"].isna().sum()
    n_cmj = len(cmj_sub)
    print(f"  Unweighting (CMJ): {missing_unw}/{n_cmj} jumps missing ({100 * missing_unw / n_cmj if n_cmj else 0:.1f}%)")

    # Propulsion & Landing check
    for col, label in [
        ("Pred_Propulsion_Start", "Propulsion"),
        ("Pred_Landing_Start", "Landing"),
    ]:
        missing = merged[col].isna().sum()
        print(f"  {label}: {missing}/{total} jumps missing ({100 * missing / total:.1f}%)")

    print("\n" + "=" * 70)
    print("   TIMING ERROR SUMMARY (Mean Absolute Error, ms)")
    print("=" * 70)
    err_cols = [
        "Err_Absorption_Start_ms",
        "Err_Unweighting_Start_ms",
        "Err_Propulsion_Start_ms",
        "Err_Propulsion_End_ms",
        "Err_Landing_Start_ms",
        "Err_Landing_End_ms",
    ]
    print(merged[err_cols].mean().round(1))

    print("\n--- Breakdown by Jump Type ---")
    print(merged.groupby("Jump_Type")[err_cols].mean().round(1))

    print("\nDetailed results saved to 'zamanlama_degerlendirme_sonuclari.csv'.")


if __name__ == "__main__":
    main()