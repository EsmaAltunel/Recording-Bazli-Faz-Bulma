"""
matris.py
=========
Full-recording phase classification evaluation using Machine Learning + HMM Viterbi Decoding.
Desteklenen Zıplama Tipleri: CMJ, SJ, DJ

Classes (6 states):
    0 -> Unweighting (CMJ)
    1 -> Propulsion (CMJ, SJ, DJ)
    2 -> Flight (CMJ, SJ, DJ)
    3 -> Landing (CMJ, SJ, DJ)
    4 -> Absorption (DJ)
    5 -> Background / waiting

Evaluation Method:
    - LOSO (Leave-One-Subject-Out) cross validation.
    - HMM transition matrices estimated exclusively from training folds.
    - Continuous Viterbi decoding applied per complete recording sequence (Meta_Athlete, Meta_JumpType, Meta_Trial).

Outputs:
    - full_recording_loso_fold_results.csv
    - full_recording_phase_accuracy.csv
    - confusion_matrix_full_recording.png
    - confusion_matrix_full_CMJ.png
    - confusion_matrix_full_SJ.png
    - confusion_matrix_full_DJ.png
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 1. DATASET & CONFIGURATION
# ============================================================
print("Loading full-recording dataset...")
df = pd.read_csv("full_recording_ml_training_data.csv")

# Identify feature columns (exclude metadata and target)
features = [c for c in df.columns if not c.startswith("Meta_") and c != "Phase"]
print(f"Number of features used: {len(features)}")

athletes = df["Meta_Athlete"].unique()
jump_types = df["Meta_JumpType"].unique()

# Phase Definitions (6 States)
N_STATES = 6
PHASE_NAMES = ["Unweighting", "Propulsion", "Flight", "Landing", "Absorption", "waiting"]
ALL_LABELS = [0, 1, 2, 3, 4, 5]

print("\nClass distribution:")
for phase_id, count in df["Phase"].value_counts().sort_index().items():
    print(f"  {phase_id} ({PHASE_NAMES[int(phase_id)]}): {count}")


# ============================================================
# 2. HMM TRANSITION MATRIX ESTIMATION
# ============================================================
def estimate_empirical_transition_matrix(data_df):
    """
    Estimates state transition probabilities and initial state probabilities 
    from complete recording sequences (Meta_Athlete + Meta_JumpType + Meta_Trial).
    """
    transition_counts = np.zeros((N_STATES, N_STATES), dtype=float)
    start_counts = np.zeros(N_STATES, dtype=float)
    group_cols = ["Meta_Athlete", "Meta_JumpType", "Meta_Trial"]

    for _, group in data_df.groupby(group_cols, dropna=False):
        group = group.sort_values("Meta_Time")
        phases = group["Phase"].astype(int).values
        if len(phases) == 0:
            continue

        first_state = int(phases[0])
        if 0 <= first_state < N_STATES:
            start_counts[first_state] += 1

        for i in range(len(phases) - 1):
            curr_state = int(phases[i])
            next_state = int(phases[i + 1])
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


# ============================================================
# 3. VITERBI DECODING
# ============================================================
def apply_viterbi_filter(probabilities, trans_matrix, start_probs):
    """
    Finds the most probable phase sequence over a full recording using Viterbi algorithm.
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
        for state in range(n_states):
            transition_scores = viterbi_table[t - 1, :] + log_trans[:, state]
            best_prev = np.argmax(transition_scores)
            backpointer[t, state] = best_prev
            viterbi_table[t, state] = transition_scores[best_prev] + log_probs[t, state]

    smoothed_path = np.zeros(n_samples, dtype=int)
    smoothed_path[-1] = np.argmax(viterbi_table[-1, :])

    for t in range(n_samples - 2, -1, -1):
        smoothed_path[t] = backpointer[t + 1, smoothed_path[t + 1]]

    return smoothed_path


# ============================================================
# 4. MODEL FACTORY
# ============================================================
def get_models():
    return {
        "CatBoost": CatBoostClassifier(iterations=500, depth=6, verbose=0, random_seed=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0),
    }


# ============================================================
# 5. GLOBAL HMM TRANSITION MATRIX SUMMARY
# ============================================================
print("\n" + "=" * 80)
print(" GLOBAL HMM TRANSITION MATRIX FOR FULL RECORDINGS ")
print("=" * 80)

global_trans_matrix, global_start_probs = estimate_empirical_transition_matrix(df)
trans_df = pd.DataFrame(global_trans_matrix, index=PHASE_NAMES, columns=PHASE_NAMES)

print("\nInitial State Probabilities:")
for phase_name, prob in zip(PHASE_NAMES, global_start_probs):
    print(f"  {phase_name:12s}: {prob:.4f}")

print("\nState Transition Matrix (Row = Current, Column = Next):")
print(trans_df.round(4))
print("=" * 80)


# ============================================================
# 6. LOSO CROSS VALIDATION
# ============================================================
all_results = []
strategy_jump_preds = {("General", jt): {"true": [], "pred": []} for jt in jump_types}
strategy_jump_preds.update({("Specific", jt): {"true": [], "pred": []} for jt in jump_types})

final_true = []
final_preds = []

print("\nStarting Full-Recording LOSO evaluation...")

for strategy in ["General", "Specific"]:
    print(f"\n>>> {strategy.upper()} STRATEGY")
    loop_types = jump_types if strategy == "Specific" else ["All"]

    for j_type in loop_types:
        if strategy == "Specific":
            current_df = df[df["Meta_JumpType"] == j_type].copy().reset_index(drop=True)
        else:
            current_df = df.copy().reset_index(drop=True)

        for test_athlete in athletes:
            train_data = current_df[current_df["Meta_Athlete"] != test_athlete].copy()
            test_data = current_df[current_df["Meta_Athlete"] == test_athlete].copy().reset_index(drop=True)

            if train_data.empty or test_data.empty:
                continue

            # Estimate HMM from training fold only
            transition_matrix, start_probs = estimate_empirical_transition_matrix(train_data)

            X_train = train_data[features]
            y_raw_train = train_data["Phase"]
            X_test = test_data[features]
            y_raw_test = test_data["Phase"]

            le = LabelEncoder()
            y_train = le.fit_transform(y_raw_train)
            models = get_models()

            for model_name, model in models.items():
                try:
                    model.fit(X_train, y_train)

                    test_data["Temp_Index"] = np.arange(len(test_data))
                    test_data["Phase_Pred"] = -1

                    # Group test data by complete recording
                    group_cols = ["Meta_JumpType", "Meta_Trial"]
                    for _, group in test_data.groupby(group_cols, dropna=False):
                        if group.empty:
                            continue

                        group = group.sort_values("Meta_Time")
                        g_indices = group["Temp_Index"].values
                        g_X = group[features]

                        raw_probs = model.predict_proba(g_X)
                        probs = np.zeros((len(g_X), N_STATES))
                        for idx, original_class in enumerate(le.classes_):
                            probs[:, int(original_class)] = raw_probs[:, idx]

                        smoothed_preds = apply_viterbi_filter(probs, transition_matrix, start_probs)
                        test_data.loc[g_indices, "Phase_Pred"] = smoothed_preds

                    preds = test_data["Phase_Pred"].astype(int).values
                    y_test = y_raw_test.astype(int).values

                    # Store predictions for CatBoost analysis & confusion matrices
                    if model_name == "CatBoost":
                        if strategy == "General":
                            for jt in jump_types:
                                mask = test_data["Meta_JumpType"] == jt
                                if mask.any():
                                    strategy_jump_preds[("General", jt)]["true"].extend(y_test[mask])
                                    strategy_jump_preds[("General", jt)]["pred"].extend(preds[mask])
                        else:
                            strategy_jump_preds[("Specific", j_type)]["true"].extend(y_test)
                            strategy_jump_preds[("Specific", j_type)]["pred"].extend(preds)
                            final_true.extend(y_test)
                            final_preds.extend(preds)

                    # Metrics calculation per fold
                    if strategy == "General":
                        for jt in jump_types:
                            mask = test_data["Meta_JumpType"] == jt
                            if not mask.any():
                                continue
                            weighted_f1 = f1_score(y_test[mask], preds[mask], average="weighted", zero_division=0)
                            macro_f1 = f1_score(y_test[mask], preds[mask], average="macro", zero_division=0)
                            all_results.append({
                                "Strategy": "General",
                                "Model": model_name,
                                "Jump_Type": jt,
                                "Athlete": test_athlete,
                                "Weighted_F1": weighted_f1,
                                "Macro_F1": macro_f1,
                            })
                    else:
                        weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
                        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
                        all_results.append({
                            "Strategy": "Specific",
                            "Model": model_name,
                            "Jump_Type": j_type,
                            "Athlete": test_athlete,
                            "Weighted_F1": weighted_f1,
                            "Macro_F1": macro_f1,
                        })

                    print(f"{strategy:8s} | {j_type:4s} | {model_name:12s} | Test: {test_athlete:18s} | W-F1={weighted_f1:.4f} | M-F1={macro_f1:.4f}")

                except Exception as e:
                    print(f"ERROR ({strategy} - {j_type} - {model_name} - {test_athlete}): {e}")


# ============================================================
# 7. RESULTS TABLES
# ============================================================
res_df = pd.DataFrame(all_results)
if res_df.empty:
    print("\nResults could not be generated.")
    raise SystemExit

res_df.to_csv("full_recording_loso_fold_results.csv", index=False)

weighted_report = res_df.groupby(["Model", "Strategy", "Jump_Type"])["Weighted_F1"].mean()
print("\n" + "=" * 90)
print(" FULL RECORDING MODEL COMPARISON (MEAN WEIGHTED F1) ")
print("=" * 90)
print(weighted_report.round(4))
print("=" * 90)

macro_report = res_df.groupby(["Model", "Strategy", "Jump_Type"])["Macro_F1"].mean()
print("\n" + "=" * 90)
print(" FULL RECORDING MODEL COMPARISON (MEAN MACRO F1) ")
print("=" * 90)
print(macro_report.round(4))
print("=" * 90)


# ============================================================
# 8. CONFUSION MATRICES
# ============================================================
sns.set_theme(style="white")
mat_kwargs = {
    "annot": True,
    "fmt": ".2%",
    "cmap": "Blues",
    "annot_kws": {"size": 9, "weight": "bold"},
    "linewidths": 1,
    "linecolor": "white",
    "cbar_kws": {"label": "Prediction Rate"},
}

# 1. Overall Confusion Matrix (CatBoost Specific CMJ + SJ + DJ)
if len(final_true) > 0:
    cm = confusion_matrix(final_true, final_preds, labels=ALL_LABELS)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, xticklabels=PHASE_NAMES, yticklabels=PHASE_NAMES, **mat_kwargs)
    plt.title("CatBoost + HMM — Full Recordings (CMJ + SJ + DJ)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Model Prediction (IMU/EMG + Viterbi)", fontsize=12)
    plt.ylabel("True Phase", fontsize=12)
    plt.xticks(rotation=20)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_full_recording.png", dpi=300)
    plt.close()

# 2. Jump-Specific Confusion Matrices (CMJ / SJ / DJ)
for jt in jump_types:
    data = strategy_jump_preds[("Specific", jt)]
    if len(data["true"]) == 0:
        continue

    cm_j = confusion_matrix(data["true"], data["pred"], labels=ALL_LABELS)
    cm_j_norm = cm_j.astype(float) / (cm_j.sum(axis=1, keepdims=True) + 1e-10)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_j_norm, xticklabels=PHASE_NAMES, yticklabels=PHASE_NAMES, **mat_kwargs)
    plt.title(f"CatBoost + HMM — {jt} Full Recording", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Model Prediction (IMU/EMG + Viterbi)", fontsize=12)
    plt.ylabel("True Phase", fontsize=12)
    plt.xticks(rotation=20)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_full_{jt}.png", dpi=300)
    plt.close()


# ============================================================
# 9. PHASE-WISE ACCURACY SUMMARY
# ============================================================
matrix_summary = []
for (strategy, jt), data in strategy_jump_preds.items():
    if len(data["true"]) == 0:
        continue

    cm_j = confusion_matrix(data["true"], data["pred"], labels=ALL_LABELS)
    cm_j_norm = cm_j.astype(float) / (cm_j.sum(axis=1, keepdims=True) + 1e-10)

    for i, phase_name in enumerate(PHASE_NAMES):
        # Skip phases irrelevant to specific jump types
        if jt == "SJ" and phase_name in ["Unweighting", "Absorption"]:
            continue
        if jt == "CMJ" and phase_name == "Absorption":
            continue
        if jt == "DJ" and phase_name == "Unweighting":
            continue

        row_total = cm_j[i].sum()
        if row_total == 0:
            continue

        acc_val = cm_j_norm[i, i]
        matrix_summary.append({
            "Strategy": strategy,
            "Jump_Type": jt,
            "Phase": phase_name,
            "Accuracy_%": round(acc_val * 100, 2),
        })

summary_df = pd.DataFrame(matrix_summary)
if not summary_df.empty:
    pivot_summary = summary_df.pivot_table(
        index=["Jump_Type", "Phase"],
        columns="Strategy",
        values="Accuracy_%",
    )
    print("\n" + "=" * 90)
    print(" PHASE-WISE ACCURACY — FULL RECORDINGS (%) ")
    print("=" * 90)
    print(pivot_summary)
    print("=" * 90)
    pivot_summary.to_csv("full_recording_phase_accuracy.csv")


# ============================================================
# 10. GENERAL MODEL FALSE PHASE DETECTIONS CHECK
# ============================================================
if ("General", "SJ") in strategy_jump_preds:
    sj_data = strategy_jump_preds[("General", "SJ")]
    sj_preds = np.asarray(sj_data["pred"])

    if len(sj_preds) > 0:
        false_unw = np.sum(sj_preds == 0)
        false_abs = np.sum(sj_preds == 4)
        total_sj = len(sj_preds)

        print("\n" + "=" * 80)
        print(" GENERAL MODEL — SJ FALSE PHASE CHECK ")
        print("=" * 80)
        print(f"Total SJ windows: {total_sj}")
        print(f"False Unweighting: {false_unw} ({false_unw / total_sj * 100:.2f}%)")
        print(f"False Absorption: {false_abs} ({false_abs / total_sj * 100:.2f}%)")
        print("=" * 80)

print("\nFull-recording LOSO evaluation completed successfully.")