"""
train.py

Senkronize (etiketli) kişilerin labeled full-recording feature datasıyla
CMJ, SJ ve DJ için ayrı CatBoost + empirical HMM modelleri eğitir.

Sonra senkronizasyonu bilinmeyen (etiketsiz) kişilerin unlabeled full-recording
feature datasında phase prediction yapar.

Output:
    predicted_phases.csv
"""

import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
TRAIN_PATH = "full_recording_ml_training_data.csv"
TEST_PATH = "unlabeled_full_recording_features.csv"
OUTPUT_PATH = "predicted_phases.csv"

N_STATES = 6
PHASE_NAMES = {
    0: "Unweighting",
    1: "Propulsion",
    2: "Flight",
    3: "Landing",
    4: "Absorption",
    5: "waiting",
}


# ============================================================
# 1. EMPIRICAL HMM
# ============================================================
def estimate_empirical_transition_matrix(data_df):
    """
    Full recording sequence'lerinden HMM transition matrix
    ve initial-state probabilities öğrenir.

    Sequence: Athlete + JumpType + Trial
    """
    transition_counts = np.zeros((N_STATES, N_STATES), dtype=float)
    start_counts = np.zeros(N_STATES, dtype=float)
    group_cols = ["Meta_Athlete", "Meta_JumpType", "Meta_Trial"]

    for _, group in data_df.groupby(group_cols, dropna=False):
        group = group.sort_values("Meta_Time")
        phases = group["Phase"].astype(int).values
        if len(phases) == 0:
            continue

        start_counts[phases[0]] += 1.0

        for i in range(len(phases) - 1):
            curr_state = phases[i]
            next_state = phases[i + 1]
            if 0 <= curr_state < N_STATES and 0 <= next_state < N_STATES:
                transition_counts[curr_state, next_state] += 1.0

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    trans_matrix = np.divide(
        transition_counts,
        row_sums,
        out=np.zeros_like(transition_counts),
        where=row_sums != 0,
    )

    for state in range(N_STATES):
        if trans_matrix[state].sum() == 0:
            trans_matrix[state, state] = 1.0

    start_sum = start_counts.sum()
    start_probs = start_counts / start_sum if start_sum > 0 else np.ones(N_STATES) / N_STATES
    return trans_matrix, start_probs


# ============================================================
# 2. VITERBI
# ============================================================
def apply_viterbi_filter(probabilities, trans_matrix, start_probs):
    """
    Viterbi algoritması ile en olası faz dizisini bulur.
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
            scores = viterbi_table[t - 1, :] + log_trans[:, state]
            best_prev = np.argmax(scores)
            backpointer[t, state] = best_prev
            viterbi_table[t, state] = scores[best_prev] + log_probs[t, state]

    path = np.zeros(n_samples, dtype=int)
    path[-1] = np.argmax(viterbi_table[-1])

    for t in range(n_samples - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]

    return path


# ============================================================
# 3. MAIN
# ============================================================
def main():
    print("=" * 80)
    print("FULL RECORDING TRAINING -> INFERENCE (CMJ + SJ + DJ)")
    print("=" * 80)

    # LOAD
    print("\nLoading labeled training data...")
    train_df = pd.read_csv(TRAIN_PATH)

    print("Loading unlabeled inference data...")
    test_df = pd.read_csv(TEST_PATH)

    # FEATURE CHECK
    train_features = [c for c in train_df.columns if not c.startswith("Meta_") and c != "Phase"]
    test_features = [c for c in test_df.columns if not c.startswith("Meta_")]

    missing_in_test = [c for c in train_features if c not in test_features]
    extra_in_test = [c for c in test_features if c not in train_features]

    if missing_in_test:
        raise ValueError(f"Test datasında eksik feature var:\n{missing_in_test}")
    if extra_in_test:
        raise ValueError(f"Test datasında training'de olmayan fazladan feature var:\n{extra_in_test}")

    features = train_features
    print(f"Feature count: {len(features)}")

    all_predictions = []

    # CMJ / SJ / DJ SPECIFIC MODELS
    for jump_type in ["CMJ", "SJ", "DJ"]:
        print("\n" + "=" * 70)
        print(f"TRAINING FINAL {jump_type} MODEL")
        print("=" * 70)

        train_type = train_df[train_df["Meta_JumpType"].str.upper() == jump_type].copy()
        test_type = test_df[test_df["Meta_JumpType"].str.upper() == jump_type].copy()

        if train_type.empty:
            print(f"No training data for {jump_type}")
            continue
        if test_type.empty:
            print(f"No inference data for {jump_type}")
            continue

        # HMM from all labeled training recordings
        trans_matrix, start_probs = estimate_empirical_transition_matrix(train_type)

        print("\nLearned HMM transition matrix:")
        print(
            pd.DataFrame(
                trans_matrix,
                index=[PHASE_NAMES[i] for i in range(N_STATES)],
                columns=[PHASE_NAMES[i] for i in range(N_STATES)],
            ).round(4)
        )

        # CatBoost train
        le = LabelEncoder()
        y_train = le.fit_transform(train_type["Phase"])

        model = CatBoostClassifier(iterations=500, depth=6, verbose=0, random_seed=42)

        print(f"\nTraining CatBoost on {len(train_type)} windows...")
        model.fit(train_type[features], y_train)

        print(f"Predicting {len(test_type)} windows...")

        # EACH RECORDING SEPARATELY
        recording_group_cols = ["Meta_Athlete", "Meta_Trial"]
        for (athlete, trial), group in test_type.groupby(recording_group_cols, dropna=False):
            group = group.sort_values("Meta_Time").copy()
            X_group = group[features]

            raw_probs = model.predict_proba(X_group)

            # N_STATES probability matrix
            probs = np.zeros((len(group), N_STATES), dtype=float)
            for idx, encoded_class in enumerate(model.classes_):
                original_class = int(le.inverse_transform([int(encoded_class)])[0])
                probs[:, original_class] = raw_probs[:, idx]

            phase_pred = apply_viterbi_filter(probs, trans_matrix, start_probs)

            result = group[["Meta_Athlete", "Meta_JumpType", "Meta_Trial", "Meta_Time"]].copy()
            result["Phase_Pred"] = phase_pred
            result["Phase_Name"] = [PHASE_NAMES[int(p)] for p in phase_pred]

            all_predictions.append(result)

            counts = pd.Series(phase_pred).value_counts().sort_index()
            count_text = ", ".join([f"{PHASE_NAMES[int(k)]}={v}" for k, v in counts.items()])
            print(f"  {athlete:20s} {jump_type}_{int(trial)} | {count_text}")

    # SAVE
    if not all_predictions:
        print("\nNo predictions generated.")
        return

    predicted_df = pd.concat(all_predictions, ignore_index=True)
    predicted_df = predicted_df.sort_values(
        ["Meta_Athlete", "Meta_JumpType", "Meta_Trial", "Meta_Time"]
    )
    predicted_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETED")
    print("=" * 80)
    print(f"Total predicted windows: {len(predicted_df)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()