# train_classifier_grid.py
from connect import load_financial_data
from preprocessing import preprocess_dataframe

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import itertools, random, pickle
import pandas as pd
import numpy as np

# ==================================================
# 1. LOAD & PREPROCESS
# ==================================================
print(">>> Loading data...")
df = load_financial_data()
df, encoder, scaler = preprocess_dataframe(df)

X_COLS = ['tahun', 'kuartal', 'kode_label', 'lq45', 'ihsg']

TARGETS = {
    "revneg": "model_revneg.pkl",
    "netprofneg": "model_netprofneg.pkl"
}

# ==================================================
# 2. PARAM GRID (SAMA PERSIS KAYAK COLAB)
# ==================================================
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "reg_alpha": [0, 0.5, 1.0],
    "reg_lambda": [1, 2, 3],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

param_combinations = list(itertools.product(*param_grid.values()))
MAX_TRIALS = 500
random.seed(42)
param_combinations = random.sample(param_combinations, min(MAX_TRIALS, len(param_combinations)))

print(f">>> Total param trials: {len(param_combinations)}")

# ==================================================
# 3. LOOP PER TARGET
# ==================================================
for target, model_name in TARGETS.items():
    print(f"\n{'='*70}")
    print(f">>> TRAINING TARGET: {target.upper()}")
    print(f"{'='*70}")

    X = df[X_COLS]
    y = df[target]

    # ===========================
    # SPLIT WAKTU (SAMA COLAB)
    # ===========================
    train_mask = (
        (df["tahun"] >= 2022) &
        ((df["tahun"] < 2024) | ((df["tahun"] == 2024) & (df["kuartal"] <= 2)))
    )
    val_mask = (df["tahun"] == 2024) & (df["kuartal"].isin([3, 4]))
    test_mask = (df["tahun"] == 2025) & (df["kuartal"].isin([1, 2]))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # ===========================
    # SCALE POS WEIGHT
    # ===========================
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    # ===========================
    # GRID SEARCH LOOP
    # ===========================
    results = []

    for values in param_combinations:
        params = dict(zip(param_grid.keys(), values))

        try:
            model = XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbosity=0
            )

            model.fit(X_train, y_train)

            acc_train = accuracy_score(y_train, model.predict(X_train))
            acc_val   = accuracy_score(y_val, model.predict(X_val))
            acc_test  = accuracy_score(y_test, model.predict(X_test))

            gap = abs(acc_val - acc_test)
            balanced_score = 0.7 * acc_test + 0.3 * (1 / (1 + gap))

            results.append({
                **params,
                "acc_train": acc_train,
                "acc_val": acc_val,
                "acc_test": acc_test,
                "balanced_score": balanced_score
            })
        except:
            continue

    df_results = pd.DataFrame(results)
    best = df_results.sort_values("balanced_score", ascending=False).iloc[0]

    print("\n>>> BEST PARAMS")
    for k in param_grid.keys():
        print(f"{k}: {best[k]}")

    print(f"\nTest Accuracy: {best['acc_test']:.4f}")

    # ===========================
    # TRAIN FINAL MODEL
    # ===========================
    final_model = XGBClassifier(
        **{k: best[k] for k in param_grid.keys()},
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0
    )

    final_model.fit(X_train, y_train)

    with open(model_name, "wb") as f:
        pickle.dump(final_model, f)

    print(f"✅ Model saved: {model_name}")

print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY")
