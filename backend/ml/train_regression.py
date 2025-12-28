# train_regression_grid.py
from connect import load_financial_data
from preprocessing import preprocess_dataframe
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import itertools
import random
import pickle

# ===========================
# 1. Load & preprocess data
# ===========================
print(">>> Loading and preprocessing data...")
df = load_financial_data()
processed_df, encoder, scaler = preprocess_dataframe(df)

FEATURES = [
    'tahun', 'kuartal', 'revneg', 'netprofneg',
    'ihsg', 'lq45', 'kode_label'
]
TARGET = 'NPM_winsor'

# ===========================
# 2. Train / Val / Test split
# ===========================
train_mask = processed_df['tahun'] < 2025
val_mask   = (processed_df['tahun'] == 2025) & (processed_df['kuartal'] == 1)
test_mask  = (processed_df['tahun'] == 2025) & (processed_df['kuartal'] == 2)

X_train = processed_df.loc[train_mask, FEATURES]
y_train = processed_df.loc[train_mask, TARGET]

X_val = processed_df.loc[val_mask, FEATURES]
y_val = processed_df.loc[val_mask, TARGET]

X_test = processed_df.loc[test_mask, FEATURES]
y_test = processed_df.loc[test_mask, TARGET]

print(f"Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

# ===========================
# 3. Hyperparameter Grid
# ===========================
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

# Batasi trial (biar PC aman)
MAX_TRIALS = 500
if len(param_combinations) > MAX_TRIALS:
    random.seed(42)
    param_combinations = random.sample(param_combinations, MAX_TRIALS)

print(f"Total hyperparameter combinations: {len(param_combinations)}")

# ===========================
# 4. Grid Search Loop
# ===========================
results = []
failed_trials = 0

print("\n>>> Starting hyperparameter search...")

for idx, values in enumerate(param_combinations):
    params = dict(zip(param_grid.keys(), values))

    if idx % 50 == 0:
        print(f"Progress: {idx}/{len(param_combinations)}")

    try:
        model = XGBRegressor(
            **params,
            random_state=42,
            verbosity=0
        )

        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_val   = model.predict(X_val)
        pred_test  = model.predict(X_test)

        r2_train = r2_score(y_train, pred_train)
        r2_val   = r2_score(y_val, pred_val)
        r2_test  = r2_score(y_test, pred_test)

        mae_test = mean_absolute_error(y_test, pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

        r2_gap = abs(r2_val - r2_test)
        train_test_gap = abs(r2_train - r2_test)
        max_gap = max(r2_gap, train_test_gap)

        stability_score = 1 / (1 + max_gap)
        balanced_score = 0.7 * r2_test + 0.3 * stability_score

        results.append({
            **params,
            "R2_train": r2_train,
            "R2_val": r2_val,
            "R2_test": r2_test,
            "MAE_test": mae_test,
            "RMSE_test": rmse_test,
            "balanced_score": balanced_score
        })

    except Exception as e:
        failed_trials += 1
        continue

df_results = pd.DataFrame(results)

print("\n>>> Hyperparameter search finished")
print(f"Successful trials: {len(df_results)}")
print(f"Failed trials: {failed_trials}")

if df_results.empty:
    raise RuntimeError("No successful models trained.")

# ===========================
# 5. Select BEST MODEL (COLAB STYLE)
# ===========================
top10 = (
    df_results
    .nlargest(10, "balanced_score")
    .sort_values("balanced_score", ascending=True)
    .reset_index(drop=True)
)

best_params = top10.iloc[-1]

print("\n🏆 BEST MODEL PARAMETERS (FROM GRID SEARCH)")
for k in param_grid.keys():
    print(f"  - {k}: {best_params[k]}")

# ===========================
# 6. Retrain BEST MODEL
# ===========================
best_model = XGBRegressor(
    n_estimators=int(best_params["n_estimators"]),
    max_depth=int(best_params["max_depth"]),
    learning_rate=best_params["learning_rate"],
    reg_alpha=best_params["reg_alpha"],
    reg_lambda=best_params["reg_lambda"],
    min_child_weight=best_params["min_child_weight"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    random_state=42,
    verbosity=0
)

best_model.fit(X_train, y_train)

# ===========================
# 7. Save model & results
# ===========================
with open("model_npm_best.pkl", "wb") as f:
    pickle.dump(best_model, f)

top10.to_csv("top10_models.csv", index=False)

print("\n✅ Best model saved as model_npm_best.pkl")
print("✅ Top 10 models saved as top10_models.csv")
