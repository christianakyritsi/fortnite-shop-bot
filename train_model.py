import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# ============ Load ============
df = pd.read_csv("training_data.csv", parse_dates=["ref_date"])
print(f"Total rows: {len(df):,}")

# ============ Temporal split ============
TRAIN_END = pd.Timestamp("2024-06-01", tz="UTC")
VAL_END = pd.Timestamp("2025-06-01", tz="UTC")

train = df[df["ref_date"] < TRAIN_END]
val = df[(df["ref_date"] >= TRAIN_END) & (df["ref_date"] < VAL_END)]
test = df[df["ref_date"] >= VAL_END]

print(f"Train: {len(train):,} rows ({train['returned_within_30d'].mean():.3f} positive rate)")
print(f"Val:   {len(val):,} rows ({val['returned_within_30d'].mean():.3f} positive rate)")
print(f"Test:  {len(test):,} rows ({test['returned_within_30d'].mean():.3f} positive rate)")

# ============ Features ============
NUM_FEATURES = [
    "days_since_last", "total_appearances_so_far", "item_age_days",
    "mean_gap", "median_gap", "std_gap",
    "appearances_last_90d", "appearances_last_365d",
    "month", "day_of_week",
]
CAT_FEATURES = ["rarity", "type"]

# One-hot encode categoricals
def prep(df):
    X = df[NUM_FEATURES].copy()
    X = pd.concat([X, pd.get_dummies(df["rarity"], prefix="rarity"), pd.get_dummies(df["type"], prefix="type")], axis=1)
    y = df["returned_within_30d"]
    return X, y

X_train, y_train = prep(train)
X_val, y_val = prep(val)
X_test, y_test = prep(test)

# Align columns (val/test might have categories that train doesn't, or vice versa)
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"\nFeatures: {X_train.shape[1]}")

# ============ Baseline 0: Predict the global positive rate ============
print("\n" + "=" * 50)
print("BASELINE 0: Predict global positive rate")
print("=" * 50)
baseline_rate = y_train.mean()
preds_b0 = np.full(len(y_test), baseline_rate)
print(f"Constant prediction: {baseline_rate:.3f}")
print(f"Test AUC: 0.500 (always — by definition)")
print(f"Test Brier score: {brier_score_loss(y_test, preds_b0):.4f}")

# ============ Baseline 1: Logistic Regression on days_since_last only ============
print("\n" + "=" * 50)
print("BASELINE 1: Logistic regression on `days_since_last` only")
print("=" * 50)
lr = LogisticRegression()
lr.fit(X_train[["days_since_last"]].fillna(-1), y_train)
preds_b1 = lr.predict_proba(X_test[["days_since_last"]].fillna(-1))[:, 1]
print(f"Test AUC: {roc_auc_score(y_test, preds_b1):.4f}")
print(f"Test Brier: {brier_score_loss(y_test, preds_b1):.4f}")

# ============ Main model: LightGBM ============
print("\n" + "=" * 50)
print("MODEL: LightGBM")
print("=" * 50)
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=64,
    min_child_samples=50,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
)

preds_test = model.predict_proba(X_test)[:, 1]
print(f"\nTest AUC: {roc_auc_score(y_test, preds_test):.4f}")
print(f"Test Brier: {brier_score_loss(y_test, preds_test):.4f}")

print("\nClassification report (threshold=0.5):")
print(classification_report(y_test, (preds_test > 0.5).astype(int), digits=3))

# Feature importance
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nTop 15 features:")
print(importance.head(15).to_string(index=False))

# ============ Save ============
joblib.dump({"model": model, "feature_columns": list(X_train.columns)}, "predictor.joblib")
print("\nSaved predictor.joblib")