import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = "datasets/processed/features.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "final_hierarchical_model.pkl"
)

RANDOM_STATE = 42


LABELS = [
    "RUPEE_1",
    "RUPEE_2",
    "RUPEE_5",
    "RUPEE_10",
    "RUPEE_20"
]


# ============================================================
# LOAD FEATURES
# ============================================================

print("=" * 60)
print("FINAL MODEL TRAINING")
print("=" * 60)
print()

df = pd.read_csv(FEATURE_FILE)

print(f"Loaded events: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print()


# ============================================================
# KEEP DENOMINATION EVENTS
# ============================================================

df = df[
    df["label"].isin(LABELS)
].copy()

print("Training classes:")
print(df["label"].value_counts())
print()


# ============================================================
# SELECT ML FEATURES
# ============================================================

metadata_columns = {
    "label",
    "file",
    "event_number",
    "event_time_seconds",
    "sampling_rate_hz"
}

feature_columns = []

for column in df.columns:

    if column in metadata_columns:
        continue

    if pd.api.types.is_numeric_dtype(df[column]):
        feature_columns.append(column)


print(
    f"ML features: {len(feature_columns)}"
)

print()


# ============================================================
# CLEAN FEATURES
# ============================================================

X_df = df[feature_columns].copy()

X_df = X_df.replace(
    [np.inf, -np.inf],
    np.nan
)

X_df = X_df.fillna(
    X_df.median()
)

# Convert to NumPy.
#
# This avoids the Python 3.14 / Narwhals problem.

X = X_df.to_numpy(
    dtype=np.float64
)

y = df["label"].to_numpy()


# ============================================================
# HIERARCHICAL LABELS
# ============================================================

def make_group(label):

    if label in [
        "RUPEE_1",
        "RUPEE_2",
        "RUPEE_5"
    ]:
        return "LOW"

    return "HIGH"


group_y = np.array([
    make_group(label)
    for label in y
])


# ============================================================
# MODEL CREATION
# ============================================================

def make_rf():

    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )


# ============================================================
# LEVEL 1
#
# LOW  = ₹1 / ₹2 / ₹5
# HIGH = ₹10 / ₹20
# ============================================================

print("Training LEVEL 1 classifier...")

group_model = make_rf()

group_model.fit(
    X,
    group_y
)

print("Level 1 complete.")
print()


# ============================================================
# LEVEL 2A
#
# ₹1 / ₹2 / ₹5
# ============================================================

print("Training LOW denomination classifier...")

low_mask = np.isin(
    y,
    [
        "RUPEE_1",
        "RUPEE_2",
        "RUPEE_5"
    ]
)

low_model = make_rf()

low_model.fit(
    X[low_mask],
    y[low_mask]
)

print("LOW classifier complete.")
print()


# ============================================================
# LEVEL 2B
#
# ₹10 / ₹20
# ============================================================

print("Training HIGH denomination classifier...")

high_mask = np.isin(
    y,
    [
        "RUPEE_10",
        "RUPEE_20"
    ]
)

high_model = make_rf()

high_model.fit(
    X[high_mask],
    y[high_mask]
)

print("HIGH classifier complete.")
print()


# ============================================================
# SAVE EVERYTHING
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


model_package = {

    "group_model": group_model,

    "low_model": low_model,

    "high_model": high_model,

    "feature_columns": feature_columns,

    "labels": LABELS,

    "random_state": RANDOM_STATE
}


joblib.dump(
    model_package,
    MODEL_FILE
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("FINAL MODEL READY")
print("=" * 60)
print()

print(
    f"Training events: {len(X)}"
)

print(
    f"Features: {len(feature_columns)}"
)

print()

print("Models:")
print("  LOW / HIGH")
print("  ₹1 / ₹2 / ₹5")
print("  ₹10 / ₹20")

print()

print(
    f"Saved to:\n  {MODEL_FILE}"
)

print()
