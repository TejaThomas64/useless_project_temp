import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = "datasets/processed/features.csv"

RANDOM_STATE = 42
N_SPLITS = 5

LABELS = [
    "RUPEE_1",
    "RUPEE_2",
    "RUPEE_5",
    "RUPEE_10",
    "RUPEE_20"
]


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("HIERARCHICAL RANDOM FOREST")
print("=" * 60)
print()

df = pd.read_csv(FEATURE_FILE)

print(f"Loaded dataset: {len(df)} events")
print(f"Total columns:  {len(df.columns)}")
print()

print("Original classes:")
print(df["label"].value_counts())
print()


# ============================================================
# CREATE EVENT GROUPS
# ============================================================

# V1 contains 150 events total.
#
# We don't have recording IDs in features.csv.
# Therefore we CANNOT honestly reconstruct recording-level
# groups from this file.
#
# Each event gets its own group for this experiment.

df["recording_group"] = np.arange(len(df))


# ============================================================
# REMOVE BACKGROUND
# ============================================================

df = df[
    df["label"].isin(LABELS)
].copy()

print(f"Denomination events: {len(df)}")
print()

print(df["label"].value_counts())
print()


# ============================================================
# HIERARCHICAL LABEL
# ============================================================

def make_group(label):

    if label in [
        "RUPEE_1",
        "RUPEE_2",
        "RUPEE_5"
    ]:
        return "LOW"

    return "HIGH"


df["hier_group"] = df["label"].apply(make_group)


# ============================================================
# FEATURE SELECTION
# ============================================================

metadata_columns = {
    "label",
    "recording_group",
    "hier_group",
    "event_index",
    "filename",
    "file",
    "source",
    "recording",
    "recording_id",
    "group"
}

feature_columns = []

for column in df.columns:

    if column in metadata_columns:
        continue

    if pd.api.types.is_numeric_dtype(df[column]):
        feature_columns.append(column)


print(f"Numeric features: {len(feature_columns)}")
print()


# ============================================================
# CONVERT TO NUMPY
# ============================================================
#
# IMPORTANT:
# Do this BEFORE sklearn.
#
# This avoids the Python 3.14 / Narwhals / pandas
# feature-name compatibility problem.

X = df[feature_columns].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(X.median())

X = X.to_numpy(dtype=np.float64)

y = df["label"].to_numpy()

hier_y = df["hier_group"].to_numpy()

groups = df["recording_group"].to_numpy()


print(f"X shape: {X.shape}")
print()


# ============================================================
# RANDOM FOREST
# ============================================================

def make_rf():

    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_scores = []

all_true = []
all_pred = []


print("=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)
print()


for fold, (train_idx, test_idx) in enumerate(
    cv.split(
        X,
        y,
        groups
    ),
    start=1
):

    print(f"FOLD {fold}")
    print("-" * 40)


    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    hier_train = hier_y[train_idx]


    # ========================================================
    # LEVEL 1
    # LOW VS HIGH
    # ========================================================

    group_model = make_rf()

    group_model.fit(
        X_train,
        hier_train
    )

    predicted_groups = group_model.predict(
        X_test
    )


    # ========================================================
    # LEVEL 2A
    # ₹1 VS ₹2 VS ₹5
    # ========================================================

    low_mask = np.isin(
        y_train,
        [
            "RUPEE_1",
            "RUPEE_2",
            "RUPEE_5"
        ]
    )

    low_model = make_rf()

    low_model.fit(
        X_train[low_mask],
        y_train[low_mask]
    )


    # ========================================================
    # LEVEL 2B
    # ₹10 VS ₹20
    # ========================================================

    high_mask = np.isin(
        y_train,
        [
            "RUPEE_10",
            "RUPEE_20"
        ]
    )

    high_model = make_rf()

    high_model.fit(
        X_train[high_mask],
        y_train[high_mask]
    )


    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    predictions = []

    for i in range(len(X_test)):

        sample = X_test[i].reshape(1, -1)

        if predicted_groups[i] == "LOW":

            prediction = low_model.predict(
                sample
            )[0]

        else:

            prediction = high_model.predict(
                sample
            )[0]

        predictions.append(prediction)


    predictions = np.array(predictions)


    # ========================================================
    # SCORE
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    fold_scores.append(accuracy)

    all_true.extend(y_test)
    all_pred.extend(predictions)


    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print()


# ============================================================
# RESULTS
# ============================================================

fold_scores = np.array(fold_scores)

mean_accuracy = fold_scores.mean()
std_accuracy = fold_scores.std()


print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print()

for i, score in enumerate(
    fold_scores,
    start=1
):

    print(
        f"Fold {i}: "
        f"{score * 100:.2f}%"
    )

print()

print(
    f"Mean accuracy: "
    f"{mean_accuracy * 100:.2f}%"
)

print(
    f"Std deviation: "
    f"{std_accuracy * 100:.2f}%"
)

print()


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=LABELS
)

cm_df = pd.DataFrame(
    cm,
    index=LABELS,
    columns=LABELS
)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print()

print(cm_df)
print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print()

print(
    classification_report(
        all_true,
        all_pred,
        labels=LABELS,
        zero_division=0
    )
)


# ============================================================
# SAVE
# ============================================================

pd.DataFrame({
    "fold": np.arange(1, N_SPLITS + 1),
    "accuracy": fold_scores
}).to_csv(
    "datasets/processed/hierarchical_results.csv",
    index=False
)

cm_df.to_csv(
    "datasets/processed/hierarchical_confusion_matrix.csv"
)


print("=" * 60)
print("DONE")
print("=" * 60)
print()
