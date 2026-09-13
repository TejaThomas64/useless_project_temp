import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "/home/nathan/useless/datasets"

INPUT_FILE = os.path.join(
    DATASET_ROOT,
    "processed_v2",
    "features_v2.csv"
)

OUTPUT_DIR = os.path.join(
    DATASET_ROOT,
    "processed_v2",
    "cross_validation"
)

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# LOAD DATA
# ============================================================

print()
print("========================================")
print("       V2 CROSS-VALIDATION")
print("========================================")
print()

print("Loading:")
print(INPUT_FILE)
print()

df = pd.read_csv(INPUT_FILE)

print(
    f"Total rows: {len(df)}"
)

print(
    f"Total columns: {len(df.columns)}"
)

print()


# ============================================================
# REMOVE BACKGROUND
# ============================================================

if "label" not in df.columns:
    raise ValueError(
        "ERROR: 'label' column not found."
    )

df = df[
    df["label"] != "BACKGROUND"
].copy()

print(
    f"Rows after removing background: "
    f"{len(df)}"
)

print()

print("Class distribution:")
print(
    df["label"].value_counts()
)

print()


# ============================================================
# DEFINE METADATA COLUMNS
# ============================================================

metadata_columns = [
    "file",
    "label",
    "event_number",
    "event_sample",
    "event_time_seconds",
    "sampling_rate_hz"
]


# Only use columns that actually exist
metadata_columns = [
    column
    for column in metadata_columns
    if column in df.columns
]


feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]


print(
    f"Usable ML features: "
    f"{len(feature_columns)}"
)

print()


# ============================================================
# REMOVE NON-NUMERIC FEATURES
# ============================================================

numeric_features = []

for column in feature_columns:

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):
        numeric_features.append(column)

feature_columns = numeric_features

print(
    f"Numeric ML features: "
    f"{len(feature_columns)}"
)

print()


# ============================================================
# PREPARE X / y / GROUPS
# ============================================================

X = df[
    feature_columns
].copy()

y = df[
    "label"
].copy()

# IMPORTANT:
# Events from the same recording must stay together.
groups = df[
    "file"
].copy()


print(
    f"Unique recordings: "
    f"{groups.nunique()}"
)

print()

print("Recordings:")
for filename, label in (
    df[
        ["file", "label"]
    ]
    .drop_duplicates()
    .sort_values("file")
    .values
):

    print(
        f"  {filename:25s} -> {label}"
    )

print()


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "Random Forest": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]),

    "SVM": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            SVC(
                kernel="rbf",
                class_weight="balanced"
            )
        )
    ])
}


# ============================================================
# OUTPUT STORAGE
# ============================================================

all_results = []
all_predictions = []


# ============================================================
# TRAIN / TEST EACH MODEL
# ============================================================

for model_name, model in models.items():

    print()
    print("========================================")
    print(model_name)
    print("========================================")

    fold_accuracies = []

    fold_number = 0

    for train_idx, test_idx in cv.split(
        X,
        y,
        groups
    ):

        fold_number += 1

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]

        train_groups = groups.iloc[
            train_idx
        ]

        test_groups = groups.iloc[
            test_idx
        ]

        # Safety check:
        # no recording should appear in both sets.
        overlap = set(
            train_groups
        ).intersection(
            set(test_groups)
        )

        if overlap:
            raise RuntimeError(
                "DATA LEAKAGE DETECTED: "
                f"{overlap}"
            )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        fold_accuracies.append(
            accuracy
        )

        print(
            f"Fold {fold_number}: "
            f"{accuracy * 100:.2f}%"
        )

        # Store individual predictions
        for index, actual, predicted in zip(
            test_idx,
            y_test,
            predictions
        ):

            all_predictions.append({
                "model": model_name,
                "fold": fold_number,
                "file": df.iloc[index]["file"],
                "actual": actual,
                "predicted": predicted
            })

    mean_accuracy = np.mean(
        fold_accuracies
    )

    std_accuracy = np.std(
        fold_accuracies
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

    # Store summary
    all_results.append({
        "model": model_name,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_accuracy_percent":
            mean_accuracy * 100,
        "std_accuracy_percent":
            std_accuracy * 100
    })


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.DataFrame(
    all_predictions
)

results_file = os.path.join(
    OUTPUT_DIR,
    "model_results_v2.csv"
)

predictions_file = os.path.join(
    OUTPUT_DIR,
    "predictions_v2.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

labels = sorted(
    y.unique()
)

print()
print("========================================")
print("       CONFUSION MATRICES")
print("========================================")

confusion_rows = []

for model_name in models.keys():

    model_predictions = predictions_df[
        predictions_df["model"] == model_name
    ]

    actual = model_predictions[
        "actual"
    ]

    predicted = model_predictions[
        "predicted"
    ]

    cm = confusion_matrix(
        actual,
        predicted,
        labels=labels
    )

    print()
    print(model_name)
    print()

    print(
        "Labels:",
        labels
    )

    print(cm)

    # Save matrix in long format
    for i, actual_label in enumerate(labels):

        for j, predicted_label in enumerate(labels):

            confusion_rows.append({
                "model": model_name,
                "actual": actual_label,
                "predicted": predicted_label,
                "count": cm[i, j]
            })


confusion_df = pd.DataFrame(
    confusion_rows
)

confusion_file = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix_v2.csv"
)

confusion_df.to_csv(
    confusion_file,
    index=False
)


# ============================================================
# CLASSIFICATION REPORTS
# ============================================================

print()
print("========================================")
print("       CLASSIFICATION REPORTS")
print("========================================")

for model_name in models.keys():

    model_predictions = predictions_df[
        predictions_df["model"] == model_name
    ]

    actual = model_predictions[
        "actual"
    ]

    predicted = model_predictions[
        "predicted"
    ]

    print()
    print("----------------------------------------")
    print(model_name)
    print("----------------------------------------")

    print(
        classification_report(
            actual,
            predicted,
            labels=labels,
            zero_division=0
        )
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("========================================")
print("       V2 VALIDATION COMPLETE")
print("========================================")
print()

print(
    results_df[
        [
            "model",
            "mean_accuracy_percent",
            "std_accuracy_percent"
        ]
    ].to_string(
        index=False
    )
)

print()

print("Saved:")
print(results_file)
print(predictions_file)
print(confusion_file)

print()
