import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = (
    "datasets/processed/features.csv"
)

OUTPUT_DIRECTORY = (
    "datasets/processed/cross_validation"
)

RESULTS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "cross_validation_results.txt"
)

CONFUSION_MATRIX_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "confusion_matrix.csv"
)

N_SPLITS = 5

RANDOM_STATE = 42


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print(
    "=============================================="
)

print(
    "       RECORDING-LEVEL CROSS VALIDATION"
)

print(
    "=============================================="
)

print()

print(
    f"Loading: {FEATURE_FILE}"
)

df = pd.read_csv(
    FEATURE_FILE
)

print(
    f"Total rows: {len(df)}"
)

print(
    f"Total columns: {len(df.columns)}"
)

print()


# ============================================================
# SHOW ORIGINAL CLASSES
# ============================================================

print(
    "Original class distribution:"
)

print(
    df["label"].value_counts()
)

print()


# ============================================================
# REMOVE BACKGROUND
#
# This experiment is denomination classification.
#
# Background vs impact should be a separate experiment.
# ============================================================

df = df[
    df["label"] != "BACKGROUND"
].copy()

print(
    "After removing BACKGROUND:"
)

print(
    df["label"].value_counts()
)

print()


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

# Columns containing metadata rather than vibration features.

metadata_columns = [
    "file",
    "label",
    "event_number",
    "event_time_seconds",
    "sampling_rate_hz"
]


feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]


print(
    f"Feature columns available: "
    f"{len(feature_columns)}"
)

print()

print(
    "Excluded metadata:"
)

for column in metadata_columns:

    if column in df.columns:

        print(
            f"  {column}"
        )

print()


# ============================================================
# BUILD X AND y
# ============================================================

X = df[
    feature_columns
].copy()

y = df[
    "label"
].copy()

groups = df[
    "file"
].copy()


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

for column in X.columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# ============================================================
# REMOVE COMPLETELY EMPTY FEATURES
# ============================================================

valid_columns = [
    column
    for column in X.columns
    if not X[column].isna().all()
]

X = X[
    valid_columns
]

feature_columns = valid_columns

print(
    f"Usable feature columns: "
    f"{len(feature_columns)}"
)

print()


# ============================================================
# GROUP INFORMATION
# ============================================================

print(
    "Recording groups:"
)

group_table = (
    df[
        [
            "file",
            "label"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        ["label", "file"]
    )
)

print(
    group_table.to_string(
        index=False
    )
)

print()

print(
    f"Unique recordings: "
    f"{groups.nunique()}"
)

print()


# ============================================================
# CROSS VALIDATION
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

    "Logistic Regression": Pipeline(
        [
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
                    C=1.0,
                    random_state=RANDOM_STATE
                )
            )
        ]
    ),

    "Random Forest": Pipeline(
        [
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
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    ),

    "SVM": Pipeline(
        [
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
                    C=1.0,
                    gamma="scale"
                )
            )
        ]
    )

}


# ============================================================
# STORAGE
# ============================================================

all_results = []

all_confusion_matrices = {}


# ============================================================
# RUN EACH MODEL
# ============================================================

for model_name, model in models.items():

    print()
    print(
        "=============================================="
    )

    print(
        f"MODEL: {model_name}"
    )

    print(
        "=============================================="
    )

    print()

    fold_accuracies = []

    combined_true = []

    combined_pred = []

    # --------------------------------------------------------
    # Cross-validation folds
    # --------------------------------------------------------

    for fold_number, (
        train_indices,
        test_indices
    ) in enumerate(

        cv.split(
            X,
            y,
            groups
        ),

        start=1

    ):

        X_train = X.iloc[
            train_indices
        ]

        X_test = X.iloc[
            test_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        y_test = y.iloc[
            test_indices
        ]

        train_groups = groups.iloc[
            train_indices
        ]

        test_groups = groups.iloc[
            test_indices
        ]

        # ----------------------------------------------------
        # Safety check:
        #
        # No recording may occur in both train and test.
        # ----------------------------------------------------

        overlap = set(
            train_groups
        ).intersection(
            set(test_groups)
        )

        if len(overlap) > 0:

            raise RuntimeError(
                "DATA LEAKAGE DETECTED: "
                f"{overlap}"
            )

        # ----------------------------------------------------
        # Print fold information
        # ----------------------------------------------------

        print(
            f"Fold {fold_number}"
        )

        print(
            f"  Training events: "
            f"{len(train_indices)}"
        )

        print(
            f"  Test events: "
            f"{len(test_indices)}"
        )

        print(
            f"  Training recordings: "
            f"{len(set(train_groups))}"
        )

        print(
            f"  Test recordings: "
            f"{len(set(test_groups))}"
        )

        print(
            f"  Test recordings:"
        )

        for recording in sorted(
            set(test_groups)
        ):

            recording_label = df.loc[
                df["file"] == recording,
                "label"
            ].iloc[0]

            print(
                f"    {recording_label}: "
                f"{recording}"
            )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

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

        combined_true.extend(
            y_test.tolist()
        )

        combined_pred.extend(
            predictions.tolist()
        )

        print(
            f"  Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    mean_accuracy = np.mean(
        fold_accuracies
    )

    std_accuracy = np.std(
        fold_accuracies
    )

    print(
        "----------------------------------------------"
    )

    print(
        f"Fold accuracies:"
    )

    for i, accuracy in enumerate(
        fold_accuracies,
        start=1
    ):

        print(
            f"  Fold {i}: "
            f"{accuracy * 100:.2f}%"
        )

    print()

    print(
        f"Mean accuracy: "
        f"{mean_accuracy * 100:.2f}%"
    )

    print(
        f"Standard deviation: "
        f"{std_accuracy * 100:.2f}%"
    )

    print(
        "----------------------------------------------"
    )

    # --------------------------------------------------------
    # Combined confusion matrix
    # --------------------------------------------------------

    class_order = [
        "RUPEE_1",
        "RUPEE_2",
        "RUPEE_5",
        "RUPEE_10",
        "RUPEE_20"
    ]

    cm = confusion_matrix(
        combined_true,
        combined_pred,
        labels=class_order
    )

    all_confusion_matrices[
        model_name
    ] = cm

    print()

    print(
        "Combined confusion matrix:"
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual_{c}"
            for c in class_order
        ],
        columns=[
            f"Predicted_{c}"
            for c in class_order
        ]
    )

    print(
        cm_df.to_string()
    )

    print()

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "Classification report:"
    )

    print(
        classification_report(
            combined_true,
            combined_pred,
            labels=class_order,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    all_results.append(
        {
            "model":
                model_name,

            "mean_accuracy":
                mean_accuracy,

            "std_accuracy":
                std_accuracy,

            "fold_1":
                fold_accuracies[0],

            "fold_2":
                fold_accuracies[1],

            "fold_3":
                fold_accuracies[2],

            "fold_4":
                fold_accuracies[3],

            "fold_5":
                fold_accuracies[4]
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIRECTORY,
        "model_results.csv"
    ),
    index=False
)


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

with open(
    CONFUSION_MATRIX_FILE,
    "w"
) as file:

    class_order = [
        "RUPEE_1",
        "RUPEE_2",
        "RUPEE_5",
        "RUPEE_10",
        "RUPEE_20"
    ]

    for model_name, cm in (
        all_confusion_matrices.items()
    ):

        file.write(
            f"\n{model_name}\n"
        )

        file.write(
            ",".join(
                class_order
            )
            +
            "\n"
        )

        for row in cm:

            file.write(
                ",".join(
                    str(value)
                    for value in row
                )
                +
                "\n"
            )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print(
    "=============================================="
)

print(
    "       CROSS VALIDATION COMPLETE"
)

print(
    "=============================================="
)

print()

print(
    results_df[
        [
            "model",
            "mean_accuracy",
            "std_accuracy"
        ]
    ].to_string(
        index=False
    )
)

print()

print(
    "Results saved to:"
)

print(
    f"  {OUTPUT_DIRECTORY}/model_results.csv"
)

print(
    f"  {CONFUSION_MATRIX_FILE}"
)

print()
