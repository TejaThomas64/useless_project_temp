import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "/home/nathan/useless/datasets"

INPUT_DIR = os.path.join(
    DATASET_ROOT,
    "waveform_v3"
)

NPZ_FILE = os.path.join(
    INPUT_DIR,
    "waveforms_v3.npz"
)

METADATA_FILE = os.path.join(
    INPUT_DIR,
    "metadata_v3.csv"
)

OUTPUT_DIR = os.path.join(
    INPUT_DIR,
    "raw_svm_results"
)

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_TO_NUMBER = {
    "RUPEE_1": 0,
    "RUPEE_2": 1,
    "RUPEE_5": 2,
    "RUPEE_10": 3,
    "RUPEE_20": 4
}

NUMBER_TO_LABEL = {
    0: "RUPEE_1",
    1: "RUPEE_2",
    2: "RUPEE_5",
    3: "RUPEE_10",
    4: "RUPEE_20"
}

CLASS_NAMES = [
    "RUPEE_1",
    "RUPEE_2",
    "RUPEE_5",
    "RUPEE_10",
    "RUPEE_20"
]


# ============================================================
# LOAD DATA
# ============================================================

print()
print("========================================")
print("      RAW WAVEFORM SVM V3")
print("========================================")
print()

print("Loading:")
print(NPZ_FILE)
print()

data = np.load(
    NPZ_FILE
)

X = data["X"].astype(
    np.float32
)

metadata = pd.read_csv(
    METADATA_FILE
)

print(
    f"Waveform shape: {X.shape}"
)

print(
    f"Metadata rows: {len(metadata)}"
)

print()


# ============================================================
# SAFETY CHECKS
# ============================================================

if len(X) != len(metadata):

    raise ValueError(
        "Waveform count and metadata "
        "count do not match."
    )


if "label" not in metadata.columns:

    raise ValueError(
        "Missing label column."
    )


if "file" not in metadata.columns:

    raise ValueError(
        "Missing file column."
    )


metadata["label"] = (
    metadata["label"]
    .astype(str)
    .str.strip()
)

metadata["file"] = (
    metadata["file"]
    .astype(str)
    .str.strip()
)


# ============================================================
# VERIFY LABELS
# ============================================================

unknown_labels = (
    set(metadata["label"]) -
    set(LABEL_TO_NUMBER)
)

if unknown_labels:

    raise ValueError(
        f"Unknown labels: {unknown_labels}"
    )


y = np.array([
    LABEL_TO_NUMBER[label]
    for label in metadata["label"]
])


groups = metadata[
    "file"
].values


print("Classes:")

for number, name in NUMBER_TO_LABEL.items():

    print(
        f"  {number}: {name}"
    )

print()

print("Class distribution:")

print(
    metadata["label"].value_counts()
)

print()

print(
    "Unique recordings:",
    metadata["file"].nunique()
)

print()


# ============================================================
# FLATTEN WAVEFORM
# ============================================================

# Original:
#
#     events × 3 × 400
#
# becomes:
#
#     events × 1200
#
# Each event is therefore represented by:
#
#     X[0..399]
#     Y[0..399]
#     Z[0..399]

X = X.reshape(
    X.shape[0],
    -1
)

print(
    f"Flattened waveform shape: {X.shape}"
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


fold_results = []

all_predictions = []

all_actual = []

all_predicted = []


# ============================================================
# FOLD LOOP
# ============================================================

for fold_number, (
    train_idx,
    test_idx
) in enumerate(
    cv.split(
        X,
        y,
        groups
    ),
    start=1
):

    print()
    print("========================================")
    print(
        f"FOLD {fold_number}/{N_SPLITS}"
    )
    print("========================================")
    print()

    # --------------------------------------------------------
    # Recording leakage check
    # --------------------------------------------------------

    train_files = set(
        groups[train_idx]
    )

    test_files = set(
        groups[test_idx]
    )

    overlap = (
        train_files &
        test_files
    )

    if overlap:

        raise RuntimeError(
            "DATA LEAKAGE DETECTED: "
            f"{overlap}"
        )

    print(
        f"Training events: {len(train_idx)}"
    )

    print(
        f"Testing events: {len(test_idx)}"
    )

    print()

    print(
        "Test recordings:"
    )

    for filename in sorted(
        test_files
    ):

        labels = metadata.loc[
            metadata["file"] == filename,
            "label"
        ].unique()

        print(
            f"  {filename} -> {labels[0]}"
        )

    print()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    X_train = X[
        train_idx
    ]

    X_test = X[
        test_idx
    ]

    y_train = y[
        train_idx
    ]

    y_test = y[
        test_idx
    ]

    # --------------------------------------------------------
    # Model
    #
    # StandardScaler:
    # normalize using training data only.
    #
    # PCA:
    # reduce 1200 raw waveform values to a
    # smaller representation.
    #
    # SVM:
    # classify the resulting representation.
    # --------------------------------------------------------

    model = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "pca",
            PCA(
                n_components=0.95,
                svd_solver="full",
                random_state=RANDOM_STATE
            )
        ),

        (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                class_weight="balanced"
            )
        )
    ])

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Report PCA dimensionality
    # --------------------------------------------------------

    pca = model.named_steps[
        "pca"
    ]

    print(
        f"PCA components: "
        f"{pca.n_components_}"
    )

    print(
        f"Explained variance: "
        f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    fold_results.append({

        "fold":
            fold_number,

        "accuracy":
            accuracy,

        "accuracy_percent":
            accuracy * 100,

        "pca_components":
            pca.n_components_,

        "explained_variance_percent":
            np.sum(
                pca.explained_variance_ratio_
            ) * 100,

        "train_events":
            len(train_idx),

        "test_events":
            len(test_idx)
    })

    print()

    print(
        f"Fold {fold_number} accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    for idx, actual, predicted in zip(
        test_idx,
        y_test,
        predictions
    ):

        all_predictions.append({

            "fold":
                fold_number,

            "file":
                metadata.iloc[idx]["file"],

            "actual":
                NUMBER_TO_LABEL[
                    int(actual)
                ],

            "predicted":
                NUMBER_TO_LABEL[
                    int(predicted)
                ]
        })

    all_actual.extend(
        y_test.tolist()
    )

    all_predicted.extend(
        predictions.tolist()
    )


# ============================================================
# FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    fold_results
)

mean_accuracy = results_df[
    "accuracy"
].mean()

std_accuracy = results_df[
    "accuracy"
].std(
    ddof=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_actual,
    all_predicted,
    labels=np.arange(
        len(CLASS_NAMES)
    )
)

confusion_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_actual,
    all_predicted,
    labels=np.arange(
        len(CLASS_NAMES)
    ),
    target_names=CLASS_NAMES,
    zero_division=0
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

results_file = os.path.join(
    OUTPUT_DIR,
    "raw_svm_results_v3.csv"
)

predictions_file = os.path.join(
    OUTPUT_DIR,
    "raw_svm_predictions_v3.csv"
)

confusion_file = os.path.join(
    OUTPUT_DIR,
    "raw_svm_confusion_matrix_v3.csv"
)

report_file = os.path.join(
    OUTPUT_DIR,
    "raw_svm_classification_report_v3.txt"
)


results_df.to_csv(
    results_file,
    index=False
)

pd.DataFrame(
    all_predictions
).to_csv(
    predictions_file,
    index=False
)

confusion_df.to_csv(
    confusion_file
)

with open(
    report_file,
    "w"
) as f:

    f.write(
        "RAW WAVEFORM + PCA + SVM V3\n"
    )

    f.write(
        "============================\n\n"
    )

    f.write(
        f"Mean accuracy: "
        f"{mean_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Std deviation: "
        f"{std_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        report
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("========================================")
print("       RAW SVM V3 COMPLETE")
print("========================================")
print()

print(
    results_df[
        [
            "fold",
            "accuracy_percent",
            "pca_components"
        ]
    ].to_string(
        index=False
    )
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

print(
    "Confusion matrix:"
)

print()

print(
    confusion_df
)

print()

print(
    "Classification report:"
)

print(
    report
)

print()

print("Saved:")
print(results_file)
print(predictions_file)
print(confusion_file)
print(report_file)

print()
