import os
import random

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_class_weight


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
    "cross_validation"
)

RANDOM_STATE = 42

N_SPLITS = 5

EPOCHS = 100

BATCH_SIZE = 8

PATIENCE = 12


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(
    RANDOM_STATE
)

random.seed(
    RANDOM_STATE
)

np.random.seed(
    RANDOM_STATE
)

tf.random.set_seed(
    RANDOM_STATE
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("========================================")
print("       RAW WAVEFORM V3 CNN")
print("========================================")
print()

print("Loading waveform data:")
print(NPZ_FILE)

data = np.load(
    NPZ_FILE
)

X = data["X"].astype(
    np.float32
)

metadata = pd.read_csv(
    METADATA_FILE
)

print()

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
        "Number of waveforms does not "
        "match metadata rows."
    )


required_columns = [
    "label",
    "file"
]

for column in required_columns:

    if column not in metadata.columns:

        raise ValueError(
            f"Missing metadata column: "
            f"{column}"
        )


# ============================================================
# CLEAN LABELS
# ============================================================

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
# FIXED LABEL MAPPING
# ============================================================

label_to_number = {
    "RUPEE_1": 0,
    "RUPEE_2": 1,
    "RUPEE_5": 2,
    "RUPEE_10": 3,
    "RUPEE_20": 4
}

number_to_label = {
    0: "RUPEE_1",
    1: "RUPEE_2",
    2: "RUPEE_5",
    3: "RUPEE_10",
    4: "RUPEE_20"
}

class_names = [
    "RUPEE_1",
    "RUPEE_2",
    "RUPEE_5",
    "RUPEE_10",
    "RUPEE_20"
]

num_classes = len(
    class_names
)


# ============================================================
# VERIFY LABELS
# ============================================================

unknown_labels = set(
    metadata["label"]
) - set(
    label_to_number.keys()
)

if unknown_labels:

    raise ValueError(
        "Unknown labels found: "
        f"{unknown_labels}"
    )


# Convert string labels to integers
y = np.array([
    label_to_number[label]
    for label in metadata["label"]
])


print("Classes:")

for number, name in number_to_label.items():

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
# PREPARE CNN INPUT
# ============================================================

# Original:
#
#     (events, 3, 400)
#
# Conv1D expects:
#
#     (events, 400, 3)

X = np.transpose(
    X,
    (0, 2, 1)
)

print(
    f"CNN input shape: {X.shape}"
)

print()


# ============================================================
# MODEL
# ============================================================

def build_model(
    input_shape,
    num_classes
):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=input_shape
        ),

        # ----------------------------------------------------
        # Block 1
        # ----------------------------------------------------

        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=7,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),

        tf.keras.layers.Dropout(
            0.20
        ),

        # ----------------------------------------------------
        # Block 2
        # ----------------------------------------------------

        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=5,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),

        tf.keras.layers.Dropout(
            0.25
        ),

        # ----------------------------------------------------
        # Block 3
        # ----------------------------------------------------

        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        # ----------------------------------------------------
        # Global pooling
        # ----------------------------------------------------

        tf.keras.layers.GlobalAveragePooling1D(),

        # ----------------------------------------------------
        # Dense layer
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.40
        ),

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy"
        ]
    )

    return model


# ============================================================
# GROUPS
# ============================================================

groups = metadata[
    "file"
].values


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
    # Verify no recording leakage
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
    ].copy()

    X_test = X[
        test_idx
    ].copy()

    y_train = y[
        train_idx
    ]

    y_test = y[
        test_idx
    ]

    # --------------------------------------------------------
    # Normalize using TRAINING data only
    # --------------------------------------------------------

    train_mean = np.mean(
        X_train,
        axis=(0, 1)
    )

    train_std = np.std(
        X_train,
        axis=(0, 1)
    )

    train_std[
        train_std < 1e-6
    ] = 1.0

    X_train = (
        X_train -
        train_mean
    ) / train_std

    X_test = (
        X_test -
        train_mean
    ) / train_std

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    train_classes = np.unique(
        y_train
    )

    class_weights_array = (
        compute_class_weight(
            class_weight="balanced",
            classes=train_classes,
            y=y_train
        )
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight in zip(
            train_classes,
            class_weights_array
        )
    }

    print(
        "Class weights:",
        class_weights
    )

    # --------------------------------------------------------
    # Build fresh model
    # --------------------------------------------------------

    model = build_model(
        input_shape=X_train.shape[1:],
        num_classes=num_classes
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True
        )
    )

    reduce_lr = (
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history = model.fit(
        X_train,
        y_train,

        validation_split=0.20,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=[
            early_stopping,
            reduce_lr
        ],

        verbose=0,

        shuffle=True
    )

    best_epoch = (
        np.argmin(
            history.history[
                "val_loss"
            ]
        ) + 1
    )

    print()

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{min(history.history['val_loss']):.4f}"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
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

        "best_epoch":
            best_epoch,

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
    # Store predictions
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
                number_to_label[
                    int(actual)
                ],

            "predicted":
                number_to_label[
                    int(predicted)
                ]
        })

    all_actual.extend(
        y_test.tolist()
    )

    all_predicted.extend(
        predictions.tolist()
    )

    # --------------------------------------------------------
    # Clear model
    # --------------------------------------------------------

    tf.keras.backend.clear_session()


# ============================================================
# FINAL STATISTICS
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
        num_classes
    )
)

confusion_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_actual,
    all_predicted,
    labels=np.arange(
        num_classes
    ),
    target_names=class_names,
    zero_division=0
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "cnn_results_v3.csv"
)

predictions_file = os.path.join(
    OUTPUT_DIR,
    "cnn_predictions_v3.csv"
)

confusion_file = os.path.join(
    OUTPUT_DIR,
    "cnn_confusion_matrix_v3.csv"
)

report_file = os.path.join(
    OUTPUT_DIR,
    "cnn_classification_report_v3.txt"
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
        "RAW WAVEFORM V3 CNN\n"
    )

    f.write(
        "====================\n\n"
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
print("       V3 CNN COMPLETE")
print("========================================")
print()

print(
    results_df[
        [
            "fold",
            "accuracy_percent",
            "best_epoch"
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
