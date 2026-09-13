import os
import glob

import numpy as np
import pandas as pd
from scipy import signal


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "/home/nathan/useless/datasets"

OUTPUT_DIR = os.path.join(
    DATASET_ROOT,
    "waveform_v3"
)

NPZ_FILE = os.path.join(
    OUTPUT_DIR,
    "waveforms_v3.npz"
)

METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "metadata_v3.csv"
)

DEFAULT_FS = 200.0

HIGH_PASS_CUTOFF = 2.0

EVENT_DURATION = 2.0

PRE_EVENT_TIME = 0.5

SMOOTHING_WINDOW = 11

THRESHOLD_MULTIPLIER = 4.0

MIN_EVENT_DISTANCE = 0.8


# ============================================================
# LABEL
# ============================================================

def label_from_filename(filename):

    name = os.path.basename(filename).lower()

    number = None

    for i in range(1, 15):

        if name == f"impact{i}_4g.csv":
            number = i
            break

    if number is None:
        return None

    if number <= 3:
        return "RUPEE_1"

    if number <= 6:
        return "RUPEE_2"

    if number <= 9:
        return "RUPEE_5"

    if number <= 12:
        return "RUPEE_10"

    return "RUPEE_20"


# ============================================================
# FIND FILES
# ============================================================

def find_files():

    patterns = [
        os.path.join(
            DATASET_ROOT,
            "impact",
            "*",
            "*_4g.csv"
        ),

        os.path.join(
            DATASET_ROOT,
            "impact",
            "*_4g.csv"
        )
    ]

    files = []

    for pattern in patterns:
        files.extend(
            glob.glob(pattern)
        )

    return sorted(
        list(set(files))
    )


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(filename):

    df = pd.read_csv(filename)

    required = [
        "timestamp_us",
        "x_g",
        "y_g",
        "z_g"
    ]

    for column in required:

        if column not in df.columns:
            raise ValueError(
                f"Missing {column} "
                f"in {filename}"
            )

    df = df[required].copy()

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna()

    return df


# ============================================================
# SAMPLING RATE
# ============================================================

def estimate_sampling_rate(timestamp):

    dt = np.diff(timestamp)

    # Ignore timestamp corruption
    valid = dt[
        (dt > 1000) &
        (dt < 20000)
    ]

    if len(valid) == 0:
        return DEFAULT_FS

    median_dt = np.median(valid)

    if median_dt <= 0:
        return DEFAULT_FS

    fs = 1_000_000.0 / median_dt

    if fs < 50 or fs > 500:
        return DEFAULT_FS

    return fs


# ============================================================
# HIGH PASS
# ============================================================

def highpass_filter(x, fs):

    nyquist = fs / 2.0

    cutoff = min(
        HIGH_PASS_CUTOFF,
        nyquist * 0.25
    )

    b, a = signal.butter(
        3,
        cutoff / nyquist,
        btype="highpass"
    )

    try:

        return signal.filtfilt(
            b,
            a,
            x
        )

    except ValueError:

        return x - np.mean(x)


# ============================================================
# EVENT DETECTION
# ============================================================

def detect_events(x, y, z, fs):

    magnitude = np.sqrt(
        x ** 2 +
        y ** 2 +
        z ** 2
    )

    # Smooth ONLY for detection
    window = min(
        SMOOTHING_WINDOW,
        len(magnitude)
    )

    if window % 2 == 0:
        window -= 1

    if window >= 3:

        smooth = signal.savgol_filter(
            magnitude,
            window,
            2
        )

    else:

        smooth = magnitude

    median = np.median(smooth)

    mad = np.median(
        np.abs(
            smooth - median
        )
    )

    if mad < 1e-8:
        mad = np.std(smooth) + 1e-8

    threshold = (
        median +
        THRESHOLD_MULTIPLIER * mad
    )

    peaks, _ = signal.find_peaks(
        smooth,
        height=threshold,
        distance=int(
            fs * MIN_EVENT_DISTANCE
        )
    )

    # Fallback
    if len(peaks) == 0:

        peaks, _ = signal.find_peaks(
            smooth,
            distance=int(
                fs * MIN_EVENT_DISTANCE
            ),
            prominence=max(
                np.std(smooth),
                0.01
            )
        )

    return peaks


# ============================================================
# PROCESS RECORDING
# ============================================================

def process_recording(filename):

    label = label_from_filename(
        filename
    )

    if label is None:
        return [], []

    print()
    print(
        "Processing:",
        os.path.basename(filename)
    )

    df = load_csv(filename)

    timestamp = df[
        "timestamp_us"
    ].values

    x = df[
        "x_g"
    ].values.astype(np.float32)

    y = df[
        "y_g"
    ].values.astype(np.float32)

    z = df[
        "z_g"
    ].values.astype(np.float32)

    fs = estimate_sampling_rate(
        timestamp
    )

    print(
        f"Samples: {len(df)}"
    )

    print(
        f"Sampling rate: {fs:.2f} Hz"
    )

    # --------------------------------------------------------
    # Filter each axis
    # --------------------------------------------------------

    x = highpass_filter(
        x,
        fs
    ).astype(np.float32)

    y = highpass_filter(
        y,
        fs
    ).astype(np.float32)

    z = highpass_filter(
        z,
        fs
    ).astype(np.float32)

    # --------------------------------------------------------
    # Detect using magnitude
    # --------------------------------------------------------

    peaks = detect_events(
        x,
        y,
        z,
        fs
    )

    print(
        f"Detected events: {len(peaks)}"
    )

    window_samples = int(
        EVENT_DURATION * fs
    )

    pre_samples = int(
        PRE_EVENT_TIME * fs
    )

    events = []

    metadata = []

    for event_number, peak in enumerate(
        peaks,
        start=1
    ):

        start = (
            peak -
            pre_samples
        )

        end = (
            start +
            window_samples
        )

        if start < 0:
            continue

        if end > len(x):
            continue

        event_x = x[
            start:end
        ]

        event_y = y[
            start:end
        ]

        event_z = z[
            start:end
        ]

        if len(event_x) != window_samples:
            continue

        # ----------------------------------------------------
        # Shape = (3, 400)
        # ----------------------------------------------------

        event = np.stack(
            [
                event_x,
                event_y,
                event_z
            ],
            axis=0
        )

        events.append(
            event
        )

        metadata.append({
            "file": os.path.basename(filename),
            "label": label,
            "event_number": event_number,
            "event_sample": peak,
            "event_time_seconds":
                peak / fs,
            "sampling_rate_hz": fs
        })

    return events, metadata


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "========================================"
    )
    print(
        "       RAW WAVEFORM PREPARATION V3"
    )
    print(
        "========================================"
    )
    print()

    print(
        "Using ONLY _4g.csv recordings."
    )

    print(
        "No per-event amplitude normalization."
    )

    print()

    files = find_files()

    print(
        f"Found {len(files)} recordings."
    )

    if len(files) == 0:

        print(
            "ERROR: No _4g.csv files found."
        )

        return

    all_events = []
    all_metadata = []

    for filename in files:

        events, metadata = (
            process_recording(filename)
        )

        all_events.extend(
            events
        )

        all_metadata.extend(
            metadata
        )

    if len(all_events) == 0:

        print()
        print(
            "ERROR: No valid events."
        )

        return

    X = np.stack(
        all_events
    ).astype(
        np.float32
    )

    metadata_df = pd.DataFrame(
        all_metadata
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    np.savez_compressed(
        NPZ_FILE,
        X=X
    )

    metadata_df.to_csv(
        METADATA_FILE,
        index=False
    )

    print()
    print(
        "========================================"
    )
    print(
        "       V3 PREPARATION COMPLETE"
    )
    print(
        "========================================"
    )
    print()

    print(
        "Waveform shape:"
    )

    print(
        X.shape
    )

    print()

    print(
        "Expected format:"
    )

    print(
        "(events, 3, 400)"
    )

    print()

    print(
        "Events by class:"
    )

    print(
        metadata_df[
            "label"
        ].value_counts()
    )

    print()

    print(
        "Unique recordings:"
    )

    print(
        metadata_df[
            "file"
        ].nunique()
    )

    print()

    print(
        "Saved:"
    )

    print(
        NPZ_FILE
    )

    print(
        METADATA_FILE
    )

    print()


if __name__ == "__main__":
    main()
