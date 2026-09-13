import os
import glob
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, find_peaks


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIRECTORY = "datasets"

IMPACT_DIRECTORY = os.path.join(
    DATASET_DIRECTORY,
    "impact"
)

OUTPUT_DIRECTORY = os.path.join(
    DATASET_DIRECTORY,
    "raw_processed"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "raw_events.npz"
)

METADATA_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "metadata.csv"
)

# Expected sampling rate
EXPECTED_FS = 200.0

# High-pass filter
HIGH_PASS_CUTOFF = 2.0
FILTER_ORDER = 4

# Event detection
SMOOTHING_MS = 50
THRESHOLD_MAD_MULTIPLIER = 4.0
MIN_EVENT_DISTANCE_SECONDS = 1.0

# Event window
EVENT_WINDOW_SECONDS = 2.0
PRE_EVENT_SECONDS = 0.5


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(filename):

    data = pd.read_csv(filename)

    required_columns = [
        "timestamp_us",
        "x_g",
        "y_g",
        "z_g"
    ]

    for column in required_columns:

        if column not in data.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    data = data.dropna(
        subset=required_columns
    )

    timestamp = data[
        "timestamp_us"
    ].to_numpy(
        dtype=np.int64
    )

    x = data[
        "x_g"
    ].to_numpy(
        dtype=float
    )

    y = data[
        "y_g"
    ].to_numpy(
        dtype=float
    )

    z = data[
        "z_g"
    ].to_numpy(
        dtype=float
    )

    return timestamp, x, y, z


# ============================================================
# SAMPLING RATE
# ============================================================

def estimate_sampling_rate(timestamp):

    if len(timestamp) < 2:

        return EXPECTED_FS

    differences = np.diff(
        timestamp.astype(np.float64)
    )

    # Ignore corrupted timestamp jumps.
    valid = differences[
        (differences > 1000) &
        (differences < 20000)
    ]

    if len(valid) == 0:

        return EXPECTED_FS

    median_dt = np.median(
        valid
    )

    if median_dt <= 0:

        return EXPECTED_FS

    return (
        1_000_000.0 /
        median_dt
    )


# ============================================================
# HIGH-PASS FILTER
# ============================================================

def high_pass_filter(
    signal,
    fs
):

    nyquist = 0.5 * fs

    cutoff = (
        HIGH_PASS_CUTOFF /
        nyquist
    )

    b, a = butter(
        FILTER_ORDER,
        cutoff,
        btype="highpass"
    )

    return filtfilt(
        b,
        a,
        signal
    )


# ============================================================
# MAGNITUDE
#
# Used ONLY for finding impact locations.
#
# The actual CNN input remains X/Y/Z.
# ============================================================

def calculate_magnitude(
    x,
    y,
    z
):

    return np.sqrt(
        x ** 2 +
        y ** 2 +
        z ** 2
    )


# ============================================================
# SMOOTH MAGNITUDE
# ============================================================

def smooth_signal(
    signal,
    fs
):

    window_size = int(
        SMOOTHING_MS *
        fs /
        1000
    )

    window_size = max(
        1,
        window_size
    )

    if window_size <= 1:

        return np.abs(signal)

    kernel = (
        np.ones(window_size) /
        window_size
    )

    return np.convolve(
        np.abs(signal),
        kernel,
        mode="same"
    )


# ============================================================
# DETECT IMPACTS
# ============================================================

def detect_events(
    magnitude,
    fs
):

    smoothed = smooth_signal(
        magnitude,
        fs
    )

    median = np.median(
        smoothed
    )

    mad = np.median(
        np.abs(
            smoothed -
            median
        )
    )

    robust_sigma = (
        1.4826 * mad
    )

    threshold = (
        median +
        THRESHOLD_MAD_MULTIPLIER *
        robust_sigma
    )

    minimum_threshold = (
        median +
        0.02
    )

    threshold = max(
        threshold,
        minimum_threshold
    )

    minimum_distance = int(
        MIN_EVENT_DISTANCE_SECONDS *
        fs
    )

    peaks, properties = find_peaks(
        smoothed,
        height=threshold,
        distance=minimum_distance
    )

    return peaks


# ============================================================
# EXTRACT RAW XYZ EVENT
# ============================================================

def extract_event(
    x,
    y,
    z,
    peak,
    fs
):

    total_samples = int(
        EVENT_WINDOW_SECONDS *
        fs
    )

    pre_samples = int(
        PRE_EVENT_SECONDS *
        fs
    )

    start = (
        peak -
        pre_samples
    )

    end = (
        start +
        total_samples
    )

    # Normally the detector finds events
    # far enough from the edges.
    #
    # If not, skip them rather than padding
    # with artificial data.

    if start < 0:

        return None

    if end > len(x):

        return None

    event_x = x[start:end]
    event_y = y[start:end]
    event_z = z[start:end]

    if len(event_x) != total_samples:

        return None

    # Shape:
    #
    # (3, 400)
    #
    # Channel 0 = X
    # Channel 1 = Y
    # Channel 2 = Z

    event = np.stack(
        [
            event_x,
            event_y,
            event_z
        ],
        axis=0
    )

    return event


# ============================================================
# DATASET FILES
# ============================================================

def get_dataset_files():

    class_directories = {

        "1rup": "RUPEE_1",

        "2rup": "RUPEE_2",

        "5rup": "RUPEE_5",

        "10rup": "RUPEE_10",

        "20rup": "RUPEE_20",

    }

    files = []

    for directory, label in class_directories.items():

        folder = os.path.join(
            IMPACT_DIRECTORY,
            directory
        )

        if not os.path.exists(folder):

            continue

        csv_files = glob.glob(
            os.path.join(
                folder,
                "*.csv"
            )
        )

        for filename in csv_files:

            # Only use the new ±4g recordings.

            basename = os.path.basename(
                filename
            )

            if "_4g" not in basename:

                continue

            files.append(
                (
                    filename,
                    label
                )
            )

    return sorted(
        files
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=============================================="
    )
    print(
        "       RAW XYZ EVENT PREPARATION"
    )
    print(
        "=============================================="
    )
    print()

    dataset_files = get_dataset_files()

    if len(dataset_files) == 0:

        print(
            "ERROR: No _4g CSV files found."
        )

        return

    print(
        f"Recordings found: {len(dataset_files)}"
    )

    print()

    all_events = []
    metadata = []

    # --------------------------------------------------------
    # Process recordings
    # --------------------------------------------------------

    for recording_id, (
        filename,
        label
    ) in enumerate(
        dataset_files
    ):

        print(
            f"Processing: {filename}"
        )

        timestamp, x, y, z = load_csv(
            filename
        )

        fs = estimate_sampling_rate(
            timestamp
        )

        print(
            f"  Sampling rate: {fs:.2f} Hz"
        )

        # ----------------------------------------------------
        # Filter only for event detection.
        #
        # The resulting filtered XYZ is what we store.
        # ----------------------------------------------------

        filtered_x = high_pass_filter(
            x,
            fs
        )

        filtered_y = high_pass_filter(
            y,
            fs
        )

        filtered_z = high_pass_filter(
            z,
            fs
        )

        magnitude = calculate_magnitude(
            filtered_x,
            filtered_y,
            filtered_z
        )

        peaks = detect_events(
            magnitude,
            fs
        )

        print(
            f"  Detected events: {len(peaks)}"
        )

        successful_events = 0

        for event_number, peak in enumerate(
            peaks,
            start=1
        ):

            event = extract_event(
                filtered_x,
                filtered_y,
                filtered_z,
                peak,
                fs
            )

            if event is None:

                continue

            all_events.append(
                event
            )

            metadata.append(
                {
                    "event_id":
                        len(all_events) - 1,

                    "recording_id":
                        recording_id,

                    "filename":
                        os.path.basename(
                            filename
                        ),

                    "label":
                        label,

                    "event_number":
                        event_number,

                    "peak_sample":
                        int(peak),

                    "peak_time_seconds":
                        peak / fs,

                    "sampling_rate_hz":
                        fs,
                }
            )

            successful_events += 1

        print(
            f"  Successfully extracted: "
            f"{successful_events}"
        )

        print()

    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    if len(all_events) == 0:

        print(
            "ERROR: No events were extracted."
        )

        return

    # --------------------------------------------------------
    # Convert to NumPy array
    #
    # Shape:
    #
    # (events, 3, 400)
    # --------------------------------------------------------

    X = np.stack(
        all_events,
        axis=0
    ).astype(
        np.float32
    )

    metadata_df = pd.DataFrame(
        metadata
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_FILE,
        X=X
    )

    metadata_df.to_csv(
        METADATA_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(
        "=============================================="
    )
    print(
        "       RAW DATA PREPARATION COMPLETE"
    )
    print(
        "=============================================="
    )
    print()

    print(
        f"Total events: {len(X)}"
    )

    print(
        f"Event shape: {X.shape[1:]}"
    )

    print(
        "Channels: X, Y, Z"
    )

    print(
        "Samples/event: "
        f"{X.shape[2]}"
    )

    print(
        "Duration/event: "
        f"{X.shape[2] / EXPECTED_FS:.2f} seconds"
    )

    print()

    print(
        "Class distribution:"
    )

    print(
        metadata_df[
            "label"
        ].value_counts()
    )

    print()

    print(
        "Recordings:"
    )

    print(
        metadata_df[
            [
                "recording_id",
                "filename",
                "label"
            ]
        ].drop_duplicates(
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "Saved:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  {METADATA_FILE}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
