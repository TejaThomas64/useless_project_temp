import os
import glob
import numpy as np
import pandas as pd

from scipy.signal import (
    butter,
    filtfilt,
    find_peaks,
    windows
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIRECTORY = "datasets"

BACKGROUND_DIRECTORY = os.path.join(
    DATASET_DIRECTORY,
    "background"
)

IMPACT_DIRECTORY = os.path.join(
    DATASET_DIRECTORY,
    "impact"
)

OUTPUT_DIRECTORY = os.path.join(
    DATASET_DIRECTORY,
    "processed"
)

PLOT_DIRECTORY = os.path.join(
    OUTPUT_DIRECTORY,
    "event_plots"
)

FEATURE_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "features.csv"
)

# Expected sampling rate
EXPECTED_FS = 200.0

# High-pass filter cutoff.
# Removes gravity / very slow movement.
HIGH_PASS_CUTOFF = 2.0

FILTER_ORDER = 4

# Frequency range used for FFT features
MAX_FREQUENCY = 100.0

# Event detector parameters

SMOOTHING_MS = 50

THRESHOLD_MAD_MULTIPLIER = 4.0

MIN_EVENT_DISTANCE_SECONDS = 1.0

# Event window

EVENT_WINDOW_SECONDS = 2.0

PRE_EVENT_SECONDS = 0.5


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)

os.makedirs(
    PLOT_DIRECTORY,
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
# ESTIMATE SAMPLING RATE
# ============================================================

def estimate_sampling_rate(timestamp):

    if len(timestamp) < 2:

        return EXPECTED_FS

    differences = np.diff(
        timestamp.astype(np.float64)
    )

    # Remove invalid / corrupted timestamp jumps.
    #
    # Normal interval is around 5000 us.
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

    fs = 1_000_000.0 / median_dt

    return fs


# ============================================================
# FILTER
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

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered


# ============================================================
# ACCELERATION MAGNITUDE
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
# SMOOTH SIGNAL
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

    kernel = np.ones(
        window_size
    ) / window_size

    smoothed = np.convolve(
        np.abs(signal),
        kernel,
        mode="same"
    )

    return smoothed


# ============================================================
# EVENT DETECTION
# ============================================================

def detect_events(
    magnitude,
    fs
):

    smoothed = smooth_signal(
        magnitude,
        fs
    )

    # Robust statistics
    median = np.median(
        smoothed
    )

    mad = np.median(
        np.abs(
            smoothed - median
        )
    )

    # Convert MAD to approximately
    # standard-deviation scale.
    robust_sigma = (
        1.4826 * mad
    )

    threshold = (
        median +
        THRESHOLD_MAD_MULTIPLIER *
        robust_sigma
    )

    # Safety floor.
    #
    # Prevents an extremely quiet recording
    # from producing an almost-zero threshold.
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

    return (
        peaks,
        threshold,
        smoothed
    )


# ============================================================
# EXTRACT EVENT WINDOW
# ============================================================

def extract_event_window(
    x,
    y,
    z,
    magnitude,
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

    start = peak - pre_samples

    end = start + total_samples

    # Shift window if it goes before recording
    if start < 0:

        start = 0
        end = total_samples

    # Shift window if it goes after recording
    if end > len(x):

        end = len(x)
        start = (
            end -
            total_samples
        )

    # If recording is shorter than expected
    if start < 0:

        start = 0

    event_x = x[start:end]
    event_y = y[start:end]
    event_z = z[start:end]
    event_magnitude = magnitude[start:end]

    return (
        event_x,
        event_y,
        event_z,
        event_magnitude,
        start,
        end
    )


# ============================================================
# TIME-DOMAIN FEATURES
# ============================================================

def time_features(
    signal,
    prefix
):

    features = {}

    if len(signal) == 0:

        return features

    absolute_signal = np.abs(
        signal
    )

    features[
        f"{prefix}_mean"
    ] = np.mean(
        signal
    )

    features[
        f"{prefix}_std"
    ] = np.std(
        signal
    )

    features[
        f"{prefix}_rms"
    ] = np.sqrt(
        np.mean(
            signal ** 2
        )
    )

    features[
        f"{prefix}_peak"
    ] = np.max(
        absolute_signal
    )

    features[
        f"{prefix}_peak_to_peak"
    ] = (
        np.max(signal) -
        np.min(signal)
    )

    features[
        f"{prefix}_abs_mean"
    ] = np.mean(
        absolute_signal
    )

    features[
        f"{prefix}_energy"
    ] = np.sum(
        signal ** 2
    )

    return features


# ============================================================
# FFT FEATURES
# ============================================================

def fft_features(
    signal,
    fs,
    prefix
):

    # IMPORTANT:
    # Create a dictionary for this FFT feature group.
    features = {}

    n = len(signal)

    if n < 4:

        return features

    # Remove DC component
    signal = (
        signal -
        np.mean(signal)
    )

    # Hann window
    window = windows.hann(
        n
    )

    windowed = (
        signal *
        window
    )

    # FFT
    fft_values = np.fft.rfft(
        windowed
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / fs
    )

    magnitude = np.abs(
        fft_values
    )

    # Remove DC
    if len(magnitude) > 1:

        magnitude[0] = 0

    # Limit frequency range
    valid = (
        frequencies <=
        MAX_FREQUENCY
    )

    frequencies_valid = (
        frequencies[valid]
    )

    magnitude_valid = (
        magnitude[valid]
    )

    if len(magnitude_valid) == 0:

        return features

    # --------------------------------------------------------
    # Dominant frequency
    # --------------------------------------------------------

    dominant_index = np.argmax(
        magnitude_valid
    )

    dominant_frequency = (
        frequencies_valid[
            dominant_index
        ]
    )

    dominant_magnitude = (
        magnitude_valid[
            dominant_index
        ]
    )

    # --------------------------------------------------------
    # Spectral energy
    # --------------------------------------------------------

    spectral_energy = np.sum(
        magnitude_valid ** 2
    )

    # --------------------------------------------------------
    # Spectral centroid
    # --------------------------------------------------------

    magnitude_sum = np.sum(
        magnitude_valid
    )

    if magnitude_sum > 0:

        spectral_centroid = (
            np.sum(
                frequencies_valid *
                magnitude_valid
            )
            /
            magnitude_sum
        )

    else:

        spectral_centroid = 0.0

    features[
        f"{prefix}_dominant_frequency"
    ] = dominant_frequency

    features[
        f"{prefix}_dominant_magnitude"
    ] = dominant_magnitude

    features[
        f"{prefix}_spectral_energy"
    ] = spectral_energy

    features[
        f"{prefix}_spectral_centroid"
    ] = spectral_centroid

    # --------------------------------------------------------
    # Frequency bands
    # --------------------------------------------------------

    bands = [

        ("0_10", 0, 10),

        ("10_25", 10, 25),

        ("25_50", 25, 50),

        ("50_75", 50, 75),

        ("75_100", 75, 100),

    ]

    for name, low, high in bands:

        band_mask = (

            (frequencies_valid >= low)

            &

            (frequencies_valid < high)

        )

        band_energy = np.sum(

            magnitude_valid[
                band_mask
            ] ** 2

        )

        features[
            f"{prefix}_energy_{name}hz"
        ] = band_energy

    return features


# ============================================================
# SECONDARY PEAK FEATURES
# ============================================================

def secondary_peak_features(
    magnitude,
    fs,
    primary_index
):

    primary_amplitude = np.max(
        np.abs(magnitude)
    )

    if primary_amplitude <= 0:

        return {
            "secondary_peak": 0.0,
            "secondary_primary_ratio": 0.0,
            "secondary_delay_ms": 0.0,
            "number_of_secondary_peaks": 0,
        }

    # Start looking 80 ms after
    # primary impact.

    search_start = (
        primary_index +
        int(
            0.08 * fs
        )
    )

    # Search up to 1.5 seconds
    # after primary.

    search_end = min(
        len(magnitude),
        primary_index +
        int(
            1.5 * fs
        )
    )

    if search_start >= search_end:

        return {
            "secondary_peak": 0.0,
            "secondary_primary_ratio": 0.0,
            "secondary_delay_ms": 0.0,
            "number_of_secondary_peaks": 0,
        }

    section = np.abs(
        magnitude[
            search_start:search_end
        ]
    )

    threshold = (
        0.10 *
        primary_amplitude
    )

    peaks, properties = find_peaks(

        section,

        height=threshold,

        distance=max(
            1,
            int(
                0.08 * fs
            )
        )

    )

    if len(peaks) == 0:

        return {
            "secondary_peak": 0.0,
            "secondary_primary_ratio": 0.0,
            "secondary_delay_ms": 0.0,
            "number_of_secondary_peaks": 0,
        }

    # Select strongest secondary peak

    strongest_index = peaks[
        np.argmax(
            properties["peak_heights"]
        )
    ]

    secondary_amplitude = section[
        strongest_index
    ]

    secondary_absolute_index = (
        search_start +
        strongest_index
    )

    secondary_delay = (
        secondary_absolute_index -
        primary_index
    )

    secondary_delay_ms = (
        secondary_delay /
        fs *
        1000.0
    )

    secondary_primary_ratio = (
        secondary_amplitude /
        primary_amplitude
    )

    return {

        "secondary_peak":
            secondary_amplitude,

        "secondary_primary_ratio":
            secondary_primary_ratio,

        "secondary_delay_ms":
            secondary_delay_ms,

        "number_of_secondary_peaks":
            len(peaks),

    }


# ============================================================
# PROCESS ONE EVENT
# ============================================================

def process_event(
    filename,
    label,
    event_number,
    fs,
    filtered_x,
    filtered_y,
    filtered_z,
    magnitude,
    peak
):

    event = extract_event_window(

        filtered_x,
        filtered_y,
        filtered_z,
        magnitude,
        peak,
        fs

    )

    if event is None:

        return None

    (
        x,
        y,
        z,
        mag,
        start,
        end
    ) = event

    # IMPORTANT:
    # Create a NEW feature dictionary
    # for every event.

    event_features = {}

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    event_features[
        "file"
    ] = os.path.basename(
        filename
    )

    event_features[
        "label"
    ] = label

    event_features[
        "event_number"
    ] = event_number

    event_features[
        "event_time_seconds"
    ] = peak / fs

    event_features[
        "sampling_rate_hz"
    ] = fs

    # --------------------------------------------------------
    # Time-domain features
    # --------------------------------------------------------

    event_features.update(

        time_features(
            x,
            "x"
        )

    )

    event_features.update(

        time_features(
            y,
            "y"
        )

    )

    event_features.update(

        time_features(
            z,
            "z"
        )

    )

    event_features.update(

        time_features(
            mag,
            "magnitude"
        )

    )

    # --------------------------------------------------------
    # FFT features
    # --------------------------------------------------------

    event_features.update(

        fft_features(
            x,
            fs,
            "x_fft"
        )

    )

    event_features.update(

        fft_features(
            y,
            fs,
            "y_fft"
        )

    )

    event_features.update(

        fft_features(
            z,
            fs,
            "z_fft"
        )

    )

    event_features.update(

        fft_features(
            mag,
            fs,
            "magnitude_fft"
        )

    )

    # --------------------------------------------------------
    # Secondary impact / ring-down features
    # --------------------------------------------------------

    event_features.update(

        secondary_peak_features(
            mag,
            fs,
            peak - start
        )

    )

    return event_features


# ============================================================
# DIAGNOSTIC PLOT
# ============================================================

def save_diagnostic_plot(
    filename,
    label,
    fs,
    magnitude,
    peaks,
    threshold,
    smoothed,
    plot_directory
):

    time_axis = (
        np.arange(
            len(magnitude)
        )
        /
        fs
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time_axis,
        magnitude,
        label="Magnitude"
    )

    plt.plot(
        time_axis,
        smoothed,
        label="Smoothed"
    )

    plt.axhline(
        threshold,
        linestyle="--",
        label="Detection threshold"
    )

    if len(peaks) > 0:

        plt.scatter(
            peaks / fs,
            smoothed[peaks],
            marker="x",
            s=80,
            label="Detected events"
        )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Acceleration magnitude (g)"
    )

    plt.title(
        f"{label} - {os.path.basename(filename)}"
    )

    plt.legend()

    plt.tight_layout()

    output_name = (
        os.path.splitext(
            os.path.basename(filename)
        )[0]
        +
        ".png"
    )

    output_path = os.path.join(
        plot_directory,
        output_name
    )

    plt.savefig(
        output_path,
        dpi=120
    )

    plt.close()


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    filename,
    label,
    plot_directory
):

    print()
    print(
        f"Processing: {filename}"
    )

    print(
        f"Label: {label}"
    )

    timestamp, x, y, z = load_csv(
        filename
    )

    print(
        f"Samples: {len(x)}"
    )

    # Robust sampling rate.
    # Corrupted timestamp jumps are ignored.

    fs = estimate_sampling_rate(
        timestamp
    )

    print(
        f"Sampling rate: {fs:.2f} Hz"
    )

    # Duration based on sample count,
    # not first/last timestamp.
    #
    # This avoids corrupted timestamps
    # producing durations such as 2003 seconds.

    duration = (
        len(x) /
        fs
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    # --------------------------------------------------------
    # Remove gravity / slow movement
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Magnitude
    # --------------------------------------------------------

    magnitude = calculate_magnitude(

        filtered_x,
        filtered_y,
        filtered_z

    )

    # --------------------------------------------------------
    # Detect events
    # --------------------------------------------------------

    (
        peaks,
        threshold,
        smoothed
    ) = detect_events(

        magnitude,
        fs

    )

    print(
        f"Detected events: {len(peaks)}"
    )

    # --------------------------------------------------------
    # Diagnostic plot
    # --------------------------------------------------------

    save_diagnostic_plot(

        filename,
        label,
        fs,
        magnitude,
        peaks,
        threshold,
        smoothed,
        plot_directory

    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    results = []

    for event_number, peak in enumerate(

        peaks,

        start=1

    ):

        event_features = process_event(

            filename,
            label,
            event_number,
            fs,
            filtered_x,
            filtered_y,
            filtered_z,
            magnitude,
            peak

        )

        if event_features is not None:

            results.append(
                event_features
            )

    return results


# ============================================================
# FIND DATASET FILES
# ============================================================

def get_dataset_files():

    files = []

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    background_files = glob.glob(

        os.path.join(
            BACKGROUND_DIRECTORY,
            "*.csv"
        )

    )

    for filename in background_files:

        files.append(
            (
                filename,
                "BACKGROUND"
            )
        )

    # --------------------------------------------------------
    # Impact classes
    # --------------------------------------------------------

    class_directories = {

        "1rup": "RUPEE_1",

        "2rup": "RUPEE_2",

        "5rup": "RUPEE_5",

        "10rup": "RUPEE_10",

        "20rup": "RUPEE_20",

    }

    for directory, label in class_directories.items():

        folder = os.path.join(

            IMPACT_DIRECTORY,
            directory

        )

        if not os.path.exists(folder):

            continue

        impact_files = glob.glob(

            os.path.join(
                folder,
                "*.csv"
            )

        )

        for filename in impact_files:

            # IMPORTANT:
            #
            # Only use the new ±4g recordings.
            #
            # Old ±2g recordings are skipped.

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
        "       VIBRATION DATA PREPROCESSING"
    )

    print(
        "=============================================="
    )

    print()

    dataset_files = get_dataset_files()

    if len(dataset_files) == 0:

        print(
            "ERROR: No CSV files found."
        )

        return

    print(
        f"Files found: {len(dataset_files)}"
    )

    print()

    all_features = []

    # --------------------------------------------------------
    # Process every file
    # --------------------------------------------------------

    for filename, label in dataset_files:

        try:

            results = process_file(

                filename,
                label,
                PLOT_DIRECTORY

            )

            all_features.extend(
                results
            )

        except Exception as e:

            print(
                f"ERROR: {e}"
            )

    # --------------------------------------------------------
    # Save feature dataset
    # --------------------------------------------------------

    if len(all_features) == 0:

        print()
        print(
            "=============================================="
        )

        print(
            "No events were successfully processed."
        )

        print(
            "=============================================="
        )

        return

    features_df = pd.DataFrame(
        all_features
    )

    features_df.to_csv(
        FEATURE_FILE,
        index=False
    )

    print()
    print(
        "=============================================="
    )

    print(
        "       PREPROCESSING COMPLETE"
    )

    print(
        "=============================================="
    )

    print()

    print(
        f"Events extracted: {len(features_df)}"
    )

    print(
        f"Features: {len(features_df.columns)}"
    )

    print()

    print(
        f"Saved:"
    )

    print(
        f"  {FEATURE_FILE}"
    )

    print()

    print(
        f"Diagnostic plots:"
    )

    print(
        f"  {PLOT_DIRECTORY}"
    )

    print()

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    if "label" in features_df.columns:

        print(
            "Events by class:"
        )

        print(
            features_df[
                "label"
            ].value_counts()
        )

    print()

    print(
        "=============================================="
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
