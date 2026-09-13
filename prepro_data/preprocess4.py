import os
import glob
import warnings

import numpy as np
import pandas as pd
from scipy import signal, stats

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "/home/nathan/useless/datasets"

OUTPUT_DIR = os.path.join(DATASET_ROOT, "processed_v2")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "features_v2.csv")

# Only use the new ±4g recordings
SEARCH_PATTERNS = [
    os.path.join(DATASET_ROOT, "impact", "*", "*_4g.csv"),
    os.path.join(DATASET_ROOT, "impact", "*_4g.csv"),
]

# Expected sensor sampling rate
DEFAULT_FS = 200.0

# High-pass filter removes gravity / very slow movement
HIGH_PASS_CUTOFF = 2.0

# Event window
EVENT_DURATION = 2.0
PRE_EVENT_TIME = 0.5

# Detection parameters
SMOOTHING_WINDOW = 11
THRESHOLD_MULTIPLIER = 4.0

# Minimum distance between detected events
MIN_EVENT_DISTANCE = 0.8

# Frequency bands
FREQ_BANDS = [
    (0.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, 40.0),
    (40.0, 80.0),
    (80.0, 100.0),
]


# ============================================================
# LABEL MAPPING
# ============================================================

def label_from_filename(filename):
    """
    Convert filenames into denomination labels.

    impact1_4g.csv  -> RUPEE_1
    impact2_4g.csv  -> RUPEE_1
    impact3_4g.csv  -> RUPEE_1

    impact4_4g.csv  -> RUPEE_2
    impact5_4g.csv  -> RUPEE_2
    impact6_4g.csv  -> RUPEE_2

    impact7_4g.csv  -> RUPEE_5
    impact8_4g.csv  -> RUPEE_5
    impact9_4g.csv  -> RUPEE_5

    impact10_4g.csv -> RUPEE_10
    impact11_4g.csv -> RUPEE_10
    impact12_4g.csv -> RUPEE_10

    impact13_4g.csv -> RUPEE_20
    impact14_4g.csv -> RUPEE_20
    """

    name = os.path.basename(filename).lower()

    number = None

    for i in range(1, 15):
        if f"impact{i}_4g.csv" == name:
            number = i
            break

    if number is None:
        return None

    if number <= 3:
        return "RUPEE_1"
    elif number <= 6:
        return "RUPEE_2"
    elif number <= 9:
        return "RUPEE_5"
    elif number <= 12:
        return "RUPEE_10"
    else:
        return "RUPEE_20"


# ============================================================
# LOAD DATA
# ============================================================

def load_csv(filename):
    """
    Load one ESP32 CSV recording.
    """

    df = pd.read_csv(filename)

    required = ["timestamp_us", "x_g", "y_g", "z_g"]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Missing column '{column}' in {filename}"
            )

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna()

    if len(df) < 100:
        raise ValueError(
            f"Not enough samples in {filename}"
        )

    return df


# ============================================================
# ROBUST SAMPLING RATE
# ============================================================

def estimate_sampling_rate(timestamp_us):
    """
    Estimate sampling rate using the median timestamp difference.

    This intentionally ignores timestamp glitches.
    """

    timestamps = np.asarray(timestamp_us)

    dt = np.diff(timestamps)

    # Keep only plausible positive intervals
    dt = dt[
        (dt > 1000) &
        (dt < 20000)
    ]

    if len(dt) == 0:
        return DEFAULT_FS

    median_dt = np.median(dt)

    if median_dt <= 0:
        return DEFAULT_FS

    fs = 1_000_000.0 / median_dt

    # Reject obviously unreasonable estimates
    if fs < 50 or fs > 500:
        return DEFAULT_FS

    return fs


# ============================================================
# HIGH PASS FILTER
# ============================================================

def highpass_filter(x, fs):
    """
    Remove gravity and very slow movement.
    """

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
        filtered = signal.filtfilt(
            b,
            a,
            x
        )
    except ValueError:
        filtered = x - np.mean(x)

    return filtered


# ============================================================
# EVENT DETECTION
# ============================================================

def detect_events(x, y, z, fs):
    """
    Detect impact regions automatically.

    Magnitude is used ONLY for locating events.

    Feature extraction later keeps X/Y/Z separately.
    """

    magnitude = np.sqrt(
        x ** 2 +
        y ** 2 +
        z ** 2
    )

    # Smooth magnitude for detection
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

    # Robust baseline
    median = np.median(smooth)

    mad = np.median(
        np.abs(smooth - median)
    )

    if mad < 1e-8:
        mad = np.std(smooth) + 1e-8

    threshold = (
        median +
        THRESHOLD_MULTIPLIER * mad
    )

    candidates = np.where(
        smooth > threshold
    )[0]

    if len(candidates) == 0:
        # Fall back to strongest peaks
        peaks, _ = signal.find_peaks(
            smooth,
            distance=int(fs * MIN_EVENT_DISTANCE),
            prominence=max(
                np.std(smooth),
                0.01
            )
        )
    else:
        # Convert threshold crossings into peaks
        peaks, _ = signal.find_peaks(
            smooth,
            height=threshold,
            distance=int(fs * MIN_EVENT_DISTANCE)
        )

    # If still no events, use strongest peaks
    if len(peaks) == 0:
        peaks, _ = signal.find_peaks(
            smooth,
            distance=int(fs * MIN_EVENT_DISTANCE),
            prominence=max(
                np.std(smooth) * 0.5,
                0.005
            )
        )

    return peaks, magnitude, smooth


# ============================================================
# BASIC STATISTICS
# ============================================================

def basic_features(x):
    """
    General statistical features for one axis.
    """

    features = {}

    if len(x) == 0:
        return features

    mean = np.mean(x)
    std = np.std(x)

    rms = np.sqrt(
        np.mean(x ** 2)
    )

    abs_x = np.abs(x)

    features["mean"] = mean
    features["std"] = std
    features["variance"] = np.var(x)

    features["rms"] = rms

    features["min"] = np.min(x)
    features["max"] = np.max(x)

    features["abs_peak"] = np.max(abs_x)

    features["positive_peak"] = np.max(x)
    features["negative_peak"] = np.min(x)

    features["peak_to_peak"] = (
        np.max(x) - np.min(x)
    )

    # Total energy
    features["energy"] = np.sum(x ** 2)

    # Mean absolute value
    features["mean_absolute"] = np.mean(abs_x)

    # Crest factor
    if rms > 1e-10:
        features["crest_factor"] = (
            np.max(abs_x) / rms
        )
    else:
        features["crest_factor"] = 0.0

    # Shape statistics
    if std > 1e-10:
        features["skewness"] = stats.skew(x)
        features["kurtosis"] = stats.kurtosis(x)
    else:
        features["skewness"] = 0.0
        features["kurtosis"] = 0.0

    # Quartiles
    features["q25"] = np.percentile(x, 25)
    features["median"] = np.percentile(x, 50)
    features["q75"] = np.percentile(x, 75)

    features["iqr"] = (
        features["q75"] -
        features["q25"]
    )

    # Absolute statistics
    features["abs_mean"] = np.mean(abs_x)
    features["abs_std"] = np.std(abs_x)

    return features


# ============================================================
# TIME DOMAIN FEATURES
# ============================================================

def time_features(x, fs):
    """
    Features describing how the vibration behaves over time.
    """

    features = {}

    if len(x) == 0:
        return features

    abs_x = np.abs(x)

    peak_index = np.argmax(abs_x)

    # Time at maximum amplitude
    features["peak_time"] = (
        peak_index / fs
    )

    # First half vs second half energy
    midpoint = len(x) // 2

    first = x[:midpoint]
    second = x[midpoint:]

    if len(first) > 0:
        features["first_half_energy"] = np.sum(
            first ** 2
        )
    else:
        features["first_half_energy"] = 0.0

    if len(second) > 0:
        features["second_half_energy"] = np.sum(
            second ** 2
        )
    else:
        features["second_half_energy"] = 0.0

    total_energy = np.sum(x ** 2)

    if total_energy > 1e-10:
        features["second_first_energy_ratio"] = (
            features["second_half_energy"] /
            total_energy
        )
    else:
        features["second_first_energy_ratio"] = 0.0

    # Zero crossings
    centered = x - np.mean(x)

    zero_crossings = np.sum(
        np.diff(
            np.signbit(centered)
        )
    )

    features["zero_crossings"] = (
        zero_crossings
    )

    features["zero_crossing_rate"] = (
        zero_crossings / len(x)
    )

    # Absolute area
    features["absolute_area"] = (
        np.sum(abs_x) / fs
    )

    # Signal duration above different amplitude levels
    peak = np.max(abs_x)

    if peak > 1e-10:

        for fraction in [0.25, 0.5, 0.75]:

            threshold = peak * fraction

            count = np.sum(
                abs_x >= threshold
            )

            features[
                f"time_above_{int(fraction * 100)}pct"
            ] = count / fs

    return features


# ============================================================
# ATTACK / DECAY FEATURES
# ============================================================

def attack_decay_features(x, fs):
    """
    Describe how quickly the vibration rises and decays.
    """

    features = {}

    abs_x = np.abs(x)

    if len(abs_x) == 0:
        return features

    peak = np.max(abs_x)

    if peak < 1e-10:
        return features

    peak_index = np.argmax(abs_x)

    # Find first time signal crosses 25%, 50%, 75%
    for fraction in [0.25, 0.5, 0.75]:

        threshold = peak * fraction

        before_peak = abs_x[:peak_index + 1]

        indices = np.where(
            before_peak >= threshold
        )[0]

        if len(indices) > 0:
            attack_index = indices[0]

            features[
                f"attack_time_{int(fraction * 100)}"
            ] = (
                peak_index - attack_index
            ) / fs
        else:
            features[
                f"attack_time_{int(fraction * 100)}"
            ] = 0.0

        after_peak = abs_x[peak_index:]

        indices = np.where(
            after_peak <= threshold
        )[0]

        if len(indices) > 0:
            decay_index = (
                peak_index +
                indices[0]
            )

            features[
                f"decay_time_{int(fraction * 100)}"
            ] = (
                decay_index - peak_index
            ) / fs
        else:
            features[
                f"decay_time_{int(fraction * 100)}"
            ] = (
                len(x) - peak_index
            ) / fs

    return features


# ============================================================
# FFT FEATURES
# ============================================================

def fft_features(x, fs):
    """
    Frequency-domain features.
    """

    features = {}

    n = len(x)

    if n < 4:
        return features

    # Remove mean before FFT
    x_centered = x - np.mean(x)

    window = np.hanning(n)

    spectrum = np.fft.rfft(
        x_centered * window
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / fs
    )

    magnitude = np.abs(spectrum)

    power = magnitude ** 2

    # Ignore DC
    if len(power) > 1:
        power_no_dc = power[1:]
        freq_no_dc = frequencies[1:]
    else:
        power_no_dc = power
        freq_no_dc = frequencies

    total_power = np.sum(power_no_dc)

    features["spectral_energy"] = (
        total_power
    )

    # Dominant frequency
    if len(power_no_dc) > 0:

        dominant_index = np.argmax(
            power_no_dc
        )

        features["dominant_frequency"] = (
            freq_no_dc[dominant_index]
        )

    else:
        features["dominant_frequency"] = 0.0

    # Spectral centroid
    if total_power > 1e-10:

        features["spectral_centroid"] = (
            np.sum(
                freq_no_dc * power_no_dc
            ) / total_power
        )

    else:
        features["spectral_centroid"] = 0.0

    # Spectral bandwidth
    centroid = features[
        "spectral_centroid"
    ]

    if total_power > 1e-10:

        features["spectral_bandwidth"] = np.sqrt(
            np.sum(
                ((freq_no_dc - centroid) ** 2)
                * power_no_dc
            ) / total_power
        )

    else:
        features["spectral_bandwidth"] = 0.0

    # Spectral rolloff
    if total_power > 1e-10:

        cumulative = np.cumsum(
            power_no_dc
        )

        target = (
            0.85 * total_power
        )

        rolloff_index = np.searchsorted(
            cumulative,
            target
        )

        rolloff_index = min(
            rolloff_index,
            len(freq_no_dc) - 1
        )

        features["spectral_rolloff_85"] = (
            freq_no_dc[rolloff_index]
        )

    else:
        features["spectral_rolloff_85"] = 0.0

    # Frequency bands
    for low, high in FREQ_BANDS:

        mask = (
            (freq_no_dc >= low) &
            (freq_no_dc < high)
        )

        band_energy = np.sum(
            power_no_dc[mask]
        )

        band_name = (
            f"{int(low)}_{int(high)}hz"
        )

        features[
            f"band_energy_{band_name}"
        ] = band_energy

        if total_power > 1e-10:

            features[
                f"band_ratio_{band_name}"
            ] = (
                band_energy /
                total_power
            )

        else:

            features[
                f"band_ratio_{band_name}"
            ] = 0.0

    # Spectral entropy
    if total_power > 1e-10:

        probabilities = (
            power_no_dc /
            total_power
        )

        probabilities = probabilities[
            probabilities > 0
        ]

        entropy = -np.sum(
            probabilities *
            np.log2(probabilities)
        )

        features[
            "spectral_entropy"
        ] = entropy

    else:

        features[
            "spectral_entropy"
        ] = 0.0

    return features


# ============================================================
# ZERO CROSSING / OSCILLATION FEATURES
# ============================================================

def oscillation_features(x, fs):
    """
    Describe oscillatory behaviour after impact.
    """

    features = {}

    if len(x) < 10:
        return features

    centered = x - np.mean(x)

    # Positive/negative changes
    diff = np.diff(centered)

    features["mean_abs_derivative"] = (
        np.mean(np.abs(diff))
    )

    features["std_derivative"] = (
        np.std(diff)
    )

    # Number of local peaks
    prominence = max(
        np.std(centered) * 0.3,
        1e-6
    )

    peaks, _ = signal.find_peaks(
        np.abs(centered),
        prominence=prominence,
        distance=max(
            1,
            int(fs * 0.02)
        )
    )

    features["num_abs_peaks"] = (
        len(peaks)
    )

    # Peak density
    features["peak_density"] = (
        len(peaks) /
        (len(x) / fs)
    )

    return features


# ============================================================
# PER-AXIS FEATURE EXTRACTION
# ============================================================

def extract_axis_features(x, fs, prefix):
    """
    Extract all V2 features for one axis.
    """

    features = {}

    basic = basic_features(x)

    for key, value in basic.items():
        features[f"{prefix}_{key}"] = value

    time = time_features(
        x,
        fs
    )

    for key, value in time.items():
        features[f"{prefix}_{key}"] = value

    attack_decay = attack_decay_features(
        x,
        fs
    )

    for key, value in attack_decay.items():
        features[f"{prefix}_{key}"] = value

    fft = fft_features(
        x,
        fs
    )

    for key, value in fft.items():
        features[f"{prefix}_{key}"] = value

    oscillation = oscillation_features(
        x,
        fs
    )

    for key, value in oscillation.items():
        features[f"{prefix}_{key}"] = value

    return features


# ============================================================
# CROSS-AXIS FEATURES
# ============================================================

def cross_axis_features(x, y, z):
    """
    Features describing relationships between X/Y/Z.
    """

    features = {}

    # Correlations
    if (
        np.std(x) > 1e-10 and
        np.std(y) > 1e-10
    ):
        features["corr_xy"] = np.corrcoef(
            x, y
        )[0, 1]
    else:
        features["corr_xy"] = 0.0

    if (
        np.std(x) > 1e-10 and
        np.std(z) > 1e-10
    ):
        features["corr_xz"] = np.corrcoef(
            x, z
        )[0, 1]
    else:
        features["corr_xz"] = 0.0

    if (
        np.std(y) > 1e-10 and
        np.std(z) > 1e-10
    ):
        features["corr_yz"] = np.corrcoef(
            y, z
        )[0, 1]
    else:
        features["corr_yz"] = 0.0

    # Axis energies
    energy_x = np.sum(x ** 2)
    energy_y = np.sum(y ** 2)
    energy_z = np.sum(z ** 2)

    total_energy = (
        energy_x +
        energy_y +
        energy_z
    )

    features["xyz_total_energy"] = (
        total_energy
    )

    if total_energy > 1e-10:

        features["x_energy_ratio"] = (
            energy_x / total_energy
        )

        features["y_energy_ratio"] = (
            energy_y / total_energy
        )

        features["z_energy_ratio"] = (
            energy_z / total_energy
        )

    else:

        features["x_energy_ratio"] = 0.0
        features["y_energy_ratio"] = 0.0
        features["z_energy_ratio"] = 0.0

    # Relative RMS
    rms_x = np.sqrt(np.mean(x ** 2))
    rms_y = np.sqrt(np.mean(y ** 2))
    rms_z = np.sqrt(np.mean(z ** 2))

    features["rms_x_over_y"] = (
        rms_x / (rms_y + 1e-10)
    )

    features["rms_x_over_z"] = (
        rms_x / (rms_z + 1e-10)
    )

    features["rms_y_over_z"] = (
        rms_y / (rms_z + 1e-10)
    )

    # Peak ratios
    peak_x = np.max(np.abs(x))
    peak_y = np.max(np.abs(y))
    peak_z = np.max(np.abs(z))

    features["peak_x_over_y"] = (
        peak_x / (peak_y + 1e-10)
    )

    features["peak_x_over_z"] = (
        peak_x / (peak_z + 1e-10)
    )

    features["peak_y_over_z"] = (
        peak_y / (peak_z + 1e-10)
    )

    # Overall acceleration magnitude
    magnitude = np.sqrt(
        x ** 2 +
        y ** 2 +
        z ** 2
    )

    mag_basic = basic_features(
        magnitude
    )

    for key, value in mag_basic.items():
        features[f"magnitude_{key}"] = value

    return features


# ============================================================
# PROCESS ONE EVENT
# ============================================================

def process_event(
    x,
    y,
    z,
    peak_index,
    fs,
    event_number
):
    """
    Extract a fixed-duration event around an impact.
    """

    window_samples = int(
        EVENT_DURATION * fs
    )

    pre_samples = int(
        PRE_EVENT_TIME * fs
    )

    start = peak_index - pre_samples

    end = start + window_samples

    # Need enough samples
    if start < 0:
        return None

    if end > len(x):
        return None

    event_x = x[start:end]
    event_y = y[start:end]
    event_z = z[start:end]

    # Make sure exact length is maintained
    if len(event_x) != window_samples:
        return None

    event_features = {}

    # --------------------------------------------------------
    # X
    # --------------------------------------------------------

    x_features = extract_axis_features(
        event_x,
        fs,
        "x"
    )

    event_features.update(
        x_features
    )

    # --------------------------------------------------------
    # Y
    # --------------------------------------------------------

    y_features = extract_axis_features(
        event_y,
        fs,
        "y"
    )

    event_features.update(
        y_features
    )

    # --------------------------------------------------------
    # Z
    # --------------------------------------------------------

    z_features = extract_axis_features(
        event_z,
        fs,
        "z"
    )

    event_features.update(
        z_features
    )

    # --------------------------------------------------------
    # Cross-axis
    # --------------------------------------------------------

    cross_features = cross_axis_features(
        event_x,
        event_y,
        event_z
    )

    event_features.update(
        cross_features
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    event_features[
        "event_number"
    ] = event_number

    event_features[
        "event_sample"
    ] = peak_index

    event_features[
        "event_time_seconds"
    ] = peak_index / fs

    event_features[
        "sampling_rate_hz"
    ] = fs

    return event_features


# ============================================================
# PROCESS ONE RECORDING
# ============================================================

def process_recording(filename):
    """
    Process a complete recording.
    """

    print()
    print("----------------------------------------")
    print(
        "Processing:",
        os.path.basename(filename)
    )
    print("----------------------------------------")

    label = label_from_filename(
        filename
    )

    if label is None:
        print("Could not determine label.")
        return []

    try:
        df = load_csv(filename)
    except Exception as e:
        print("ERROR:", e)
        return []

    timestamp = df[
        "timestamp_us"
    ].values

    x = df["x_g"].values.astype(float)
    y = df["y_g"].values.astype(float)
    z = df["z_g"].values.astype(float)

    fs = estimate_sampling_rate(
        timestamp
    )

    print(
        f"Samples: {len(df)}"
    )

    print(
        f"Estimated sampling rate: {fs:.2f} Hz"
    )

    # --------------------------------------------------------
    # Filter each axis independently
    # --------------------------------------------------------

    x_filtered = highpass_filter(
        x,
        fs
    )

    y_filtered = highpass_filter(
        y,
        fs
    )

    z_filtered = highpass_filter(
        z,
        fs
    )

    # --------------------------------------------------------
    # Detect events using magnitude
    # --------------------------------------------------------

    peaks, magnitude, smooth = detect_events(
        x_filtered,
        y_filtered,
        z_filtered,
        fs
    )

    print(
        f"Detected peaks: {len(peaks)}"
    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    recording_features = []

    for event_number, peak in enumerate(
        peaks,
        start=1
    ):

        event_features = process_event(
            x_filtered,
            y_filtered,
            z_filtered,
            peak,
            fs,
            event_number
        )

        if event_features is None:
            continue

        event_features[
            "file"
        ] = os.path.basename(filename)

        event_features[
            "label"
        ] = label

        recording_features.append(
            event_features
        )

    print(
        f"Valid events extracted: "
        f"{len(recording_features)}"
    )

    return recording_features


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("========================================")
    print("       FEATURE EXTRACTION V2")
    print("========================================")
    print()

    print(
        "Using ONLY ±4g recordings"
    )

    print(
        "Old ±2g recordings are excluded."
    )

    print()

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    files = []

    for pattern in SEARCH_PATTERNS:

        matches = glob.glob(
            pattern
        )

        files.extend(matches)

    # Remove duplicates
    files = sorted(
        list(set(files))
    )

    print(
        f"Found {len(files)} recordings."
    )

    if len(files) == 0:

        print()
        print(
            "ERROR: No _4g.csv files found."
        )

        print(
            "Expected location:"
        )

        print(
            DATASET_ROOT
        )

        return

    # --------------------------------------------------------
    # Show recordings
    # --------------------------------------------------------

    print()

    for filename in files:

        label = label_from_filename(
            filename
        )

        print(
            f"{os.path.basename(filename):25s}"
            f" -> {label}"
        )

    print()

    # --------------------------------------------------------
    # Process everything
    # --------------------------------------------------------

    all_features = []

    for filename in files:

        recording_features = process_recording(
            filename
        )

        all_features.extend(
            recording_features
        )

    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    if len(all_features) == 0:

        print()
        print(
            "ERROR: No events were extracted."
        )

        return

    features_df = pd.DataFrame(
        all_features
    )

    # --------------------------------------------------------
    # Put metadata first
    # --------------------------------------------------------

    metadata_columns = [
        "file",
        "label",
        "event_number",
        "event_sample",
        "event_time_seconds",
        "sampling_rate_hz",
    ]

    other_columns = [
        column
        for column in features_df.columns
        if column not in metadata_columns
    ]

    features_df = features_df[
        metadata_columns +
        other_columns
    ]

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    features_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("========================================")
    print("       V2 PREPROCESSING COMPLETE")
    print("========================================")
    print()

    print(
        f"Events extracted: "
        f"{len(features_df)}"
    )

    print(
        f"Features total: "
        f"{len(features_df.columns)}"
    )

    print(
        f"Actual ML features: "
        f"{len(other_columns)}"
    )

    print()

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
        "Events by recording:"
    )

    print(
        features_df[
            "file"
        ].value_counts()
    )

    print()

    print(
        f"Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()


if __name__ == "__main__":
    main()
