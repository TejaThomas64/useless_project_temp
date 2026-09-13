import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ============================================================
# CONFIGURATION
# ============================================================

X_FILE = "X.npy"
METADATA_FILE = "metadata.csv"

OUTPUT_FILE = "post_impact_features.csv"

FS = 200.0

# Event window currently contains:
# 0.5 sec before impact
# 1.5 sec after impact

PRE_EVENT_SECONDS = 0.5

# Post-impact regions
SEGMENTS = [
    ("post_0_100ms", 0.0, 0.10),
    ("post_100_200ms", 0.10, 0.20),
    ("post_200_500ms", 0.20, 0.50),
    ("post_500_1000ms", 0.50, 1.00),
    ("post_1000_1500ms", 1.00, 1.50),
]

# Secondary impact detection
SECONDARY_START_SECONDS = 0.08
SECONDARY_END_SECONDS = 1.50

# Minimum separation between secondary peaks
MIN_SECONDARY_DISTANCE_SECONDS = 0.08

# Secondary peaks must have this prominence
# relative to the noise level.
NOISE_MAD_MULTIPLIER = 5.0


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    X = np.load(X_FILE)

    metadata = pd.read_csv(
        METADATA_FILE
    )

    print()
    print("Loaded data")
    print("-----------------------------")
    print("X shape:", X.shape)
    print("Metadata shape:", metadata.shape)

    return X, metadata


# ============================================================
# MAGNITUDE
# ============================================================

def calculate_magnitude(event):

    x = event[0]
    y = event[1]
    z = event[2]

    return np.sqrt(
        x ** 2 +
        y ** 2 +
        z ** 2
    )


# ============================================================
# RMS
# ============================================================

def rms(signal):

    return np.sqrt(
        np.mean(
            signal ** 2
        )
    )


# ============================================================
# ENERGY
# ============================================================

def energy(signal):

    return np.sum(
        signal ** 2
    )


# ============================================================
# NOISE LEVEL
#
# Estimate noise from the pre-impact region.
# ============================================================

def estimate_noise(signal, peak_index):

    pre_event = signal[
        :peak_index
    ]

    if len(pre_event) < 10:

        return 0.0

    median = np.median(
        pre_event
    )

    mad = np.median(
        np.abs(
            pre_event -
            median
        )
    )

    return 1.4826 * mad


# ============================================================
# FIND PRIMARY PEAK
#
# The metadata peak is relative to the original recording.
# In X.npy, the impact should normally be around:
#
# 0.5 sec * 200 Hz = sample 100
#
# We locate the strongest magnitude peak around that region.
# ============================================================

def find_primary_peak(magnitude):

    expected_peak = int(
        PRE_EVENT_SECONDS * FS
    )

    search_start = max(
        0,
        expected_peak - int(0.15 * FS)
    )

    search_end = min(
        len(magnitude),
        expected_peak + int(0.15 * FS)
    )

    region = magnitude[
        search_start:search_end
    ]

    if len(region) == 0:

        return expected_peak

    local_peak = np.argmax(
        region
    )

    return (
        search_start +
        local_peak
    )


# ============================================================
# SEGMENT FEATURES
# ============================================================

def extract_segment_features(
    signal,
    primary_peak
):

    features = {}

    for name, start_sec, end_sec in SEGMENTS:

        start = (
            primary_peak +
            int(start_sec * FS)
        )

        end = (
            primary_peak +
            int(end_sec * FS)
        )

        start = max(
            0,
            start
        )

        end = min(
            len(signal),
            end
        )

        segment = signal[
            start:end
        ]

        if len(segment) == 0:

            features[
                name + "_rms"
            ] = 0.0

            features[
                name + "_energy"
            ] = 0.0

            continue

        features[
            name + "_rms"
        ] = rms(
            segment
        )

        features[
            name + "_energy"
        ] = energy(
            segment
        )

    return features


# ============================================================
# DECAY FEATURES
# ============================================================

def extract_decay_features(
    magnitude,
    primary_peak
):

    features = {}

    primary_value = abs(
        magnitude[primary_peak]
    )

    if primary_value <= 0:

        primary_value = 1e-9

    # RMS immediately after impact
    windows = [
        ("decay_0_100ms", 0.0, 0.10),
        ("decay_100_200ms", 0.10, 0.20),
        ("decay_200_500ms", 0.20, 0.50),
        ("decay_500_1000ms", 0.50, 1.00),
        ("decay_1000_1500ms", 1.00, 1.50),
    ]

    for name, start_sec, end_sec in windows:

        start = (
            primary_peak +
            int(start_sec * FS)
        )

        end = (
            primary_peak +
            int(end_sec * FS)
        )

        end = min(
            len(magnitude),
            end
        )

        segment = magnitude[
            start:end
        ]

        if len(segment) == 0:

            features[name] = 0.0

        else:

            features[name] = (
                rms(segment) /
                primary_value
            )

    return features


# ============================================================
# SECONDARY IMPACT FEATURES
# ============================================================

def extract_secondary_features(
    magnitude,
    primary_peak
):

    features = {}

    primary_amplitude = abs(
        magnitude[primary_peak]
    )

    if primary_amplitude <= 0:

        primary_amplitude = 1e-9

    # Estimate noise before primary impact
    noise = estimate_noise(
        magnitude,
        primary_peak
    )

    # Absolute minimum prominence
    minimum_prominence = max(
        noise * NOISE_MAD_MULTIPLIER,
        primary_amplitude * 0.03
    )

    start = (
        primary_peak +
        int(
            SECONDARY_START_SECONDS *
            FS
        )
    )

    end = (
        primary_peak +
        int(
            SECONDARY_END_SECONDS *
            FS
        )
    )

    end = min(
        len(magnitude),
        end
    )

    if end <= start:

        return {
            "secondary_count": 0,
            "secondary_largest_amplitude": 0.0,
            "secondary_largest_ratio": 0.0,
            "secondary_delay_ms": 0.0,
            "secondary_energy": 0.0,
        }

    post_signal = magnitude[
        start:end
    ]

    distance = max(
        1,
        int(
            MIN_SECONDARY_DISTANCE_SECONDS *
            FS
        )
    )

    peaks, properties = find_peaks(
        post_signal,
        prominence=minimum_prominence,
        distance=distance
    )

    if len(peaks) == 0:

        return {
            "secondary_count": 0,
            "secondary_largest_amplitude": 0.0,
            "secondary_largest_ratio": 0.0,
            "secondary_delay_ms": 0.0,
            "secondary_energy": energy(
                post_signal
            ),
        }

    amplitudes = np.abs(
        post_signal[peaks]
    )

    largest_index = np.argmax(
        amplitudes
    )

    largest_amplitude = (
        amplitudes[largest_index]
    )

    largest_peak = (
        peaks[largest_index] +
        start
    )

    delay_ms = (
        (largest_peak - primary_peak) /
        FS *
        1000
    )

    return {
        "secondary_count":
            len(peaks),

        "secondary_largest_amplitude":
            largest_amplitude,

        "secondary_largest_ratio":
            largest_amplitude /
            primary_amplitude,

        "secondary_delay_ms":
            delay_ms,

        "secondary_energy":
            energy(
                post_signal
            ),
    }


# ============================================================
# PRIMARY IMPACT FEATURES
# ============================================================

def extract_primary_features(
    magnitude,
    primary_peak
):

    start = max(
        0,
        primary_peak - int(0.05 * FS)
    )

    end = min(
        len(magnitude),
        primary_peak + int(0.05 * FS)
    )

    region = magnitude[
        start:end
    ]

    return {
        "primary_peak":
            abs(
                magnitude[
                    primary_peak
                ]
            ),

        "primary_rms":
            rms(region),

        "primary_energy":
            energy(region),
    }


# ============================================================
# PROCESS ONE EVENT
# ============================================================

def process_event(event):

    magnitude = calculate_magnitude(
        event
    )

    primary_peak = find_primary_peak(
        magnitude
    )

    features = {}

    features.update(
        extract_primary_features(
            magnitude,
            primary_peak
        )
    )

    features.update(
        extract_segment_features(
            magnitude,
            primary_peak
        )
    )

    features.update(
        extract_decay_features(
            magnitude,
            primary_peak
        )
    )

    features.update(
        extract_secondary_features(
            magnitude,
            primary_peak
        )
    )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=============================================="
    )
    print(
        "       POST-IMPACT FEATURE EXTRACTION"
    )
    print(
        "=============================================="
    )

    X, metadata = load_data()

    if len(X) != len(metadata):

        raise ValueError(
            "X.npy and metadata.csv have "
            "different numbers of events."
        )

    all_features = []

    for i in range(len(X)):

        if i % 10 == 0:

            print(
                f"Processing event "
                f"{i + 1}/{len(X)}"
            )

        event_features = process_event(
            X[i]
        )

        all_features.append(
            event_features
        )

    features_df = pd.DataFrame(
        all_features
    )

    # Keep metadata with every feature row
    result = pd.concat(
        [
            metadata.reset_index(
                drop=True
            ),
            features_df
        ],
        axis=1
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        "=============================================="
    )
    print(
        "       FEATURE EXTRACTION COMPLETE"
    )
    print(
        "=============================================="
    )

    print()
    print(
        "Events:",
        len(result)
    )

    print(
        "New features:",
        len(features_df.columns)
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Feature columns:"
    )

    for column in features_df.columns:

        print(
            " ",
            column
        )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
