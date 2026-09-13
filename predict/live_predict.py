import time
import socket
import serial
import joblib
import numpy as np

from preprocess import (
    estimate_sampling_rate,
    high_pass_filter,
    calculate_magnitude,
    detect_events,
    process_event
)


# ============================================================
# CONFIG
# ============================================================

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

MODEL_FILE = "models/final_hierarchical_model.pkl"

# Collect this much live data before checking for impacts.
BUFFER_SECONDS = 5.0

EXPECTED_FS = 200.0


# ============================================================
# ETHERNET CONFIG
# ============================================================

# Teammate's Windows physical Ethernet IP
TEAMMATE_IP = "192.168.50.2"

# TCP port used by receiver.py
TEAMMATE_PORT = 5001


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LIVE VIBRATION DENOMINATION DETECTOR")
print("=" * 60)
print()

print("Loading model...")

try:

    model_package = joblib.load(
        MODEL_FILE
    )

except Exception as e:

    print()
    print("ERROR: Could not load model.")
    print(e)
    sys.exit(1)


group_model = model_package["group_model"]
low_model = model_package["low_model"]
high_model = model_package["high_model"]

feature_columns = model_package["feature_columns"]

print(
    f"Loaded {len(feature_columns)} features."
)

print()


# ============================================================
# ETHERNET SENDER
# ============================================================

def send_prediction(prediction):

    print(
        f"Sending prediction to teammate..."
    )

    try:

        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        ) as client:

            client.settimeout(3)

            client.connect(
                (
                    TEAMMATE_IP,
                    TEAMMATE_PORT
                )
            )

            message = prediction + "\n"

            client.sendall(
                message.encode("utf-8")
            )

        print(
            f"Sent successfully: {prediction}"
        )

        return True

    except socket.timeout:

        print(
            "WARNING: Connection to teammate timed out."
        )

        return False

    except ConnectionRefusedError:

        print(
            "WARNING: Teammate receiver refused the connection."
        )

        print(
            "Make sure receiver.py is running on Windows."
        )

        return False

    except OSError as e:

        print(
            f"WARNING: Ethernet send failed: {e}"
        )

        return False


# ============================================================
# SERIAL CONNECTION
# ============================================================

print(
    f"Connecting to ESP32 on {SERIAL_PORT}..."
)

try:

    ser = serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=1
    )

except Exception as e:

    print()
    print("ERROR: Could not open ESP32 serial port.")
    print(e)
    sys.exit(1)


# ============================================================
# GIVE ESP32 TIME TO RESET
# ============================================================

time.sleep(2)

ser.reset_input_buffer()


print()
print("=" * 60)
print("READY")
print("=" * 60)
print()

print("ESP32 connected.")
print(
    f"Ethernet target: "
    f"{TEAMMATE_IP}:{TEAMMATE_PORT}"
)

print()

print("Drop a coin on the plate.")
print()

print("The system will:")
print("  1. Read vibration from ESP32")
print("  2. Detect the impact")
print("  3. Extract V1 features")
print("  4. Classify the denomination")
print("  5. Send prediction over Ethernet")
print()

print("Press Ctrl+C to stop.")
print()


# ============================================================
# SERIAL READER
# ============================================================

def read_samples(duration_seconds):

    timestamps = []
    xs = []
    ys = []
    zs = []

    start_time = time.time()

    while (
        time.time() - start_time
        < duration_seconds
    ):

        line = ser.readline()

        if not line:
            continue

        try:

            text = line.decode(
                "utf-8",
                errors="ignore"
            ).strip()

        except Exception:

            continue

        if not text:
            continue

        if text == "FFT_DATA_START":
            continue

        parts = text.split(",")

        if len(parts) != 4:
            continue

        try:

            timestamp = int(parts[0])

            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

        except ValueError:

            continue

        timestamps.append(timestamp)
        xs.append(x)
        ys.append(y)
        zs.append(z)

    return (
        np.array(
            timestamps,
            dtype=np.int64
        ),

        np.array(
            xs,
            dtype=float
        ),

        np.array(
            ys,
            dtype=float
        ),

        np.array(
            zs,
            dtype=float
        )
    )


# ============================================================
# PREDICT ONE EVENT
# ============================================================

def predict_event(
    timestamp,
    x,
    y,
    z,
    event_peak,
    fs
):

    # --------------------------------------------------------
    # Filter exactly like V1 preprocessing
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
    # Calculate acceleration magnitude
    # --------------------------------------------------------

    magnitude = calculate_magnitude(
        filtered_x,
        filtered_y,
        filtered_z
    )


    # --------------------------------------------------------
    # Extract V1 features
    # --------------------------------------------------------

    event_features = process_event(
        "LIVE",
        "UNKNOWN",
        1,
        fs,
        filtered_x,
        filtered_y,
        filtered_z,
        magnitude,
        event_peak
    )


    if event_features is None:

        return None


    # --------------------------------------------------------
    # Build feature vector in exact training order
    # --------------------------------------------------------

    values = []

    for column in feature_columns:

        value = event_features.get(
            column,
            np.nan
        )

        values.append(value)


    feature_vector = np.array(
        values,
        dtype=float
    ).reshape(1, -1)


    # --------------------------------------------------------
    # Handle invalid values
    # --------------------------------------------------------

    feature_vector = np.nan_to_num(
        feature_vector,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    # --------------------------------------------------------
    # LEVEL 1
    #
    # LOW  = ₹1 / ₹2 / ₹5
    # HIGH = ₹10 / ₹20
    # --------------------------------------------------------

    predicted_group = group_model.predict(
        feature_vector
    )[0]


    # --------------------------------------------------------
    # LEVEL 2
    # --------------------------------------------------------

    if predicted_group == "LOW":

        prediction = low_model.predict(
            feature_vector
        )[0]

    else:

        prediction = high_model.predict(
            feature_vector
        )[0]


    return prediction


# ============================================================
# MAIN LIVE LOOP
# ============================================================

try:

    while True:

        print(
            "Listening...",
            end=" ",
            flush=True
        )


        timestamp, x, y, z = read_samples(
            BUFFER_SECONDS
        )


        print(
            f"received {len(x)} samples"
        )


        # ----------------------------------------------------
        # Check enough data
        # ----------------------------------------------------

        if len(x) < 300:

            print(
                "Not enough samples. Trying again."
            )

            print()

            continue


        # ----------------------------------------------------
        # Estimate sampling rate
        # ----------------------------------------------------

        fs = estimate_sampling_rate(
            timestamp
        )


        # ----------------------------------------------------
        # Filter for impact detection
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


        # ----------------------------------------------------
        # Calculate magnitude
        # ----------------------------------------------------

        magnitude = calculate_magnitude(
            filtered_x,
            filtered_y,
            filtered_z
        )


        # ----------------------------------------------------
        # Detect impacts
        # ----------------------------------------------------

        peaks, threshold, smoothed = detect_events(
            magnitude,
            fs
        )


        if len(peaks) == 0:

            print(
                "No impact detected."
            )

            print()

            continue


        print(
            f"Detected {len(peaks)} impact(s)."
        )


        # ----------------------------------------------------
        # Predict every detected event
        # ----------------------------------------------------

        for event_number, peak in enumerate(
            peaks,
            start=1
        ):

            prediction = predict_event(
                timestamp,
                x,
                y,
                z,
                int(peak),
                fs
            )


            if prediction is None:

                print(
                    f"Event {event_number}: "
                    "could not extract window."
                )

                continue


            # =================================================
            # CONVERT MODEL LABEL
            # =================================================

            denomination = prediction.replace(
                "RUPEE_",
                "₹"
            )


            # =================================================
            # SEND TO TEAMMATE
            # =================================================

            send_prediction(
                denomination
            )


            # =================================================
            # DISPLAY RESULT LOCALLY
            # =================================================

            print()
            print(
                "########################################"
            )

            print(
                f"       DETECTED: {denomination}"
            )

            print(
                "########################################"
            )

            print()


        print()


except KeyboardInterrupt:

    print()
    print()
    print("Stopping...")


finally:

    ser.close()

    print(
        "ESP32 serial connection closed."
    )
