import serial
import numpy as np
import matplotlib.pyplot as plt
import csv
import time


# ============================================================
# CONFIG
# ============================================================

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

# Number of samples to collect
NUM_SAMPLES = 2048

# Expected ESP32 sampling frequency
SAMPLE_RATE = 200.0

# Output CSV
CSV_FILE = "vibration_data.csv"


# ============================================================
# CONNECT TO ESP32
# ============================================================

print("=" * 60)
print("LIVE VIBRATION FFT ANALYSIS")
print("=" * 60)
print()

print(f"Connecting to ESP32 on {SERIAL_PORT}...")

try:
    ser = serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=1
    )

except Exception as e:
    print()
    print("ERROR: Could not open serial port.")
    print(e)
    raise SystemExit(1)


# ESP32 resets when serial connection opens
time.sleep(2)

ser.reset_input_buffer()

print("Connected.")
print()
print("Collecting vibration data...")
print(f"Samples: {NUM_SAMPLES}")
print(f"Expected sampling rate: {SAMPLE_RATE} Hz")
print()


# ============================================================
# COLLECT DATA
# ============================================================

timestamps = []
x_values = []
y_values = []
z_values = []

while len(x_values) < NUM_SAMPLES:

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

    # ESP32 startup message
    if text == "FFT_DATA_START":
        continue

    # Ignore other ESP32 messages
    if not text[0].isdigit():
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
    x_values.append(x)
    y_values.append(y)
    z_values.append(z)

    if len(x_values) % 100 == 0:

        print(
            f"\rCollected "
            f"{len(x_values)}/{NUM_SAMPLES} samples",
            end="",
            flush=True
        )


ser.close()

print()
print()
print("Data collection complete.")


# ============================================================
# NUMPY ARRAYS
# ============================================================

timestamps = np.array(
    timestamps,
    dtype=np.int64
)

x_values = np.array(
    x_values,
    dtype=float
)

y_values = np.array(
    y_values,
    dtype=float
)

z_values = np.array(
    z_values,
    dtype=float
)


# ============================================================
# SAVE RAW DATA
# ============================================================

with open(CSV_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "timestamp_us",
        "x_g",
        "y_g",
        "z_g"
    ])

    for i in range(NUM_SAMPLES):

        writer.writerow([
            timestamps[i],
            x_values[i],
            y_values[i],
            z_values[i]
        ])

print(f"Raw data saved to: {CSV_FILE}")


# ============================================================
# CALCULATE ACTUAL SAMPLING RATE
# ============================================================

time_difference = np.diff(timestamps) / 1_000_000.0

# Use median instead of mean.
# This is more robust if a timestamp is occasionally corrupted.

median_dt = np.median(time_difference)

actual_sample_rate = 1.0 / median_dt

print()
print(
    f"Actual sampling rate: "
    f"{actual_sample_rate:.2f} Hz"
)


# ============================================================
# REMOVE DC COMPONENT
# ============================================================

# Remove the average value from each axis.
#
# The MPU6050 measures gravity and other static
# acceleration. For vibration analysis we are
# interested mainly in changes around the mean.

x_signal = x_values - np.mean(x_values)
y_signal = y_values - np.mean(y_values)
z_signal = z_values - np.mean(z_values)


# ============================================================
# APPLY HANN WINDOW
# ============================================================

window = np.hanning(NUM_SAMPLES)

x_windowed = x_signal * window
y_windowed = y_signal * window
z_windowed = z_signal * window


# ============================================================
# FFT
# ============================================================

x_fft = np.fft.rfft(
    x_windowed
)

y_fft = np.fft.rfft(
    y_windowed
)

z_fft = np.fft.rfft(
    z_windowed
)


# ============================================================
# FREQUENCY AXIS
# ============================================================

frequencies = np.fft.rfftfreq(
    NUM_SAMPLES,
    d=1.0 / actual_sample_rate
)


# ============================================================
# FFT MAGNITUDE
# ============================================================

x_magnitude = np.abs(x_fft)

y_magnitude = np.abs(y_fft)

z_magnitude = np.abs(z_fft)


# ============================================================
# FIND DOMINANT FREQUENCIES
# ============================================================

# Ignore the 0 Hz component.

x_peak_index = (
    np.argmax(x_magnitude[1:]) + 1
)

y_peak_index = (
    np.argmax(y_magnitude[1:]) + 1
)

z_peak_index = (
    np.argmax(z_magnitude[1:]) + 1
)


x_peak_frequency = frequencies[x_peak_index]
y_peak_frequency = frequencies[y_peak_index]
z_peak_frequency = frequencies[z_peak_index]


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("DOMINANT FREQUENCIES")
print("=" * 60)

print(
    f"X peak: {x_peak_frequency:.2f} Hz"
)

print(
    f"Y peak: {y_peak_frequency:.2f} Hz"
)

print(
    f"Z peak: {z_peak_frequency:.2f} Hz"
)

print()

print(
    f"FFT frequency resolution: "
    f"{actual_sample_rate / NUM_SAMPLES:.3f} Hz"
)


# ============================================================
# TIME DOMAIN PLOT
# ============================================================

time_seconds = (
    timestamps - timestamps[0]
) / 1_000_000.0


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    time_seconds,
    x_signal,
    label="X"
)

plt.plot(
    time_seconds,
    y_signal,
    label="Y"
)

plt.plot(
    time_seconds,
    z_signal,
    label="Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration change (g)")

plt.title(
    "Vibration Signal"
)

plt.legend()
plt.grid()

plt.tight_layout()


# ============================================================
# FFT PLOT
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    frequencies,
    x_magnitude,
    label="X"
)

plt.plot(
    frequencies,
    y_magnitude,
    label="Y"
)

plt.plot(
    frequencies,
    z_magnitude,
    label="Z"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("FFT Magnitude")

plt.title(
    "Vibration Frequency Spectrum"
)

# MPU6050 is sampled at approximately 200 Hz,
# so Nyquist frequency is approximately 100 Hz.

plt.xlim(
    0,
    min(100, actual_sample_rate / 2)
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("FFT ANALYSIS COMPLETE")
print("=" * 60)
