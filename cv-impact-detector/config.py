"""
Centralized Configuration for CV Impact Detector
Adjust these parameters to calibrate the system for your camera layout, lighting, and plate setup.
"""
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")

# Ensure required output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# --- Camera / Input Settings ---
CAMERA_INDEX = 0             # Default webcam index
FRAME_WIDTH = 640            # Target capture width (0 for camera default)
FRAME_HEIGHT = 480           # Target capture height (0 for camera default)

# --- Region of Interest (ROI) ---
# Coordinates normalized between 0.0 and 1.0 (relative to full frame width/height)
# You can adjust these in GUI or edit here to restrict detection strictly to the metal plate.
ROI_X_MIN = 0.20  # Left boundary (20% of frame width)
ROI_Y_MIN = 0.20  # Top boundary (20% of frame height)
ROI_X_MAX = 0.80  # Right boundary (80% of frame width)
ROI_Y_MAX = 0.80  # Bottom boundary (80% of frame height)

# --- Image Preprocessing ---
BLUR_KERNEL_SIZE = (5, 5)    # Gaussian blur kernel size (must be odd tuple)
DIFF_THRESHOLD = 25          # Pixel difference threshold (0-255) for frame differencing
MORPH_KERNEL_SIZE = (3, 3)   # Morphological noise removal kernel size

# --- Detection & Filtering Thresholds ---
MIN_CONTOUR_AREA = 50        # Ignore contours smaller than this pixel count (removes sensor noise)
LARGEST_CONTOUR_MIN_AREA = 100 # Minimum area for the primary motion region

# --- Impact Decision Rules ---
# Motion score is calculated as (changed_pixels / total_roi_pixels)
MOTION_DETECTED_THRESHOLD = 0.01  # Motion score > 1% flags MOTION DETECTED
IMPACT_MOTION_SCORE_MIN = 0.04    # Motion score > 4% required for impact candidate

# Temporal metrics (Spike vs Continuous movement)
# Impacts are sudden spikes that decay quickly.
SUDDENNESS_THRESHOLD = 0.03       # Rapid increase in motion_score between consecutive frames
MAX_IMPACT_DURATION_MS = 400      # Max duration (ms) for an event to be classed as an impact
                                  # Continuous motion > 400ms is classified as ongoing movement/walking

COOLDOWN_MS = 1000                # Time (ms) to wait before recording another impact event

# --- Logging & Results ---
SAVE_EVENTS_JSON = True
EVENTS_FILE_PATH = os.path.join(RESULTS_DIR, "impact_events.json")
