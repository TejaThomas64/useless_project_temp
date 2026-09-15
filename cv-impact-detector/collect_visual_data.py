"""
Visual Data Collector for CV Impact Detector
Captures frame features, logs frame-by-frame metrics to CSV, and records
manually labeled ground truth events (IMPACT / NO_IMPACT) to JSON.
"""
import os
import sys
import time
import json
import csv
import argparse
import cv2
import numpy as np

import config
from detector import ImpactDetector

CSV_HEADERS = [
    "timestamp",
    "relative_time",
    "motion_score",
    "suddenness",
    "duration_ms",
    "largest_area",
    "contour_count",
    "location_x",
    "location_y",
    "detector_state",
    "label"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect and label visual feature data")
    parser.add_argument("--source", type=str, default=None, help="Camera/video stream URL (e.g., IP webcam)")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Webcam device index")
    parser.add_argument("--output", "--csv", type=str, default="visual_features.csv",
                        help="Output CSV filename or path (default: visual_features.csv)")
    parser.add_argument("--json", type=str, default=None,
                        help="Output JSON filename or path for labeled events (optional)")
    return parser.parse_args()


def resolve_paths(csv_arg, json_arg):
    """Resolves absolute paths for CSV and JSON output files."""
    # If path contains directory separators, use as-is (converted to absolute), otherwise save in RESULTS_DIR
    if os.path.dirname(csv_arg):
        csv_path = os.path.abspath(csv_arg)
    else:
        csv_path = os.path.join(config.RESULTS_DIR, csv_arg)

    if json_arg:
        if os.path.dirname(json_arg):
            json_path = os.path.abspath(json_arg)
        else:
            json_path = os.path.join(config.RESULTS_DIR, json_arg)
    else:
        # Derive JSON filename from CSV filename
        base, _ = os.path.splitext(csv_path)
        if os.path.basename(csv_path) == "visual_features.csv":
            json_path = os.path.join(os.path.dirname(csv_path), "visual_labeled_events.json")
        else:
            json_path = f"{base}_events.json"

    return csv_path, json_path


def init_storage(csv_path, json_path):
    """Ensures results directory, CSV header, and JSON file exist."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Initialize CSV file with headers if it does not exist
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            
    # Initialize JSON file if it does not exist
    if not os.path.exists(json_path):
        with open(json_path, mode="w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def get_next_event_id(json_path):
    """Gets the next incremental event_id from the JSON file."""
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return max(item.get("event_id", 0) for item in data) + 1
        except Exception:
            pass
    return 1


def save_labeled_json_event(json_path, event_record):
    """Appends a labeled event record to the JSON file."""
    events = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                events = json.load(f)
        except Exception:
            events = []
    
    events.append(event_record)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)


def draw_hud(frame, metrics, contours, last_action_msg, start_time, csv_filename):
    """Draws ROI, motion contours, telemetry, and control instructions on frame."""
    x_min, y_min, x_max, y_max = metrics["roi_bounds"]
    state = metrics["state"]
    
    # 1. Draw ROI Box
    roi_color = (255, 255, 0)  # Cyan/Yellow
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), roi_color, 2)
    cv2.putText(frame, "ROI (Metal Plate)", (x_min + 5, y_min - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1, cv2.LINE_AA)

    # 2. Draw Motion Contours
    for c in contours:
        c_offset = c + np.array([x_min, y_min])
        cv2.drawContours(frame, [c_offset], -1, (0, 255, 255), 1)

    # 3. Target Reticle on Motion Location
    loc_x, loc_y = metrics["location_full"]
    if loc_x > 0 and loc_y > 0:
        reticle_color = (0, 0, 255) if state == "IMPACT DETECTED" else (0, 255, 0)
        cv2.circle(frame, (loc_x, loc_y), 12, reticle_color, 2)
        cv2.line(frame, (loc_x - 15, loc_y), (loc_x + 15, loc_y), reticle_color, 1)
        cv2.line(frame, (loc_x, loc_y - 15), (loc_x, loc_y + 15), reticle_color, 1)

    # 4. Telemetry Box
    bg_color = (40, 40, 40)
    cv2.rectangle(frame, (10, 10), (380, 185), bg_color, -1)
    cv2.rectangle(frame, (10, 10), (380, 185), (200, 200, 200), 1)

    rel_time = round(time.time() - start_time, 2)
    cv2.putText(frame, f"STATE: {state}", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"CSV File: {csv_filename}", (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Time: {rel_time}s | Motion: {metrics['motion_score']:.4f}", (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Suddenness: {metrics['suddenness']:.4f}", (20, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Duration: {metrics['duration_ms']} ms | Area: {metrics['largest_area']}", (20, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Contours: {len(contours)} | Loc: {metrics['location_full']}", (20, 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    
    # Last Action Feedback Line
    cv2.putText(frame, f"Last Log: {last_action_msg}", (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # 5. Instructions Banner at Frame Bottom
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 35), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.putText(frame, "KEYS: [I] = Label IMPACT | [N] = Label NO_IMPACT | [Q] = Quit", (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def main():
    args = parse_args()
    csv_path, json_path = resolve_paths(args.output, args.json)
    init_storage(csv_path, json_path)

    # Determine Video Source
    if args.source:
        print(f"[INFO] Opening stream URL: {args.source}")
        cap = cv2.VideoCapture(args.source)
    elif args.video:
        print(f"[INFO] Opening video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        print(f"[INFO] Opening webcam index: {args.camera}")
        cap = cv2.VideoCapture(args.camera)

        if config.FRAME_WIDTH > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        if config.FRAME_HEIGHT > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open camera/video source.")
        sys.exit(1)

    detector = ImpactDetector()
    start_time = time.time()
    last_action_msg = "Ready (Logging to CSV)"
    csv_filename = os.path.basename(csv_path)

    print("\n--- VISUAL DATA COLLECTOR ---")
    print(f"Features CSV : {csv_path}")
    print(f"Events JSON  : {json_path}")
    print("Press [I] to label current frame as IMPACT")
    print("Press [N] to label current frame as NO_IMPACT")
    print("Press [Q] to quit\n")

    # Open CSV file handle in append mode for low overhead
    with open(csv_path, mode="a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)

        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[WARNING] End of video stream.")
                    break

            now_ts = time.time()
            rel_time = round(now_ts - start_time, 3)

            # Process frame through ImpactDetector
            metrics, roi_thresh, contours = detector.process_frame(frame)

            # Check key press first so manual label applies immediately to current frame
            key = cv2.waitKey(1) & 0xFF
            current_label = "UNLABELED"

            if key in [ord('i'), ord('I')]:
                current_label = "IMPACT"
                last_action_msg = f"LABELED: IMPACT at {rel_time}s"
                print(f"[EVENT LOGGED] IMPACT | relative_time={rel_time}s | motion_score={metrics['motion_score']}")

                # Save JSON record
                event_record = {
                    "event_id": get_next_event_id(json_path),
                    "label": "IMPACT",
                    "timestamp": round(now_ts, 3),
                    "relative_time": rel_time,
                    "motion_score": metrics["motion_score"],
                    "suddenness": metrics["suddenness"],
                    "duration_ms": metrics["duration_ms"],
                    "largest_area": metrics["largest_area"],
                    "contour_count": len(contours),
                    "location": list(metrics["location_full"]),
                    "detector_state": metrics["state"]
                }
                save_labeled_json_event(json_path, event_record)

            elif key in [ord('n'), ord('N')]:
                current_label = "NO_IMPACT"
                last_action_msg = f"LABELED: NO_IMPACT at {rel_time}s"
                print(f"[EVENT LOGGED] NO_IMPACT | relative_time={rel_time}s | motion_score={metrics['motion_score']}")

                # Save JSON record
                event_record = {
                    "event_id": get_next_event_id(json_path),
                    "label": "NO_IMPACT",
                    "timestamp": round(now_ts, 3),
                    "relative_time": rel_time,
                    "motion_score": metrics["motion_score"],
                    "suddenness": metrics["suddenness"],
                    "duration_ms": metrics["duration_ms"],
                    "largest_area": metrics["largest_area"],
                    "contour_count": len(contours),
                    "location": list(metrics["location_full"]),
                    "detector_state": metrics["state"]
                }
                save_labeled_json_event(json_path, event_record)

            elif key in [ord('q'), ord('Q')]:
                print("[INFO] Exiting visual data collection.")
                break

            # Append numerical features for current frame to CSV
            row = [
                round(now_ts, 3),
                rel_time,
                metrics["motion_score"],
                metrics["suddenness"],
                metrics["duration_ms"],
                metrics["largest_area"],
                len(contours),
                metrics["location_full"][0],
                metrics["location_full"][1],
                metrics["state"],
                current_label
            ]
            csv_writer.writerow(row)
            csv_file.flush()

            # Render HUD and show windows
            vis_frame = draw_hud(frame.copy(), metrics, contours, last_action_msg, start_time, csv_filename)
            cv2.imshow("CV Visual Data Collector", vis_frame)
            cv2.imshow("ROI Differencing Mask", roi_thresh)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
