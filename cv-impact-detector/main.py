"""
Main Execution Script for CV Impact Detector
Handles video stream capture (webcam or video file), frame processing loop,
real-time HUD display, and terminal log output.
"""
import sys
import os
import argparse
import time
import cv2
import config
from detector import ImpactDetector

def parse_args():
    parser = argparse.ArgumentParser(description="Computer Vision Impact Detector for Metal Plate")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file (default: webcam index 0)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Webcam device index (default: 0)")
    parser.add_argument("--source", type=str, default=None, help="Camera/video stream URL")
    return parser.parse_args()

def draw_hud(frame, metrics, contours):
    """Overlays ROI box, detected contours, impact reticle, and telemetry HUD on frame."""
    x_min, y_min, x_max, y_max = metrics["roi_bounds"]
    state = metrics["state"]
    
    # 1. Draw ROI Box
    roi_color = (255, 255, 0) # Cyan/Yellow for ROI boundary
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), roi_color, 2)
    cv2.putText(frame, "ROI (Metal Plate)", (x_min + 5, y_min - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1, cv2.LINE_AA)
                
    # 2. Draw contours offset to full frame coordinates
    for c in contours:
        c_offset = c + np.array([x_min, y_min])
        cv2.drawContours(frame, [c_offset], -1, (0, 255, 255), 1)

    # 3. Draw Impact Reticle if motion/impact detected
    if state in ["IMPACT CANDIDATE", "IMPACT DETECTED"]:
        loc_x, loc_y = metrics["location_full"]
        if loc_x > 0 and loc_y > 0:
            target_color = (0, 0, 255) if state == "IMPACT DETECTED" else (0, 165, 255)
            cv2.circle(frame, (loc_x, loc_y), 15, target_color, 2)
            cv2.line(frame, (loc_x - 20, loc_y), (loc_x + 20, loc_y), target_color, 2)
            cv2.line(frame, (loc_x, loc_y - 20), (loc_x, loc_y + 20), target_color, 2)

    # 4. State HUD Colors
    if state == "NO IMPACT":
        bg_color = (0, 150, 0)       # Green
    elif state == "MOTION DETECTED":
        bg_color = (0, 140, 255)     # Orange
    elif state == "IMPACT CANDIDATE":
        bg_color = (0, 200, 255)     # Yellow-Orange
    else:
        bg_color = (0, 0, 220)       # Red

    # Overlay Telemetry Panel (Top Left)
    cv2.rectangle(frame, (10, 10), (320, 140), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (320, 140), bg_color, 2)
    
    cv2.putText(frame, f"STATE: {state}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Motion Score: {metrics['motion_score']:.4f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Confidence: {metrics['confidence']:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Duration: {metrics['duration_ms']} ms", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Location: {metrics['location_full']}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                
    return frame

import numpy as np # Import numpy inside main.py as well for array offsets

def main():
    args = parse_args()
    
    # Select Video Source (File or Camera Index)
    if args.source:
        print(f"[INFO] Opening stream: {args.source}")
        cap = cv2.VideoCapture(args.source)
    elif args.video:
        print(f"[INFO] Opening video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        print(f"[INFO] Opening camera index: {args.camera}")
        cap = cv2.VideoCapture(args.camera)
        
        # Configure Camera Resolution if requested
        if config.FRAME_WIDTH > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        if config.FRAME_HEIGHT > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source.")
        sys.exit(1)

    detector = ImpactDetector()
    last_printed_state = None
    
    print("\n--- CV IMPACT DETECTOR RUNNING ---")
    print("Press 'q' to quit | Press 'r' to reset detector baseline | Press 's' to save snapshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            # If video file ended, loop back to start or break
            if args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[WARNING] Failed to grab frame from camera.")
                break

        # Run Detector Core
        metrics, roi_thresh, contours = detector.process_frame(frame)
        current_state = metrics["state"]

        # Print formatted log output to console on state transition or impact trigger
        if current_state != last_printed_state or metrics["event_triggered"]:
            print(f"{current_state}")
            if current_state != "NO IMPACT":
                if current_state in ["IMPACT CANDIDATE", "IMPACT DETECTED"]:
                    print(f"confidence: {metrics['confidence']}")
                    print(f"location: {metrics['location_full']}")
                print(f"motion_score: {metrics['motion_score']}")
                print(f"suddenness: {metrics['suddenness']}")
                print(f"duration: {metrics['duration_ms']} ms")
            print() # Blank line
            last_printed_state = current_state

        # Draw visual telemetry HUD
        vis_frame = draw_hud(frame.copy(), metrics, contours)

        # Show GUI Windows
        cv2.imshow("CV Impact Detector Feed", vis_frame)
        cv2.imshow("ROI Differencing Mask", roi_thresh)

        # Key Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quitting application.")
            break
        elif key == ord('r'):
            print("[INFO] Resetting detector baseline.")
            detector = ImpactDetector()
        elif key == ord('s'):
            snap_path = os.path.join(config.RESULTS_DIR, f"snapshot_{int(time.time())}.jpg")
            cv2.imwrite(snap_path, vis_frame)
            print(f"[INFO] Saved snapshot to {snap_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
