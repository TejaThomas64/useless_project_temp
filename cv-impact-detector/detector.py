"""
Impact Detector Core Processing Module
Handles ROI cropping, 3-frame differencing, noise filtering, contour detection,
feature extraction, and temporal impact decision logic.
"""
import time
import json
import os
import cv2
import numpy as np
import config

class ImpactDetector:
    def __init__(self):
        # Frame buffers for 3-frame differencing
        self.prev_frame1 = None  # t - 1 frame (gray, blurred ROI)
        self.prev_frame2 = None  # t - 2 frame (gray, blurred ROI)
        
        # Temporal tracking state
        self.motion_start_time = None
        self.last_impact_time = 0
        self.prev_motion_score = 0.0
        
        # State indicators
        self.current_state = "NO IMPACT"
        
    def get_roi_bounds(self, frame_shape):
        """
        Calculates pixel coordinates for ROI based on config ratios.
        Returns (x_min, y_min, x_max, y_max) in pixel units.
        """
        height, width = frame_shape[:2]
        x_min = int(config.ROI_X_MIN * width)
        y_min = int(config.ROI_Y_MIN * height)
        x_max = int(config.ROI_X_MAX * width)
        y_max = int(config.ROI_Y_MAX * height)
        
        # Ensure ROI has valid positive dimensions
        x_min = max(0, min(x_min, width - 10))
        y_min = max(0, min(y_min, height - 10))
        x_max = max(x_min + 10, min(x_max, width))
        y_max = max(y_min + 10, min(y_max, height))
        
        return x_min, y_min, x_max, y_max

    def process_frame(self, frame):
        """
        Processes a single input frame.
        Returns a tuple: (metrics, visualized_roi_mask, list_of_contours)
        """
        current_time_ms = int(time.time() * 1000)
        h, w = frame.shape[:2]
        x_min, y_min, x_max, y_max = self.get_roi_bounds((h, w))
        
        # 1. Extract Fixed ROI containing the metal plate
        roi = frame[y_min:y_max, x_min:x_max]
        roi_height, roi_width = roi.shape[:2]
        roi_area = float(roi_height * roi_width)
        
        # 2. Grayscale conversion
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 3. Gaussian Blur noise reduction
        blurred = cv2.GaussianBlur(gray, config.BLUR_KERNEL_SIZE, 0)
        
        # Initialize frame buffer for the first few frames
        if self.prev_frame1 is None:
            self.prev_frame1 = blurred
            return self._default_result(current_time_ms), np.zeros_like(gray), []
        if self.prev_frame2 is None:
            self.prev_frame2 = self.prev_frame1
            self.prev_frame1 = blurred
            return self._default_result(current_time_ms), np.zeros_like(gray), []

        # 4. 3-Frame Differencing (Reduces ghosting and highlights sudden motion)
        diff1 = cv2.absdiff(blurred, self.prev_frame1)
        diff2 = cv2.absdiff(self.prev_frame1, self.prev_frame2)
        diff_combined = cv2.bitwise_and(diff1, diff2)
        
        # 5. Thresholding
        _, thresh = cv2.threshold(diff_combined, config.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # 6. Morphological Noise Removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_KERNEL_SIZE)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)

        # 7. Contour Detection
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        valid_contours = [c for c in contours if cv2.contourArea(c) >= config.MIN_CONTOUR_AREA]
        
        # 8. Feature Extraction
        total_motion_pixels = sum(cv2.contourArea(c) for c in valid_contours)
        motion_score = round(total_motion_pixels / roi_area, 4)
        
        largest_area = 0
        impact_location_roi = (0, 0)
        impact_location_full = (0, 0)
        
        if valid_contours:
            largest_c = max(valid_contours, key=cv2.contourArea)
            largest_area = int(cv2.contourArea(largest_c))
            M = cv2.moments(largest_c)
            if M["m00"] != 0:
                cx_roi = int(M["m10"] / M["m00"])
                cy_roi = int(M["m01"] / M["m00"])
                impact_location_roi = (cx_roi, cy_roi)
                impact_location_full = (x_min + cx_roi, y_min + cy_roi)
        
        # Suddenness: rate of change of motion score
        suddenness = round(motion_score - self.prev_motion_score, 4)
        
        # Track motion duration
        if motion_score >= config.MOTION_DETECTED_THRESHOLD:
            if self.motion_start_time is None:
                self.motion_start_time = current_time_ms
            duration_ms = current_time_ms - self.motion_start_time
        else:
            self.motion_start_time = None
            duration_ms = 0

        # 9. Impact Decision Logic
        state, confidence, event_triggered = self._evaluate_impact(
            motion_score, suddenness, duration_ms, largest_area, current_time_ms
        )
        
        self.current_state = state
        self.prev_motion_score = motion_score
        
        # Update frame buffer
        self.prev_frame2 = self.prev_frame1
        self.prev_frame1 = blurred
        
        metrics = {
            "timestamp": int(time.time()),
            "state": state,
            "motion_score": motion_score,
            "largest_area": largest_area,
            "suddenness": suddenness,
            "duration_ms": duration_ms,
            "confidence": confidence,
            "location_roi": impact_location_roi,
            "location_full": impact_location_full,
            "event_triggered": event_triggered,
            "roi_bounds": (x_min, y_min, x_max, y_max)
        }
        
        # 10. Save JSON output if an event triggered
        if event_triggered and config.SAVE_EVENTS_JSON:
            self._save_event(metrics)
            
        return metrics, cleaned, valid_contours

    def _evaluate_impact(self, motion_score, suddenness, duration_ms, largest_area, current_time_ms):
        """
        Decision engine that distinguishes sharp physical impacts from continuous background motion.
        """
        in_cooldown = (current_time_ms - self.last_impact_time) < config.COOLDOWN_MS
        
        # Baseline check: No significant motion
        if motion_score < config.MOTION_DETECTED_THRESHOLD:
            return "NO IMPACT", 0.0, False
        
        # Continuous motion past max impact duration is ongoing movement/walking/hands
        if duration_ms > config.MAX_IMPACT_DURATION_MS:
            return "MOTION DETECTED", 0.25, False

        # Calculate experimental confidence score (0.0 to 1.0)
        # Based on motion magnitude, suddenness spike, and spatial concentration
        conf_motion = min(1.0, motion_score / (config.IMPACT_MOTION_SCORE_MIN * 2))
        conf_sudden = min(1.0, suddenness / config.SUDDENNESS_THRESHOLD) if suddenness > 0 else 0.0
        confidence = round(float(0.5 * conf_motion + 0.5 * conf_sudden), 2)

        # Impact criteria: Sharp sudden spike within short duration
        is_sudden = suddenness >= config.SUDDENNESS_THRESHOLD
        has_area = motion_score >= config.IMPACT_MOTION_SCORE_MIN
        
        if has_area and is_sudden and not in_cooldown:
            self.last_impact_time = current_time_ms
            return "IMPACT DETECTED", max(0.60, confidence), True
        elif has_area or is_sudden:
            return "IMPACT CANDIDATE", confidence, False
        else:
            return "MOTION DETECTED", confidence, False

    def _save_event(self, metrics):
        """Appends confirmed impact event to JSON file."""
        event_data = {
            "event": "IMPACT_DETECTED",
            "timestamp": metrics["timestamp"],
            "confidence": metrics["confidence"],
            "impact_location": list(metrics["location_full"]),
            "motion_score": metrics["motion_score"],
            "duration_ms": metrics["duration_ms"],
            "largest_contour_area": metrics["largest_area"]
        }
        
        events = []
        if os.path.exists(config.EVENTS_FILE_PATH):
            try:
                with open(config.EVENTS_FILE_PATH, "r") as f:
                    events = json.load(f)
            except Exception:
                events = []
                
        events.append(event_data)
        
        with open(config.EVENTS_FILE_PATH, "w") as f:
            json.dump(events, f, indent=2)

    def _default_result(self, current_time_ms):
        return {
            "timestamp": int(time.time()),
            "state": "NO IMPACT",
            "motion_score": 0.0,
            "largest_area": 0,
            "suddenness": 0.0,
            "duration_ms": 0,
            "confidence": 0.0,
            "location_roi": (0, 0),
            "location_full": (0, 0),
            "event_triggered": False,
            "roi_bounds": (0, 0, 0, 0)
        }
