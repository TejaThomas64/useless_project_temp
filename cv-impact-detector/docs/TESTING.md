# CV Impact Detector — Manual Test Suite (`TESTING.md`)

This document outlines the systematic manual testing procedure for calibrating and evaluating the OpenCV-based impact detector.

Record your observations directly into the table provided below.

---

## 1. Test Setup Instructions

1. **Mounting**: Position your webcam on a stable tripod or stand pointing directly at the metal plate. Ensure the camera itself does not vibrate when the table is tapped.
2. **ROI Calibration**: Adjust `ROI_X_MIN`, `ROI_Y_MIN`, `ROI_X_MAX`, `ROI_Y_MAX` in [`config.py`](file:///c:/Users/Theja/Desktop/vibracoin/cv-impact-detector/config.py) so the Cyan box covers ONLY the surface of the metal plate.
3. **Lighting**: Use constant room illumination; avoid flickering lights or direct sunlight shadows across the plate.

---

## 2. Test Execution Matrix

Perform each test scenario 3 times and note down the observed behavior, detected state, and motion score.

| Test # | Test Scenario | Action Description | Expected Result | Pass / Fail | Observed State & Motion Score | Notes / Calibration Adjustments |
|---|---|---|---|---|---|---|
| **1** | **Stationary Plate** | Plate completely undisturbed for 30 seconds. | `NO IMPACT`<br>(`motion_score` < 0.01) | | | Should show zero contours in mask window. |
| **2** | **Background Movement** | Person walking or waving hands in background *outside* ROI box. | `NO IMPACT`<br>(Ignored by ROI) | | | If triggered, tighten ROI boundaries in `config.py`. |
| **3** | **Typing Nearby** | Type heavily on a keyboard on the same table. | `NO IMPACT` or `MOTION DETECTED` | | | Small table vibration shouldn't trigger `IMPACT DETECTED`. |
| **4** | **Walking Nearby** | Heavy footsteps near the plate table. | `NO IMPACT` | | | Tests camera mount rigidity and shadow immunity. |
| **5** | **Touching Plate** | Gently rest hand or touch plate for 2 seconds. | `MOTION DETECTED`<br>(Duration > 400ms) | | | Must NOT trigger impact event due to long duration. |
| **6** | **Placing Object** | Gently place a coin or coin-stack on the plate. | `MOTION DETECTED` or `IMPACT CANDIDATE` | | | Motion score moderate, suddenness low. |
| **7** | **Dropping Coin / Object** | Drop a coin onto the metal plate from 5–10 cm height. | **`IMPACT DETECTED`**<br>(Confidence > 0.60, Duration < 350ms) | | | **Primary Target Event**: Instant sharp motion spike. |
| **8** | **Accidental Disturbance** | Light tap on table edge or light plate nudge. | `IMPACT CANDIDATE` or `NO IMPACT` | | | Adjust `IMPACT_MOTION_SCORE_MIN` if false positives occur. |

---

## 3. How to Record & Save Test Results

1. **JSON Logs**: Every time `IMPACT DETECTED` triggers, an entry is automatically saved to [`data/results/impact_events.json`](file:///c:/Users/Theja/Desktop/vibracoin/cv-impact-detector/data/results/impact_events.json).
2. **Snapshots**: Press `s` while the program is running to save a annotated snapshot image into [`data/results/`](file:///c:/Users/Theja/Desktop/vibracoin/cv-impact-detector/data/results/).
3. **Recording Test Videos**: You can record short test clips and place them in [`data/videos/`](file:///c:/Users/Theja/Desktop/vibracoin/cv-impact-detector/data/videos/) to run reproducible test runs via:
   ```powershell
   python main.py --video data/videos/coin_drop.mp4
   ```

---

## 4. Troubleshooting & Calibration Rules

- **Too Many False Positives (Walking / Background triggering impacts)**:
  - Increase `ROI_X_MIN`/`ROI_Y_MIN` to make ROI tighter around metal plate.
  - Increase `IMPACT_MOTION_SCORE_MIN` (e.g., from `0.04` to `0.07`).
  - Increase `SUDDENNESS_THRESHOLD` (e.g., from `0.03` to `0.05`).

- **Missed Real Impacts (Dropping coin does not trigger `IMPACT DETECTED`)**:
  - Decrease `DIFF_THRESHOLD` (e.g., from `25` to `18`).
  - Decrease `IMPACT_MOTION_SCORE_MIN` (e.g., from `0.04` to `0.02`).
  - Decrease `SUDDENNESS_THRESHOLD` (e.g., from `0.03` to `0.015`).
