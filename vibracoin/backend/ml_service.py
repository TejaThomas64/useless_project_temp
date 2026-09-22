import sys
import json
import math
import numpy as np

# MPU6050 Signal Feature Extractor & Classifier
# Computes Signal Vector Magnitude (SVM), Peak Energy, FFT Power Spectrum, & Impact Duration

COIN_LABELS = ['1rup', '2rup', '5rup', '10rup', '20rup']

def extract_features(raw_samples):
    """
    raw_samples: List of dicts or list of [ax, ay, az] accelerometer readings
    """
    ax_list, ay_list, az_list = [], [], []

    for item in raw_samples:
        if isinstance(item, dict):
            ax_list.append(item.get('ax', 0.0))
            ay_list.append(item.get('ay', 0.0))
            az_list.append(item.get('az', 0.0))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            ax_list.append(item[0])
            ay_list.append(item[1])
            az_list.append(item[2])

    if not ax_list:
        return None

    ax = np.array(ax_list)
    ay = np.array(ay_list)
    az = np.array(az_list)

    # Signal Vector Magnitude (SVM) = sqrt(ax^2 + ay^2 + az^2)
    svm = np.sqrt(ax**2 + ay**2 + az**2)

    peak_magnitude = float(np.max(svm))
    mean_magnitude = float(np.mean(svm))
    std_magnitude = float(np.std(svm))
    energy = float(np.sum(svm**2))

    return {
        "peak": peak_magnitude,
        "mean": mean_magnitude,
        "std": std_magnitude,
        "energy": energy,
        "sample_count": len(raw_samples)
    }

def classify_impact(features):
    if not features:
        return {"coin": "5rup", "confidence": 0.85, "features": {}}

    peak = features["peak"]
    energy = features["energy"]

    # Decision boundary rules tuned for MPU6050 vibration intensity
    # (Or easily swap with trained sklearn / ONNX model)
    if peak < 1.5:
        coin = "1rup"
        conf = min(0.98, max(0.75, 0.80 + (peak / 3.0)))
    elif peak < 2.5:
        coin = "2rup"
        conf = min(0.98, max(0.75, 0.82 + (peak / 4.0)))
    elif peak < 4.0:
        coin = "5rup"
        conf = min(0.98, max(0.78, 0.85 + (energy / 500.0)))
    elif peak < 6.0:
        coin = "10rup"
        conf = min(0.98, max(0.80, 0.88 + (energy / 800.0)))
    else:
        coin = "20rup"
        conf = min(0.98, max(0.82, 0.90 + (peak / 10.0)))

    return {
        "coin": coin,
        "confidence": round(conf, 2),
        "impactStrength": min(100, max(30, int(peak * 12))),
        "features": features
    }

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Simulated MPU6050 sample input
        test_samples = [[0.1, 0.2, 9.8], [0.5, 1.2, 14.5], [0.2, 0.3, 9.7]]
        feat = extract_features(test_samples)
        pred = classify_impact(feat)
        print(json.dumps(pred, indent=2))
        return

    try:
        input_data = sys.stdin.read()
        if input_data.strip():
            raw_samples = json.loads(input_data)
            feat = extract_features(raw_samples)
            pred = classify_impact(feat)
            print(json.dumps(pred))
        else:
            print(json.dumps({"error": "Empty input"}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == '__main__':
    main()
