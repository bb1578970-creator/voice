from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extraction import extract_features_from_record

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR.parent / "dataset"
DATASET_DIR = DATASET_ROOT / "voice-icar-federico-ii-database-1.0.0"
MODEL_PATH = BASE_DIR / "voice_model.pkl"

X = []
y = []


def parse_label(text):
    text = text.lower()
    if "diagnosis" in text and "healthy" in text:
        return 0
    return 1


for info_path in sorted(DATASET_DIR.glob("*-info.txt")):
    record_name = info_path.name.replace("-info.txt", "")
    record_path = DATASET_DIR / record_name

    try:
        text = info_path.read_text(encoding="utf-8", errors="ignore")
        features = extract_features_from_record(record_path)

        X.append(features)
        y.append(parse_label(text))

    except Exception as e:
        print(f"Skipping {info_path.name}: {e}")

if len(X) == 0:
    raise RuntimeError(f"No valid records found in {DATASET_DIR}")

X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=int)

print(f"Samples loaded: {len(X)}")
print(f"Class distribution: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, pred, target_names=["Healthy", "Pathological"]))

joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")