from pathlib import Path

import joblib
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from feature_extraction import extract_features_from_audio

app = Flask(__name__, static_folder=str((Path(__file__).resolve().parent.parent / "frontend")), template_folder=str((Path(__file__).resolve().parent.parent / "frontend")))
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MODEL_PATH = BASE_DIR / "voice_model.pkl"
TEMP_PATH = BASE_DIR / "temp.webm"

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return send_file(FRONTEND_DIR / "index.html")


@app.route("/index.html")
def index_html():
    return send_file(FRONTEND_DIR / "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"})

    audio_file = request.files["audio"]
    audio_file.save(TEMP_PATH)

    try:
        features = extract_features_from_audio(TEMP_PATH)
        prediction = model.predict([features])[0]
        result = "Healthy" if prediction == 0 else "Pathological"
        return jsonify({"prediction": result})
    except Exception as exc:
        return jsonify({"error": str(exc)})
    finally:
        if TEMP_PATH.exists():
            TEMP_PATH.unlink()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)