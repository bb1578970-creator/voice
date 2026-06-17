from pathlib import Path

import librosa
import numpy as np
import wfdb


def _normalize_signal(signal):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    signal = librosa.util.normalize(signal)
    return signal


def _compute_feature_vector(signal, sample_rate):
    signal = _normalize_signal(signal)

    # Trim leading/trailing silence to reduce noise impact
    signal, _ = librosa.effects.trim(signal, top_db=25)
    if signal.size == 0:
        signal = np.zeros(1, dtype=np.float32)

    zcr = librosa.feature.zero_crossing_rate(signal)
    rms = librosa.feature.rms(signal)
    spec_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)
    spec_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)

    features = np.concatenate(
        [
            [
                float(np.mean(signal)),
                float(np.std(signal)),
                float(np.max(signal)),
                float(np.min(signal)),
                float(np.mean(np.abs(signal))),
                float(np.sqrt(np.mean(signal ** 2))),
            ],
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            [float(np.mean(zcr)), float(np.std(zcr)),
             float(np.mean(rms)), float(np.std(rms)),
             float(np.mean(spec_centroid)), float(np.std(spec_centroid)),
             float(np.mean(spec_bandwidth)), float(np.std(spec_bandwidth)),
             float(np.mean(spec_rolloff)), float(np.std(spec_rolloff))],
        ]
    )
    return features.astype(float)


def extract_features_from_record(record_path):
    record_path = Path(record_path)
    record = wfdb.rdrecord(str(record_path))

    if hasattr(record, "p_signal") and record.p_signal is not None:
        signal = record.p_signal
        if signal.ndim > 1:
            signal = signal[:, 0]
        sample_rate = record.fs
    else:
        raise ValueError(f"Could not read waveform from {record_path}")

    return _compute_feature_vector(signal, sample_rate)


def extract_features_from_audio(audio_path):
    signal, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    return _compute_feature_vector(signal, sample_rate)