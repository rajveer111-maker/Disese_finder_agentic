"""
FILE-WISE SPECTRA INFERENCE
===========================

Reads one .edf/.csv file or a folder of .edf/.csv files.
Converts each file to the same training input format:

    windows:   (1, N, 19, 1024)
    sfreq:     200 Hz
    channels:  19 standard EEG channels
    z-score:   per channel
    phase_id:  default Rest = 0
    device_id: default 0
    protocol_id: default 0

Prediction:
    0 -> Control
    1 -> Schizophrenic / Patient

Expected:
    model.py containing SPECTRA
    trained checkpoint best_model.pt

Manual config only. No argparse.
"""

import re
import warnings
from pathlib import Path
from fractions import Fraction

import numpy as np
import pandas as pd
import torch
import mne

from scipy.signal import resample_poly
from model import SPECTRA

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


# ============================================================
# MANUAL CONFIG
# ============================================================

CFG = {
    # Can be one .edf/.csv file OR a folder containing files
    "input_path": "/kaggle/input/datasets/nikhilkushwaha2529/aszed-153/ASZED-153/ASZED/version_1.1/node_1/subset_1/subject_12/1/Phase 1.edf",

    # Your trained model checkpoint
    "checkpoint_path": "/kaggle/input/datasets/nikhilkushwaha2529/res-sz-spectra/spectra_normal_run/best_model.pt",

    # Output CSV
    "output_csv": "filewise_predictions.csv",

    # CSV does not store sampling rate, so set it manually for CSV files
    "csv_sampling_rate": 200,

    # Search inside subfolders if input_path is a folder
    "recursive": True,

    # Same as training
    "target_sfreq": 200,
    "target_tlen": 1024,
    "stride": 512,

    # Unknown metadata defaults.
    # Used because inference should work irrespective of device/phase/session.
    "default_phase_id": 0,
    "default_device_id": 0,
    "default_protocol_id": 0,

    # Pad if file is shorter than 1024 samples after resampling
    "pad_short_file": True,

    "device": (
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    ),
}


# ============================================================
# CHANNEL CONSTANTS
# ============================================================

STANDARD_19 = [
    "Fp1", "Fp2", "F3",  "F4",  "C3",  "C4",
    "P3",  "P4",  "O1",  "O2",  "F7",  "F8",
    "T3",  "T4",  "T5",  "T6",  "Fz",  "Cz",  "Pz"
]

CH_NORM = {
    "Fp1[1]": "Fp1", "Fp2[2]": "Fp2", "F3[3]": "F3",
    "F4[4]": "F4", "C3[5]": "C3", "C4[6]": "C4",
    "P3[7]": "P3", "P4[8]": "P4", "O1[9]": "O1",
    "O2[10]": "O2", "F7[11]": "F7", "F8[12]": "F8",
    "T3[13]": "T3", "T4[14]": "T4", "T5[15]": "T5",
    "T6[16]": "T6", "Fz[17]": "Fz", "Pz[18]": "Pz",
    "Cz[19]": "Cz",

    "EEG Fp1-LE": "Fp1", "EEG Fp2-LE": "Fp2",
    "EEG F3-LE": "F3", "EEG F4-LE": "F4",
    "EEG C3-LE": "C3", "EEG C4-LE": "C4",
    "EEG P3-LE": "P3", "EEG P4-LE": "P4",
    "EEG O1-LE": "O1", "EEG O2-LE": "O2",
    "EEG F7-LE": "F7", "EEG F8-LE": "F8",
    "EEG T3-LE": "T3", "EEG T4-LE": "T4",
    "EEG T5-LE": "T5", "EEG T6-LE": "T6",
    "EEG Fz-LE": "Fz", "EEG Cz-LE": "Cz",
    "EEG Pz-LE": "Pz",
}


# ============================================================
# PREPROCESSING
# ============================================================

def normalize_channel_name(name):
    s = str(name).strip()

    if s in CH_NORM:
        return CH_NORM[s]

    s = re.sub(r"^EEG\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"-LE$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"-REF$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\[\d+\]$", "", s)

    standard_map = {ch.lower(): ch for ch in STANDARD_19}
    return standard_map.get(s.lower(), s)


def force_19_channels(data, channel_names=None):
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D data, got {data.shape}")

    C, T = data.shape

    if channel_names is not None and len(channel_names) == C:
        names = [normalize_channel_name(x) for x in channel_names]
        name_to_idx = {}

        for i, ch in enumerate(names):
            if ch not in name_to_idx:
                name_to_idx[ch] = i

        out = np.zeros((19, T), dtype=np.float32)
        found = 0

        for out_i, ch in enumerate(STANDARD_19):
            if ch in name_to_idx:
                out[out_i] = data[name_to_idx[ch]]
                found += 1

        # same practical condition used in training: enough standard channels
        if found >= 15:
            return out

    # fallback: first 19 channels
    if C >= 19:
        return data[:19].astype(np.float32)

    # pad missing channels with zeros
    out = np.zeros((19, T), dtype=np.float32)
    out[:C] = data
    return out


def resample_to_200(data, original_sfreq):
    target = float(CFG["target_sfreq"])
    original = float(original_sfreq)

    if abs(original - target) < 1e-6:
        return data.astype(np.float32)

    frac = Fraction(target / original).limit_denominator(1000)
    return resample_poly(data, up=frac.numerator, down=frac.denominator, axis=1).astype(np.float32)


def z_score(data, eps=1e-8):
    mu = data.mean(axis=1, keepdims=True)
    sd = data.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((data - mu) / sd).astype(np.float32)


def make_windows(data):
    window_size = int(CFG["target_tlen"])
    stride = int(CFG["stride"])

    C, T = data.shape

    if C != 19:
        raise ValueError(f"Expected 19 channels, got {C}")

    if T < window_size:
        if not CFG["pad_short_file"]:
            raise ValueError(f"File too short: {T} samples, required {window_size}")

        pad = np.zeros((19, window_size - T), dtype=np.float32)
        data = np.concatenate([data, pad], axis=1)
        return data[None, :, :].astype(np.float32)

    windows = []
    start = 0

    while start + window_size <= data.shape[1]:
        windows.append(data[:, start:start + window_size])
        start += stride

    return np.stack(windows, axis=0).astype(np.float32)


# ============================================================
# FILE LOADERS
# ============================================================

def load_edf_file(path):
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    data = raw.get_data().astype(np.float32)
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)
    return data, sfreq, ch_names


def load_csv_file(path):
    # try normal header CSV
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number]).copy()

    # remove likely time columns
    drop_cols = []
    for col in numeric.columns:
        c = str(col).lower()
        if c in ["time", "timestamp", "timestamps", "t", "index", "sample"]:
            drop_cols.append(col)

    if drop_cols:
        numeric = numeric.drop(columns=drop_cols)

    # fallback for no-header CSV
    if numeric.shape[1] < 2:
        df = pd.read_csv(path, header=None)
        numeric = df.select_dtypes(include=[np.number]).copy()

    arr = numeric.to_numpy(dtype=np.float32)

    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("CSV has no valid numeric signal data")

    rows, cols = arr.shape

    # Decide orientation
    if rows > cols:
        # time x channels
        data = arr.T
        ch_names = list(numeric.columns)
    else:
        # channels x time
        data = arr
        ch_names = None

    return data.astype(np.float32), float(CFG["csv_sampling_rate"]), ch_names


def file_to_batch(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".edf":
        data, sfreq, ch_names = load_edf_file(path)
    elif suffix == ".csv":
        data, sfreq, ch_names = load_csv_file(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    data = force_19_channels(data, ch_names)
    data = resample_to_200(data, sfreq)
    data = z_score(data)
    windows = make_windows(data)

    n = windows.shape[0]

    batch = {
        "windows": torch.tensor(windows, dtype=torch.float32).unsqueeze(0),
        "phase_ids": torch.full((1, n), int(CFG["default_phase_id"]), dtype=torch.long),
        "pad_mask": torch.zeros((1, n), dtype=torch.bool),
        "device_id": torch.tensor([int(CFG["default_device_id"])], dtype=torch.long),
        "protocol_id": torch.tensor([int(CFG["default_protocol_id"])], dtype=torch.long),
    }

    meta = {
        "original_sfreq": float(sfreq),
        "target_sfreq": float(CFG["target_sfreq"]),
        "num_windows": int(n),
        "samples_after_resample": int(data.shape[1]),
        "file_type": suffix.replace(".", ""),
    }

    return batch, meta


# ============================================================
# MODEL + PREDICTION
# ============================================================

def load_model(checkpoint_path, device):
    model = SPECTRA().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_one_file(model, path, device):
    batch, meta = file_to_batch(path)

    out = model(
        windows=batch["windows"].to(device),
        phase_ids=batch["phase_ids"].to(device),
        pad_mask=batch["pad_mask"].to(device),
        device_id=batch["device_id"].to(device),
        protocol_id=batch["protocol_id"].to(device),
    )

    probs = torch.softmax(out["logits"], dim=-1)

    prob_control = float(probs[0, 0].detach().cpu().item())
    prob_schizophrenic = float(probs[0, 1].detach().cpu().item())
    pred = int(torch.argmax(probs, dim=-1).item())

    result = {
        "file_path": str(path),
        "prediction": pred,
        "prediction_label": "Schizophrenic" if pred == 1 else "Control",
        "prob_control": prob_control,
        "prob_schizophrenic": prob_schizophrenic,
    }

    result.update(meta)
    return result


def collect_files(input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    if CFG["recursive"]:
        files = list(input_path.rglob("*.edf")) + list(input_path.rglob("*.csv"))
    else:
        files = list(input_path.glob("*.edf")) + list(input_path.glob("*.csv"))

    return sorted(files)


def main():
    device = torch.device(CFG["device"])
    print(f"Using device: {device}")

    model = load_model(CFG["checkpoint_path"], device)

    files = collect_files(CFG["input_path"])

    if len(files) == 0:
        raise RuntimeError("No .edf or .csv files found.")

    print(f"Found files: {len(files)}")

    results = []

    for i, file_path in enumerate(files, 1):
        try:
            result = predict_one_file(model, file_path, device)
            results.append(result)

            print(
                f"[{i}/{len(files)}] OK | {file_path.name} -> "
                f"{result['prediction_label']} | "
                f"Control={result['prob_control']:.4f}, "
                f"Schizophrenic={result['prob_schizophrenic']:.4f}"
            )

        except Exception as e:
            print(f"[{i}/{len(files)}] FAILED | {file_path} | {e}")
            results.append({
                "file_path": str(file_path),
                "prediction": None,
                "prediction_label": "FAILED",
                "prob_control": None,
                "prob_schizophrenic": None,
                "error": str(e),
            })

    df = pd.DataFrame(results)
    df.to_csv(CFG["output_csv"], index=False)

    print("\nDONE")
    print(df)
    print(f"\nSaved predictions to: {CFG['output_csv']}")


if __name__ == "__main__":
    main()
