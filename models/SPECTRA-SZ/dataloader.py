"""
ASZED Dataset Loader
=====================
Handles cross-device, cross-protocol harmonization:
- Subset 1: CONTEK-KT2400 @ 200Hz, 24ch (19 EEG + extras), Protocol 1.1
- Subset 2: DISCOVERY-24E @ 256Hz, 20ch, Protocol 1.2 (Arithmetic annotated)
- Subset 3: DISCOVERY-24E @ 256Hz, 20ch, Protocol 1.2 (no Arithmetic annotation)

Phase label mapping (from keymaps.kmp):
  Protocol 1.1: Phase1=Rest, Phase2=Arithmetic, Phase3=Rest, Phase4=Oddball
  Protocol 1.2: Phase1=Rest, Phase2=ASSR, Phase3=Arithmetic, Phase4=Rest, Phase5=Oddball
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.signal import resample_poly
from torch.utils.data import Dataset, DataLoader
import torch

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────

DATASET_ROOT = Path('/kaggle/input/datasets/nikhilkushwaha2529/aszed-153/ASZED-153')
SPREADSHEET  = DATASET_ROOT / "ASZED_SpreadSheet.csv"
EEG_ROOT     = DATASET_ROOT / "ASZED" / "version_1.1" / "node_1"

# The 19 standard 10-20 EEG channels shared across both devices
STANDARD_19 = [
    "Fp1", "Fp2", "F3",  "F4",  "C3",  "C4",
    "P3",  "P4",  "O1",  "O2",  "F7",  "F8",
    "T3",  "T4",  "T5",  "T6",  "Fz",  "Cz",  "Pz"
]

# Channel name normalization map: raw EDF name → standard name
CH_NORM = {
    # Subset 1 (CONTEK) — has index suffixes
    "Fp1[1]":  "Fp1",  "Fp2[2]":  "Fp2",  "F3[3]":   "F3",
    "F4[4]":   "F4",   "C3[5]":   "C3",   "C4[6]":   "C4",
    "P3[7]":   "P3",   "P4[8]":   "P4",   "O1[9]":   "O1",
    "O2[10]":  "O2",   "F7[11]":  "F7",   "F8[12]":  "F8",
    "T3[13]":  "T3",   "T4[14]":  "T4",   "T5[15]":  "T5",
    "T6[16]":  "T6",   "Fz[17]":  "Fz",   "Pz[18]":  "Pz",
    "Cz[19]":  "Cz",
    # Subset 2/3 (DISCOVERY) — use "EEG X-LE" format
    "EEG Fp1-LE": "Fp1",  "EEG Fp2-LE": "Fp2",
    "EEG F3-LE":  "F3",   "EEG F4-LE":  "F4",
    "EEG C3-LE":  "C3",   "EEG C4-LE":  "C4",
    "EEG P3-LE":  "P3",   "EEG P4-LE":  "P4",
    "EEG O1-LE":  "O1",   "EEG O2-LE":  "O2",
    "EEG F7-LE":  "F7",   "EEG F8-LE":  "F8",
    "EEG T3-LE":  "T3",   "EEG T4-LE":  "T4",
    "EEG T5-LE":  "T5",   "EEG T6-LE":  "T6",
    "EEG Fz-LE":  "Fz",   "EEG Cz-LE":  "Cz",
    "EEG Pz-LE":  "Pz",
}

# Phase-type string labels (for PAFR routing)
PHASE_TYPES = ["Rest", "Arithmetic", "ASSR", "Oddball"]
PHASE2IDX   = {p: i for i, p in enumerate(PHASE_TYPES)}

# Protocol → ordered phase label list
PROTOCOL_PHASES = {
    "1.1": ["Rest", "Arithmetic", "Rest", "Oddball"],           # 4 phases
    "1.2": ["Rest", "ASSR",       "Arithmetic", "Rest", "Oddball"],  # 5 phases
}

# Device → ID
DEVICE2IDX   = {"CONTEK-KT2400": 0, "DISCOVERY-24E": 1}
PROTOCOL2IDX = {"1.1": 0, "1.2": 1}
TARGET_SFREQ = 200   # Hz — resample everything to this
TARGET_TLEN  = 1024  # samples at 200Hz → 5.12 seconds per window

# 10-20 adjacency matrix for 19 standard channels (used in graph loss)
# fmt: off
CH_IDX = {ch: i for i, ch in enumerate(STANDARD_19)}
ADJACENCY_PAIRS = [
    ("Fp1","F3"),("Fp1","F7"),("Fp1","Fp2"),
    ("Fp2","F4"),("Fp2","F8"),
    ("F7","F3"),("F7","T3"),
    ("F3","Fz"),("F3","C3"),
    ("Fz","F4"),("Fz","Cz"),
    ("F4","F8"),("F4","C4"),
    ("F8","T4"),
    ("T3","C3"),("T3","T5"),
    ("C3","Cz"),("C3","P3"),
    ("Cz","C4"),("Cz","Pz"),
    ("C4","T4"),("C4","P4"),
    ("T4","T6"),
    ("T5","P3"),("T5","O1"),
    ("P3","Pz"),
    ("Pz","P4"),
    ("P4","T6"),("P4","O2"),
    ("T6","O2"),
    ("O1","O2"),
]

def build_adjacency_matrix() -> torch.Tensor:
    """Returns a (19, 19) symmetric adjacency tensor for the 10-20 system."""
    A = torch.zeros(19, 19)
    for a, b in ADJACENCY_PAIRS:
        i, j = CH_IDX[a], CH_IDX[b]
        A[i, j] = A[j, i] = 1.0
    return A
# fmt: on


# ─── Metadata Parsing ─────────────────────────────────────────────────────────

def parse_session_info(path: str) -> Dict:
    """Parse a session_info.gnr file into a metadata dict."""
    info = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Recording Device:"):
                info["device"] = line.split(":", 1)[1].strip()
            elif line.startswith("Protocol:"):
                info["protocol"] = line.split(":", 1)[1].strip()
            elif line.startswith("Arithmetic task:"):
                info["arithmetic_annotated"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("category="):
                info["category"] = line.split("=", 1)[1].strip()
    return info


def get_subject_label(spreadsheet_df: pd.DataFrame, subject_id: str) -> Optional[int]:
    """Returns 1 for Patient, 0 for Control. None if not found."""
    row = spreadsheet_df[spreadsheet_df["sn"] == subject_id]
    if row.empty:
        return None
    return 1 if row.iloc[0]["category"] == "Patient" else 0


# ─── Signal Loading & Harmonization ───────────────────────────────────────────

def load_edf_standardized(edf_path: str) -> Optional[np.ndarray]:
    """
    Load an EDF file, normalize channel names, select 19 standard channels,
    resample to TARGET_SFREQ. Returns (19, T) float32 array or None.
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception:
        return None

    # Rename channels using the norm map
    rename_map = {ch: CH_NORM[ch] for ch in raw.ch_names if ch in CH_NORM}
    raw.rename_channels(rename_map)

    # Keep only standard 19 channels
    available = [ch for ch in STANDARD_19 if ch in raw.ch_names]
    if len(available) < 15:   # require at least 15 of 19
        return None
    raw.pick_channels(available, ordered=True)

    # Resample if needed
    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, npad="auto")

    data = raw.get_data().astype(np.float32)  # (C, T)

    # Pad or fill missing channels with zeros
    if data.shape[0] < 19:
        pad = np.zeros((19 - data.shape[0], data.shape[1]), dtype=np.float32)
        data = np.vstack([data, pad])

    return data  # (19, T)


def z_score_normalize(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel Z-score normalization."""
    mu  = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + eps
    return (data - mu) / std


def window_signal(data: np.ndarray, window_size: int = TARGET_TLEN,
                  stride: int = None) -> List[np.ndarray]:
    """
    Slice a (19, T) array into overlapping windows of (19, window_size).
    If T < window_size, zero-pad. Returns list of windows.
    """
    if stride is None:
        stride = window_size // 2  # 50% overlap

    T = data.shape[1]
    if T < window_size:
        pad = np.zeros((19, window_size - T), dtype=np.float32)
        data = np.concatenate([data, pad], axis=1)
        return [data]

    windows = []
    start = 0
    while start + window_size <= data.shape[1]:
        windows.append(data[:, start: start + window_size])
        start += stride
    return windows


# ─── Dataset Class ────────────────────────────────────────────────────────────

class ASZEDSession:
    """Container for a single EEG recording session."""
    __slots__ = ["subject_id", "label", "device_id", "protocol_id",
                 "phase_data", "phase_type_ids"]

    def __init__(self, subject_id, label, device_id, protocol_id,
                 phase_data, phase_type_ids):
        self.subject_id    = subject_id
        self.label         = label
        self.device_id     = device_id
        self.protocol_id   = protocol_id
        self.phase_data    = phase_data     # dict: {phase_name: [(19, 1024), ...]}
        self.phase_type_ids = phase_type_ids  # dict: {phase_name: int}


class ASZEDDataset(Dataset):
    """
    PyTorch Dataset for ASZED. Each __getitem__ returns all phase windows
    for a single recording session, along with meta-tensors.

    Returns dict:
      windows      : (N_windows, 19, 1024) — per phase windows
      phase_ids    : (N_windows,) — phase type index for each window
      device_id    : scalar int
      protocol_id  : scalar int
      label        : scalar int (0=Control, 1=Patient)
      subject_id   : str
    """

    def __init__(
        self,
        eeg_root: Path = EEG_ROOT,
        spreadsheet: Path = SPREADSHEET,
        selected_phases: Optional[List[str]] = None,
        min_duration_s: float = 3.0,
        window_size: int = TARGET_TLEN,
        augment: bool = False,
        subject_ids: Optional[List[str]] = None,
    ):
        self.eeg_root        = eeg_root
        self.augment         = augment
        self.window_size     = window_size
        self.min_duration_s  = min_duration_s
        self.selected_phases = selected_phases  # None = use all phases

        df = pd.read_csv(spreadsheet)
        self._labels = {
            row["sn"]: (1 if row["category"] == "Patient" else 0)
            for _, row in df.iterrows()
        }

        print("Scanning dataset...")
        self.samples: List[ASZEDSession] = []
        self._scan(subject_ids)
        print(f"  → {len(self.samples)} sessions loaded.")

    def _scan(self, allowed_subjects: Optional[List[str]]):
        """Walk all subset/subject/session directories and load metadata."""
        for subset in sorted(self.eeg_root.iterdir()):
            if not subset.is_dir() or not subset.name.startswith("subset_"):
                continue

            for subj_dir in sorted(subset.iterdir()):
                if not subj_dir.is_dir() or not subj_dir.name.startswith("subject_"):
                    continue
                subj_id = subj_dir.name

                if allowed_subjects and subj_id not in allowed_subjects:
                    continue

                label = self._labels.get(subj_id)
                if label is None:
                    continue

                for sess_dir in sorted(subj_dir.iterdir()):
                    if not sess_dir.is_dir() or not sess_dir.name.isdigit():
                        continue

                    info_path = sess_dir / "session_info.gnr"
                    if not info_path.exists():
                        continue

                    meta = parse_session_info(str(info_path))
                    protocol = meta.get("protocol", "1.1").strip("'")
                    device   = meta.get("device", "CONTEK-KT2400")

                    dev_id  = DEVICE2IDX.get(device, 0)
                    prot_id = PROTOCOL2IDX.get(protocol, 0)
                    phase_labels = PROTOCOL_PHASES.get(protocol, PROTOCOL_PHASES["1.1"])

                    edf_files = sorted(sess_dir.glob("Phase *.edf"))
                    if not edf_files:
                        continue

                    phase_data    = {}
                    phase_type_ids = {}

                    for idx, edf in enumerate(edf_files):
                        phase_name = f"Phase {idx + 1}"
                        if idx < len(phase_labels):
                            ptype = phase_labels[idx]
                        else:
                            ptype = "Rest"

                        if self.selected_phases and ptype not in self.selected_phases:
                            continue

                        data = load_edf_standardized(str(edf))
                        if data is None:
                            continue

                        # Skip very short recordings
                        if data.shape[1] / TARGET_SFREQ < self.min_duration_s:
                            continue

                        data = z_score_normalize(data)
                        windows = window_signal(data, self.window_size)
                        if windows:
                            phase_data[phase_name]     = windows
                            phase_type_ids[phase_name] = PHASE2IDX.get(ptype, 0)

                    if phase_data:
                        self.samples.append(
                            ASZEDSession(subj_id, label, dev_id,
                                         prot_id, phase_data, phase_type_ids)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sess = self.samples[idx]

        all_windows  = []
        all_phase_ids = []

        for phase_name, windows in sess.phase_data.items():
            ptype_id = sess.phase_type_ids[phase_name]
            for w in windows:
                x = torch.tensor(w, dtype=torch.float32)
                if self.augment:
                    x = self._augment(x)
                all_windows.append(x)
                all_phase_ids.append(ptype_id)

        return {
            "windows":     torch.stack(all_windows, dim=0),   # (N, 19, 1024)
            "phase_ids":   torch.tensor(all_phase_ids, dtype=torch.long),
            "device_id":   torch.tensor(sess.device_id, dtype=torch.long),
            "protocol_id": torch.tensor(sess.protocol_id, dtype=torch.long),
            "label":       torch.tensor(sess.label, dtype=torch.long),
            "subject_id":  sess.subject_id,
        }

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Light augmentation: Gaussian noise + channel dropout + time masking."""
        # Gaussian noise
        x = x + torch.randn_like(x) * 0.02

        # Channel dropout (zero out 1 random channel)
        if torch.rand(1).item() < 0.3:
            ch = torch.randint(0, 19, (1,)).item()
            x[ch] = 0.0

        # Time masking (mask a random 64-sample window)
        if torch.rand(1).item() < 0.3:
            t_start = torch.randint(0, x.shape[-1] - 64, (1,)).item()
            x[:, t_start: t_start + 64] = 0.0

        return x


def collate_sessions(batch: List[Dict]) -> Dict:
    """
    Custom collate: pads session windows to the same count per batch.
    Returns padded tensors for batched training.
    """
    max_n = max(b["windows"].shape[0] for b in batch)

    windows_list  = []
    phase_id_list = []
    mask_list     = []
    dev_ids, prot_ids, labels, subj_ids = [], [], [], []

    for b in batch:
        n = b["windows"].shape[0]
        pad = max_n - n

        w  = b["windows"]
        ph = b["phase_ids"]

        if pad > 0:
            w  = torch.cat([w, torch.zeros(pad, 19, TARGET_TLEN)], dim=0)
            ph = torch.cat([ph, torch.full((pad,), -1, dtype=torch.long)], dim=0)

        mask = torch.zeros(max_n, dtype=torch.bool)
        mask[n:] = True   # True = padded (ignored in attention/loss)

        windows_list.append(w)
        phase_id_list.append(ph)
        mask_list.append(mask)
        dev_ids.append(b["device_id"])
        prot_ids.append(b["protocol_id"])
        labels.append(b["label"])
        subj_ids.append(b["subject_id"])

    return {
        "windows":     torch.stack(windows_list),   # (B, N, 19, 1024)
        "phase_ids":   torch.stack(phase_id_list),  # (B, N)
        "pad_mask":    torch.stack(mask_list),       # (B, N) — True=padded
        "device_id":   torch.stack(dev_ids),         # (B,)
        "protocol_id": torch.stack(prot_ids),        # (B,)
        "label":       torch.stack(labels),          # (B,)
        "subject_ids": subj_ids,
    }


def get_subject_split(dataset: ASZEDDataset, n_folds: int = 5, fold: int = 0,
                      seed: int = 42) -> Tuple[ASZEDDataset, ASZEDDataset]:
    """Subject-level stratified k-fold split to prevent data leakage."""
    import random
    random.seed(seed)

    # Collect unique subjects per label
    patient_subjs  = sorted({s.subject_id for s in dataset.samples if s.label == 1})
    control_subjs  = sorted({s.subject_id for s in dataset.samples if s.label == 0})

    random.shuffle(patient_subjs)
    random.shuffle(control_subjs)

    def kfold_split(items, k, f):
        size = len(items) // k
        test = items[f * size: (f + 1) * size]
        train = items[:f * size] + items[(f + 1) * size:]
        return train, test

    p_train, p_test = kfold_split(patient_subjs, n_folds, fold)
    c_train, c_test = kfold_split(control_subjs, n_folds, fold)

    train_ids = set(p_train + c_train)
    test_ids  = set(p_test  + c_test)

    # Filter samples
    train_ds = ASZEDDataset.__new__(ASZEDDataset)
    train_ds.__dict__.update({k: v for k, v in dataset.__dict__.items()
                               if k != "samples"})
    train_ds.samples = [s for s in dataset.samples if s.subject_id in train_ids]
    train_ds.augment = True

    test_ds = ASZEDDataset.__new__(ASZEDDataset)
    test_ds.__dict__.update({k: v for k, v in dataset.__dict__.items()
                              if k != "samples"})
    test_ds.samples = [s for s in dataset.samples if s.subject_id in test_ids]
    test_ds.augment = False

    return train_ds, test_ds
