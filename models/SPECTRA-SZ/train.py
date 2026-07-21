"""
SPECTRA SESSION-SPLIT TRAINING PIPELINE
=======================================

Fixes:
1) Does NOT use spreadsheet category labels.
2) Reads subject label from each subject's userfile.gnr:
      category=Control  -> 0
      category=Patient  -> 1

Adds:
1) Train on session 1, validate on session 2.
2) Optional test on other sessions if available.
3) Phase-wise metrics.
4) Session-wise metrics.
5) Saves:
      best_model.pt
      val_predictions.csv
      test_predictions.csv
      final_results.json

Expected imports:
    dataloader.py -> your dataset/helper code
    model.py      -> your SPECTRA model code

Important:
    This script reuses your original preprocessing:
        load_edf_standardized()
        z_score_normalize()
        window_signal()

Manual config only. No argparse.
"""

import os
import re
import json
import time
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# IMPORT YOUR EXISTING FILES
# ============================================================

from dataloader import (
    TARGET_TLEN,
    TARGET_SFREQ,
    DEVICE2IDX,
    PROTOCOL2IDX,
    PHASE2IDX,
    PROTOCOL_PHASES,
    parse_session_info,
    load_edf_standardized,
    z_score_normalize,
    window_signal,
    collate_sessions,
)

from spectra_sz import (
    SPECTRA,
    SPECTRALoss,
)


# ============================================================
# MANUAL CONFIG
# ============================================================

CFG = {
    # Dataset root containing subset_1, subset_2, ...
    "eeg_root": "/kaggle/input/datasets/nikhilkushwaha2529/aszed-153/ASZED-153/ASZED/version_1.1/node_1",

    # Output folder
    "output_dir": "./spectra_session_split_run",

    # Split style
    # train_session = "1"
    # val_session   = "2"
    # test_sessions = "others" means all sessions except train and val
    "train_session": "1",
    "val_session": "2",
    "test_sessions": "others",

    # Data
    "selected_phases": None,      # None or ["Rest"], ["Arithmetic"], etc.
    "min_duration_s": 3.0,
    "window_size": 1024,

    # Training
    "epochs": 200,
    "batch_size": 4,
    "lr": 3e-5,
    "weight_decay": 1e-5,
    "patience": 150,
    "seed": 42,
    "use_amp": True,

    # Loss
    "lambda_contrast": 0.3,
    "lambda_spatial": 0.1,
    "lambda_balance": 0.05,
    "temperature": 0.1,

    # Device
    "device": (
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    ),
}


# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LABEL FROM userfile.gnr
# ============================================================

def parse_userfile_gnr(path):
    """
    Reads subject metadata from userfile.gnr.

    Example:
        age=28
        sex=F
        category=Control
        language=English
        site=...

    Returns dict.
    """
    info = {}

    path = Path(path)

    if not path.exists():
        return info

    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or "=" not in line:
                continue

            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()

    return info


def label_from_category(category):
    """
    Converts category string to numeric label.

    Control -> 0
    Patient -> 1
    """
    if category is None:
        return None

    c = str(category).strip().lower()

    if c == "control":
        return 0

    if c == "patient":
        return 1

    return None


def get_subject_label_from_userfile(subject_dir):
    """
    Reads label only from subject_dir/userfile.gnr.
    Does not use spreadsheet.
    """
    userfile = Path(subject_dir) / "userfile.gnr"
    info = parse_userfile_gnr(userfile)
    return label_from_category(info.get("category"))


# ============================================================
# SESSION OBJECT
# ============================================================

class ASZEDSessionByUserfile:
    __slots__ = [
        "subject_id",
        "session_id",
        "label",
        "device_id",
        "protocol_id",
        "phase_data",
        "phase_type_ids",
    ]

    def __init__(
        self,
        subject_id,
        session_id,
        label,
        device_id,
        protocol_id,
        phase_data,
        phase_type_ids,
    ):
        self.subject_id = subject_id
        self.session_id = session_id
        self.label = label
        self.device_id = device_id
        self.protocol_id = protocol_id
        self.phase_data = phase_data
        self.phase_type_ids = phase_type_ids


# ============================================================
# DATASET USING userfile.gnr LABELS
# ============================================================

class ASZEDUserfileDataset(Dataset):
    """
    Dataset where each item is one session.

    Label source:
        subject_x/userfile.gnr -> category

    Returned dict shape is compatible with your original collate_sessions():
        windows      : (N, 19, 1024)
        phase_ids    : (N,)
        device_id    : scalar
        protocol_id  : scalar
        label        : scalar
        subject_id   : str

    Added:
        session_id   : str
    """

    def __init__(
        self,
        eeg_root,
        selected_phases=None,
        min_duration_s=3.0,
        window_size=1024,
        augment=False,
        allowed_sessions=None,
    ):
        self.eeg_root = Path(eeg_root)
        self.selected_phases = selected_phases
        self.min_duration_s = min_duration_s
        self.window_size = window_size
        self.augment = augment
        self.allowed_sessions = allowed_sessions

        self.samples = []

        print("Scanning dataset using userfile.gnr labels...")
        self._scan()
        print(f"Loaded sessions: {len(self.samples)}")

    def _scan(self):
        for subset_dir in sorted(self.eeg_root.iterdir()):

            if not subset_dir.is_dir() or not subset_dir.name.startswith("subset_"):
                continue

            for subject_dir in sorted(subset_dir.iterdir()):

                if not subject_dir.is_dir() or not subject_dir.name.startswith("subject_"):
                    continue

                subject_id = subject_dir.name

                label = get_subject_label_from_userfile(subject_dir)

                if label is None:
                    print(f"Skipping {subject_id}: missing/unknown userfile.gnr category")
                    continue

                for session_dir in sorted(subject_dir.iterdir()):

                    if not session_dir.is_dir():
                        continue

                    if not session_dir.name.isdigit():
                        continue

                    session_id = session_dir.name

                    if self.allowed_sessions is not None:
                        if session_id not in self.allowed_sessions:
                            continue

                    info_path = session_dir / "session_info.gnr"

                    if info_path.exists():
                        meta = parse_session_info(str(info_path))
                    else:
                        meta = {}

                    protocol = str(meta.get("protocol", "1.1")).strip("'")
                    device_name = meta.get("device", "CONTEK-KT2400")

                    device_id = DEVICE2IDX.get(device_name, 0)
                    protocol_id = PROTOCOL2IDX.get(protocol, 0)

                    phase_labels = PROTOCOL_PHASES.get(
                        protocol,
                        PROTOCOL_PHASES["1.1"],
                    )

                    edf_files = sorted(
                        session_dir.glob("Phase *.edf"),
                        key=lambda p: extract_phase_number(p.name),
                    )

                    if not edf_files:
                        edf_files = sorted(session_dir.glob("*.edf"))

                    if not edf_files:
                        continue

                    phase_data = {}
                    phase_type_ids = {}

                    for idx, edf_path in enumerate(edf_files):

                        if idx < len(phase_labels):
                            phase_type = phase_labels[idx]
                        else:
                            phase_type = "Rest"

                        if self.selected_phases is not None:
                            if phase_type not in self.selected_phases:
                                continue

                        data = load_edf_standardized(str(edf_path))

                        if data is None:
                            continue

                        if data.shape[1] / TARGET_SFREQ < self.min_duration_s:
                            continue

                        data = z_score_normalize(data)

                        windows = window_signal(
                            data,
                            window_size=self.window_size,
                            stride=self.window_size // 2,
                        )

                        if not windows:
                            continue

                        phase_name = f"Phase {idx + 1}"
                        phase_data[phase_name] = windows
                        phase_type_ids[phase_name] = PHASE2IDX.get(phase_type, 0)

                    if phase_data:
                        self.samples.append(
                            ASZEDSessionByUserfile(
                                subject_id=subject_id,
                                session_id=session_id,
                                label=label,
                                device_id=device_id,
                                protocol_id=protocol_id,
                                phase_data=phase_data,
                                phase_type_ids=phase_type_ids,
                            )
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sess = self.samples[idx]

        all_windows = []
        all_phase_ids = []

        for phase_name, windows in sess.phase_data.items():
            phase_id = sess.phase_type_ids[phase_name]

            for w in windows:
                x = torch.tensor(w, dtype=torch.float32)

                if self.augment:
                    x = self._augment(x)

                all_windows.append(x)
                all_phase_ids.append(phase_id)

        return {
            "windows": torch.stack(all_windows, dim=0),
            "phase_ids": torch.tensor(all_phase_ids, dtype=torch.long),
            "device_id": torch.tensor(sess.device_id, dtype=torch.long),
            "protocol_id": torch.tensor(sess.protocol_id, dtype=torch.long),
            "label": torch.tensor(sess.label, dtype=torch.long),
            "subject_id": sess.subject_id,
            "session_id": sess.session_id,
        }

    def _augment(self, x):
        # Same style as your original dataset.
        x = x + torch.randn_like(x) * 0.02

        if torch.rand(1).item() < 0.3:
            ch = torch.randint(0, 19, (1,)).item()
            x[ch] = 0.0

        if torch.rand(1).item() < 0.3 and x.shape[-1] > 64:
            t_start = torch.randint(0, x.shape[-1] - 64, (1,)).item()
            x[:, t_start:t_start + 64] = 0.0

        return x


def extract_phase_number(filename):
    m = re.search(r"Phase\s*(\d+)", filename, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 9999


# ============================================================
# COLLATE WITH SESSION ID
# ============================================================

def collate_sessions_with_session_id(batch):
    """
    Same as your collate_sessions but keeps session_ids.
    """
    out = collate_sessions(batch)
    out["session_ids"] = [b["session_id"] for b in batch]
    return out


# ============================================================
# SPLIT DATASETS: TRAIN SESSION 1, VAL SESSION 2
# ============================================================

def create_session_split_datasets():
    eeg_root = CFG["eeg_root"]

    train_ds = ASZEDUserfileDataset(
        eeg_root=eeg_root,
        selected_phases=CFG["selected_phases"],
        min_duration_s=CFG["min_duration_s"],
        window_size=CFG["window_size"],
        augment=True,
        allowed_sessions={str(CFG["train_session"])},
    )

    val_ds = ASZEDUserfileDataset(
        eeg_root=eeg_root,
        selected_phases=CFG["selected_phases"],
        min_duration_s=CFG["min_duration_s"],
        window_size=CFG["window_size"],
        augment=False,
        allowed_sessions={str(CFG["val_session"])},
    )

    if CFG["test_sessions"] == "others":
        all_ds = ASZEDUserfileDataset(
            eeg_root=eeg_root,
            selected_phases=CFG["selected_phases"],
            min_duration_s=CFG["min_duration_s"],
            window_size=CFG["window_size"],
            augment=False,
            allowed_sessions=None,
        )

        blocked = {str(CFG["train_session"]), str(CFG["val_session"])}

        test_ds = ASZEDUserfileDataset.__new__(ASZEDUserfileDataset)
        test_ds.__dict__.update(all_ds.__dict__)
        test_ds.samples = [
            s for s in all_ds.samples
            if s.session_id not in blocked
        ]
        test_ds.augment = False

    elif CFG["test_sessions"] is None:
        test_ds = None

    else:
        test_ds = ASZEDUserfileDataset(
            eeg_root=eeg_root,
            selected_phases=CFG["selected_phases"],
            min_duration_s=CFG["min_duration_s"],
            window_size=CFG["window_size"],
            augment=False,
            allowed_sessions={str(x) for x in CFG["test_sessions"]},
        )

    print("\nSESSION SPLIT")
    print("-------------")
    print(f"Train session {CFG['train_session']} samples: {len(train_ds)}")
    print(f"Val session   {CFG['val_session']} samples: {len(val_ds)}")

    if test_ds is not None:
        print(f"Test samples: {len(test_ds)}")
    else:
        print("Test samples: None")

    print_label_counts("Train", train_ds)
    print_label_counts("Val", val_ds)

    if test_ds is not None:
        print_label_counts("Test", test_ds)

    return train_ds, val_ds, test_ds


def print_label_counts(name, ds):
    counts = defaultdict(int)

    for s in ds.samples:
        counts[s.label] += 1

    print(
        f"{name} label count -> "
        f"Control: {counts[0]}, Patient: {counts[1]}"
    )


# ============================================================
# METRICS HELPERS
# ============================================================

PHASE_ID_TO_NAME = {v: k for k, v in PHASE2IDX.items()}


def safe_auc(y_true, y_prob):
    try:
        if len(set(y_true)) < 2:
            return 0.0
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return 0.0


def compute_metrics(y_true, y_pred, y_prob):
    if len(y_true) == 0:
        return {
            "acc": 0.0,
            "f1": 0.0,
            "auc": 0.0,
            "cm": [[0, 0], [0, 0]],
        }

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(safe_auc(y_true, y_prob)),
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def phase_subset_batch(batch, phase_id):
    """
    Creates a batch where only windows of one phase type are kept.

    Important:
        Some sessions may not contain that phase.
        Return None if no window exists for the requested phase.
    """
    windows = batch["windows"]
    phase_ids = batch["phase_ids"]
    pad_mask = batch["pad_mask"]

    B = windows.shape[0]

    new_windows = []
    new_phase_ids = []
    new_pad_mask = []

    valid_any = False

    for b in range(B):
        valid = (~pad_mask[b]) & (phase_ids[b] == phase_id)

        if valid.sum().item() == 0:
            # dummy padded one window
            w = torch.zeros(1, windows.shape[2], windows.shape[3], dtype=windows.dtype)
            ph = torch.full((1,), phase_id, dtype=torch.long)
            pm = torch.ones(1, dtype=torch.bool)
        else:
            valid_any = True
            w = windows[b][valid]
            ph = phase_ids[b][valid]
            pm = torch.zeros(w.shape[0], dtype=torch.bool)

        new_windows.append(w)
        new_phase_ids.append(ph)
        new_pad_mask.append(pm)

    if not valid_any:
        return None

    max_n = max(w.shape[0] for w in new_windows)

    padded_windows = []
    padded_phase_ids = []
    padded_masks = []

    for w, ph, pm in zip(new_windows, new_phase_ids, new_pad_mask):
        pad = max_n - w.shape[0]

        if pad > 0:
            w = torch.cat(
                [
                    w,
                    torch.zeros(
                        pad,
                        windows.shape[2],
                        windows.shape[3],
                        dtype=windows.dtype,
                    ),
                ],
                dim=0,
            )

            ph = torch.cat(
                [
                    ph,
                    torch.full((pad,), phase_id, dtype=torch.long),
                ],
                dim=0,
            )

            pm = torch.cat(
                [
                    pm,
                    torch.ones(pad, dtype=torch.bool),
                ],
                dim=0,
            )

        padded_windows.append(w)
        padded_phase_ids.append(ph)
        padded_masks.append(pm)

    return {
        "windows": torch.stack(padded_windows, dim=0),
        "phase_ids": torch.stack(padded_phase_ids, dim=0),
        "pad_mask": torch.stack(padded_masks, dim=0),
        "device_id": batch["device_id"],
        "protocol_id": batch["protocol_id"],
        "label": batch["label"],
        "subject_ids": batch["subject_ids"],
        "session_ids": batch["session_ids"],
    }


# ============================================================
# TRAIN
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    adj,
    scaler,
):
    model.train()

    total_loss = 0.0
    all_y = []
    all_pred = []

    for batch in loader:
        windows = batch["windows"].to(device)
        phase_ids = batch["phase_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        device_id = batch["device_id"].to(device)
        protocol_id = batch["protocol_id"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        use_amp_now = scaler is not None and device.type == "cuda"

        with torch.cuda.amp.autocast(enabled=use_amp_now):
            out = model(
                windows=windows,
                phase_ids=phase_ids,
                pad_mask=pad_mask,
                device_id=device_id,
                protocol_id=protocol_id,
            )

            loss, _ = criterion(
                logits=out["logits"],
                proj_embed=out["proj_embed"],
                spatial_feat=out["spatial_feat"],
                gates=out["gates"],
                labels=labels,
                adj=adj,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

        preds = out["logits"].argmax(dim=-1)

        all_y.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_y, all_pred)
    f1 = f1_score(all_y, all_pred, zero_division=0)

    return total_loss / max(len(loader), 1), acc, f1


# ============================================================
# EVALUATE SESSION-WISE
# ============================================================

@torch.no_grad()
def evaluate_sessionwise(
    model,
    loader,
    criterion,
    device,
    adj,
    save_predictions_path=None,
):
    model.eval()

    total_loss = 0.0

    y_true = []
    y_pred = []
    y_prob = []

    rows = []

    for batch in loader:
        windows = batch["windows"].to(device)
        phase_ids = batch["phase_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        device_id = batch["device_id"].to(device)
        protocol_id = batch["protocol_id"].to(device)
        labels = batch["label"].to(device)

        out = model(
            windows=windows,
            phase_ids=phase_ids,
            pad_mask=pad_mask,
            device_id=device_id,
            protocol_id=protocol_id,
        )

        loss, _ = criterion(
            logits=out["logits"],
            proj_embed=out["proj_embed"],
            spatial_feat=out["spatial_feat"],
            gates=out["gates"],
            labels=labels,
            adj=adj,
        )

        total_loss += loss.item()

        probs = torch.softmax(out["logits"], dim=-1)[:, 1]
        preds = out["logits"].argmax(dim=-1)

        labels_np = labels.detach().cpu().numpy().tolist()
        preds_np = preds.detach().cpu().numpy().tolist()
        probs_np = probs.detach().cpu().numpy().tolist()

        y_true.extend(labels_np)
        y_pred.extend(preds_np)
        y_prob.extend(probs_np)

        for i in range(len(labels_np)):
            rows.append(
                {
                    "subject_id": batch["subject_ids"][i],
                    "session_id": batch["session_ids"][i],
                    "true_label": labels_np[i],
                    "pred_label": preds_np[i],
                    "prob_patient": probs_np[i],
                    "correct": int(labels_np[i] == preds_np[i]),
                }
            )

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = float(total_loss / max(len(loader), 1))

    if save_predictions_path is not None:
        pd.DataFrame(rows).to_csv(save_predictions_path, index=False)

    return metrics, rows


# ============================================================
# EVALUATE PHASE-WISE
# ============================================================

@torch.no_grad()
def evaluate_phasewise(
    model,
    loader,
    device,
):
    """
    Phase-wise accuracy:
        For each phase type, keep only windows of that phase and run prediction.

    This tells you:
        Rest-only accuracy
        Arithmetic-only accuracy
        ASSR-only accuracy
        Oddball-only accuracy
    """
    model.eval()

    phase_results = {}

    for phase_name, phase_id in PHASE2IDX.items():

        y_true = []
        y_pred = []
        y_prob = []

        for batch in loader:
            phase_batch = phase_subset_batch(batch, phase_id)

            if phase_batch is None:
                continue

            windows = phase_batch["windows"].to(device)
            phase_ids = phase_batch["phase_ids"].to(device)
            pad_mask = phase_batch["pad_mask"].to(device)
            device_id = phase_batch["device_id"].to(device)
            protocol_id = phase_batch["protocol_id"].to(device)
            labels = phase_batch["label"].to(device)

            out = model(
                windows=windows,
                phase_ids=phase_ids,
                pad_mask=pad_mask,
                device_id=device_id,
                protocol_id=protocol_id,
            )

            probs = torch.softmax(out["logits"], dim=-1)[:, 1]
            preds = out["logits"].argmax(dim=-1)

            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())
            y_prob.extend(probs.detach().cpu().numpy().tolist())

        phase_results[phase_name] = compute_metrics(y_true, y_pred, y_prob)
        phase_results[phase_name]["n_sessions_with_phase"] = len(y_true)

    return phase_results


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(CFG["seed"])

    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(CFG["device"])
    print(f"\nUsing device: {device}")

    train_ds, val_ds, test_ds = create_session_split_datasets()

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty. Check train_session and eeg_root.")

    if len(val_ds) == 0:
        raise RuntimeError("Validation dataset is empty. Check val_session and eeg_root.")

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        collate_fn=collate_sessions_with_session_id,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        collate_fn=collate_sessions_with_session_id,
        num_workers=0,
    )

    test_loader = None

    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=CFG["batch_size"],
            shuffle=False,
            collate_fn=collate_sessions_with_session_id,
            num_workers=0,
        )

    model = SPECTRA().to(device)

    print(f"\nParameters: {model.count_parameters():,}")

    criterion = SPECTRALoss(
        lambda_contrast=CFG["lambda_contrast"],
        lambda_spatial=CFG["lambda_spatial"],
        lambda_balance=CFG["lambda_balance"],
        temperature=CFG["temperature"],
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
    )

    scaler = (
        torch.cuda.amp.GradScaler()
        if CFG["use_amp"] and device.type == "cuda"
        else None
    )

    adj = model.adj_matrix.to(device)

    best_auc = -1.0
    best_acc = -1.0
    patience_counter = 0

    best_path = output_dir / "best_model.pt"

    print("\nSTART TRAINING")
    print("=" * 70)

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_f1 = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            adj=adj,
            scaler=scaler,
        )

        val_metrics, _ = evaluate_sessionwise(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            adj=adj,
            save_predictions_path=None,
        )

        scheduler.step()

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d} | "
            f"TrainLoss={tr_loss:.4f} "
            f"TrainAcc={tr_acc:.4f} "
            f"TrainF1={tr_f1:.4f} | "
            f"ValLoss={val_metrics['loss']:.4f} "
            f"ValAcc={val_metrics['acc']:.4f} "
            f"ValF1={val_metrics['f1']:.4f} "
            f"ValAUC={val_metrics['auc']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Prefer AUC when possible, but if AUC is 0 because only one class exists,
        # fall back to accuracy.
        score_auc = val_metrics["auc"]
        score_acc = val_metrics["acc"]

        improved = False

        if score_auc > best_auc:
            improved = True
        elif score_auc == best_auc and score_acc > best_acc:
            improved = True

        if improved:
            best_auc = score_auc
            best_acc = score_acc
            patience_counter = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_val_auc": best_auc,
                    "best_val_acc": best_acc,
                    "cfg": CFG,
                },
                best_path,
            )

            print("✓ Best model saved")

        else:
            patience_counter += 1

        if patience_counter >= CFG["patience"]:
            print("\nEarly stopping triggered")
            break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\nLOADING BEST MODEL")

    ckpt = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ========================================================
    # FINAL VAL SESSION-WISE AND PHASE-WISE
    # ========================================================

    print("\nFINAL VALIDATION EVALUATION")

    val_metrics, _ = evaluate_sessionwise(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        adj=adj,
        save_predictions_path=output_dir / "val_predictions.csv",
    )

    val_phase_metrics = evaluate_phasewise(
        model=model,
        loader=val_loader,
        device=device,
    )

    print("\nVAL SESSION-WISE")
    print(json.dumps(val_metrics, indent=4))

    print("\nVAL PHASE-WISE")
    print(json.dumps(val_phase_metrics, indent=4))

    # ========================================================
    # TEST IF AVAILABLE
    # ========================================================

    test_metrics = None
    test_phase_metrics = None

    if test_loader is not None:
        print("\nFINAL TEST EVALUATION")

        test_metrics, _ = evaluate_sessionwise(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            adj=adj,
            save_predictions_path=output_dir / "test_predictions.csv",
        )

        test_phase_metrics = evaluate_phasewise(
            model=model,
            loader=test_loader,
            device=device,
        )

        print("\nTEST SESSION-WISE")
        print(json.dumps(test_metrics, indent=4))

        print("\nTEST PHASE-WISE")
        print(json.dumps(test_phase_metrics, indent=4))

    final_results = {
        "best_val_auc": float(best_auc),
        "best_val_acc": float(best_acc),
        "val_sessionwise": val_metrics,
        "val_phasewise": val_phase_metrics,
        "test_sessionwise": test_metrics,
        "test_phasewise": test_phase_metrics,
    }

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("\nDONE")
    print(f"Saved best model to: {best_path}")
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
