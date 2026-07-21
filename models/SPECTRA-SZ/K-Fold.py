"""
PANTHEON Training Script
========================
Subject-stratified 5-Fold cross-validation training pipeline.
Trains the PANTHEON model on ASZED with combined loss.
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)
from pathlib import Path

# Add pantheon dir to path
sys.path.insert(0, str(Path('/kaggle/input/datasets/nikhilkushwaha2529/aszed-153/ASZED-153').parent))
# from model import SPECTRA, SPECTRALoss
# from dataloader import ASZEDDataset, collate_sessions, get_subject_split

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    # Paths
    "eeg_root":     "/kaggle/input/datasets/nikhilkushwaha2529/aszed-153/ASZED-153/ASZED/version_1.1/node_1",
    "spreadsheet":  "/kaggle/input/datasets/nikhilkushwaha2529/aszed-153/ASZED-153/ASZED_SpreadSheet.csv",
    "output_dir":   "spectra/runs",

    # Dataset
    "selected_phases":  None,
    "min_duration_s":   3.0,
    "window_size":      1024,
    "n_folds":          5,

    # Training
    "epochs":           80,
    "batch_size":       4,   # per-GPU; DataParallel doubles effective batch
    "lr":               3e-4,
    "weight_decay":     1e-4,
    "patience":         15,
    "seed":             42,
    "use_amp":          True,  # mixed precision (fp16) — halves VRAM

    # Loss weights
    "lambda_contrast":  0.3,
    "lambda_spatial":   0.1,
    "lambda_balance":   0.05,
    "temperature":      0.07,

    # Device
    "device": "cuda" if torch.cuda.is_available() else
              "mps"  if torch.backends.mps.is_available() else "cpu",

    # Device
    "device": "cuda" if torch.cuda.is_available() else
              "mps"  if torch.backends.mps.is_available() else "cpu",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Train / Eval
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, adj, scaler):
    model.train()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        windows     = batch["windows"].to(device)
        phase_ids   = batch["phase_ids"].to(device)
        pad_mask    = batch["pad_mask"].to(device)
        device_id   = batch["device_id"].to(device)
        protocol_id = batch["protocol_id"].to(device)
        labels      = batch["label"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            out = model(windows, phase_ids, pad_mask, device_id, protocol_id)
            loss, loss_info = criterion(
                logits       = out["logits"],
                proj_embed   = out["proj_embed"],
                spatial_feat = out["spatial_feat"],
                gates        = out["gates"],
                labels       = labels,
                adj          = adj,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / max(n_batches, 1), acc, f1


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, adj):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        windows     = batch["windows"].to(device)
        phase_ids   = batch["phase_ids"].to(device)
        pad_mask    = batch["pad_mask"].to(device)
        device_id   = batch["device_id"].to(device)
        protocol_id = batch["protocol_id"].to(device)
        labels      = batch["label"].to(device)

        with torch.cuda.amp.autocast(enabled=CFG["use_amp"] and device.type == "cuda"):
            out = model(windows, phase_ids, pad_mask, device_id, protocol_id)
            loss, _ = criterion(
                logits       = out["logits"],
                proj_embed   = out["proj_embed"],
                spatial_feat = out["spatial_feat"],
                gates        = out["gates"],
                labels       = labels,
                adj          = adj,
            )

        total_loss += loss.item()
        n_batches  += 1

        probs = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds).tolist()

    return (total_loss / max(n_batches, 1)), acc, f1, auc, cm


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop (k-fold)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(CFG["seed"])
    device = torch.device(CFG["device"])
    print(f"Using device: {device}")

    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset once
    print("\nLoading ASZED dataset...")
    full_dataset = ASZEDDataset(
        eeg_root       = Path(CFG["eeg_root"]),
        spreadsheet    = Path(CFG["spreadsheet"]),
        selected_phases = CFG["selected_phases"],
        min_duration_s  = CFG["min_duration_s"],
        window_size     = CFG["window_size"],
        augment         = False,     # augment controlled per split
    )
    print(f"Total sessions: {len(full_dataset)}")

    # Cross-validation
    fold_results = []

    for fold in range(CFG["n_folds"]):
        print(f"\n{'='*55}")
        print(f"  FOLD {fold + 1}/{CFG['n_folds']}")
        print(f"{'='*55}")

        train_ds, val_ds = get_subject_split(
            full_dataset, n_folds=CFG["n_folds"],
            fold=fold, seed=CFG["seed"]
        )
        print(f"  Train sessions: {len(train_ds)} | Val sessions: {len(val_ds)}")

        train_loader = DataLoader(
            train_ds, batch_size=CFG["batch_size"],
            shuffle=True, collate_fn=collate_sessions,
            num_workers=0, pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=CFG["batch_size"],
            shuffle=False, collate_fn=collate_sessions,
            num_workers=0, pin_memory=False,
        )

        # ── Model: wrap in DataParallel for multi-GPU ───────────────────────
        base_model = SPECTRA().to(device)
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"  Using {n_gpus} GPUs via DataParallel")
            model = torch.nn.DataParallel(base_model)
        else:
            model = base_model
        adj = base_model.adj_matrix.to(device)

        criterion = SPECTRALoss(
            lambda_contrast = CFG["lambda_contrast"],
            lambda_spatial  = CFG["lambda_spatial"],
            lambda_balance  = CFG["lambda_balance"],
            temperature     = CFG["temperature"],
        )
        optimizer = optim.AdamW(
            base_model.parameters(),
            lr=CFG["lr"], weight_decay=CFG["weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2
        )
        scaler = torch.cuda.amp.GradScaler() if CFG["use_amp"] and device.type == "cuda" else None

        best_val_auc  = 0.0
        best_ckpt     = output_dir / f"best_fold{fold}.pt"
        patience_cnt  = 0
        fold_log      = []

        n_params = base_model.count_parameters()
        print(f"  Parameters: {n_params:,}  |  GPUs: {max(n_gpus,1)}  |  AMP: {scaler is not None}")

        for epoch in range(1, CFG["epochs"] + 1):
            t0 = time.time()

            tr_loss, tr_acc, tr_f1 = train_epoch(
                model, train_loader, optimizer, criterion, device, adj, scaler
            )
            val_loss, val_acc, val_f1, val_auc, val_cm = eval_epoch(
                model, val_loader, criterion, device, adj
            )
            scheduler.step()

            elapsed = time.time() - t0
            row = {
                "epoch": epoch,
                "tr_loss": round(tr_loss, 4),
                "tr_acc":  round(tr_acc, 4),
                "tr_f1":   round(tr_f1, 4),
                "val_loss": round(val_loss, 4),
                "val_acc":  round(val_acc, 4),
                "val_f1":   round(val_f1, 4),
                "val_auc":  round(val_auc, 4),
                "time_s":  round(elapsed, 1),
            }
            fold_log.append(row)

            print(
                f"  Ep {epoch:03d} | "
                f"TrLoss={tr_loss:.4f} TrAcc={tr_acc:.3f} | "
                f"ValLoss={val_loss:.4f} ValAcc={val_acc:.3f} "
                f"ValF1={val_f1:.3f} ValAUC={val_auc:.3f} | {elapsed:.1f}s"
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "val_f1":  val_f1,
                    "cm":      val_cm,
                    "cfg":     CFG,
                }, best_ckpt)
                print(f"  ✓ New best AUC={val_auc:.4f} — checkpoint saved.")
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt >= CFG["patience"]:
                print(f"  Early stopping at epoch {epoch}.")
                break

        # Save fold log
        log_path = output_dir / f"fold{fold}_log.json"
        with open(log_path, "w") as f:
            json.dump(fold_log, f, indent=2)

        # Load best checkpoint for this fold's final metrics
        ckpt = torch.load(best_ckpt, map_location=device)
        fold_results.append({
            "fold":    fold,
            "val_auc": ckpt["val_auc"],
            "val_acc": ckpt["val_acc"],
            "val_f1":  ckpt["val_f1"],
            "cm":      ckpt["cm"],
        })
        print(f"\n  Fold {fold+1} best → AUC={ckpt['val_auc']:.4f} "
              f"Acc={ckpt['val_acc']:.4f} F1={ckpt['val_f1']:.4f}")

    # ── Cross-fold summary ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  CROSS-VALIDATION SUMMARY")
    print(f"{'='*55}")
    aucs = [r["val_auc"] for r in fold_results]
    accs = [r["val_acc"] for r in fold_results]
    f1s  = [r["val_f1"]  for r in fold_results]
    print(f"  AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Acc:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1:   {np.mean(f1s):.4f}  ± {np.std(f1s):.4f}")

    summary = {
        "mean_auc": round(float(np.mean(aucs)), 4),
        "std_auc":  round(float(np.std(aucs)), 4),
        "mean_acc": round(float(np.mean(accs)), 4),
        "std_acc":  round(float(np.std(accs)), 4),
        "mean_f1":  round(float(np.mean(f1s)), 4),
        "std_f1":   round(float(np.std(f1s)), 4),
        "folds":    fold_results,
        "cfg":      CFG,
    }
    summary_path = output_dir / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()

