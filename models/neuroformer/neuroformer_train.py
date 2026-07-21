import os
import copy
import random
import warnings
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

CFG = dict(
    data_root    = ("/kaggle/input/datasets/nikhilkushwaha2529/"
                   "alzheimer-data/Alzheimer - Copy/SegmentedData"),
    n_channels   = 19,
    seq_len      = 2500,
    n_classes    = 3,
    batch_size   = 128,
    lr           = 1e-3,
    weight_decay = 1e-4,
    epochs       = 80,
    patience     = 15,
    label_smooth = 0.10,
    mixup_alpha  = 0.20,
    best_model_path = "best_model.h5",
    split_dir    = ".",
    seed         = 42,
)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP   = {"AD": 0, "CN": 1, "FTD": 2, "HC": 1}
CLASS_NAMES = ["AD", "CN", "FTD"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CFG["seed"])


def load_dataset(root):
    X, y = [], []
    for cls in sorted(os.listdir(root)):
        path = os.path.join(root, cls)
        if not os.path.isdir(path) or cls not in LABEL_MAP:
            continue
        for f in sorted(os.listdir(path)):
            if not f.endswith(".npy"):
                continue
            arr = np.load(os.path.join(path, f))
            if arr.shape == (CFG["n_channels"], CFG["seq_len"]):
                X.append(arr.astype(np.float32))
                y.append(LABEL_MAP[cls])
    if not X:
        raise RuntimeError("No valid EEG segments found.")
    return np.stack(X), np.array(y, dtype=np.int64)


def normalize(X):
    return (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)


def split_and_save(X, y, cfg):
    idx = np.arange(len(X))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx, y, test_size=0.30, stratify=y, random_state=cfg["seed"]
    )
    idx_val, idx_te, _, _ = train_test_split(
        idx_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=cfg["seed"]
    )
    sd = cfg["split_dir"]
    np.save(os.path.join(sd, "split_train_idx.npy"), idx_tr)
    np.save(os.path.join(sd, "split_val_idx.npy"),   idx_val)
    np.save(os.path.join(sd, "split_test_idx.npy"),  idx_te)
    np.save(os.path.join(sd, "label_map.npy"), LABEL_MAP, allow_pickle=True)
    print(f"  Splits saved to {sd}/")
    return (X[idx_tr], X[idx_val], X[idx_te], y[idx_tr], y[idx_val], y[idx_te])


class EEGDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X       = torch.from_numpy(X).float()
        self.y       = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            if random.random() < 0.5:
                x = x + 0.02 * torch.randn_like(x)
            if random.random() < 0.3:
                x = torch.roll(x, random.randint(-50, 50), dims=-1)
            if random.random() < 0.2:
                x[random.randint(0, x.shape[0] - 1)] = 0.0
        return x, self.y[idx]


class LabelSmoothCE(nn.Module):
    def __init__(self, n_classes=3, smoothing=0.1):
        super().__init__()
        self.nc = n_classes
        self.sm = smoothing

    def forward(self, logits, targets):
        log_p   = F.log_softmax(logits, dim=-1)
        eps     = self.sm / (self.nc - 1)
        one_hot = torch.full_like(log_p, eps)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.sm)
        return -(one_hot * log_p).sum(dim=-1).mean()


def mixup_batch(xb, yb, alpha):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(xb.size(0), device=xb.device)
    return lam * xb + (1 - lam) * xb[idx], yb, yb[idx], lam


class SpectralGate(nn.Module):
    def __init__(self, channels, seq_len, max_bins=256):
        super().__init__()
        self.seq_len  = seq_len
        self.max_bins = max_bins
        self.mlp = nn.Sequential(
            nn.Linear(max_bins, 64),
            nn.GELU(),
            nn.Linear(64, max_bins),
            nn.Sigmoid(),
        )

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x32  = x.float()
            Xf   = torch.fft.rfft(x32, dim=-1)
            spec = Xf.abs()[:, :, : self.max_bins].mean(1)
            gate = self.mlp(spec)
            pad  = Xf.size(-1) - self.max_bins
            if pad > 0:
                gate = torch.cat(
                    [gate, torch.ones(x.size(0), pad, device=x.device)], dim=-1
                )
            x_filt = torch.fft.irfft(Xf * gate.unsqueeze(1), n=self.seq_len, dim=-1)
        return x + x_filt.to(x.dtype)


class NeuralOscillatoryCell(nn.Module):
    def __init__(self, channels, seq_len=2500, fs=500):
        super().__init__()
        bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 45)]
        self.seq_len    = seq_len
        self.modulators = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, 1, groups=channels, bias=False),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            ) for _ in bands
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * len(bands), channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        freqs = torch.fft.rfftfreq(seq_len, d=1.0 / fs)
        masks = torch.stack([
            ((freqs >= lo) & (freqs <= hi)).float() for lo, hi in bands
        ])
        self.register_buffer("masks", masks)

    def _bandpass(self, x, mask):
        return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * mask, n=self.seq_len, dim=-1)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x32   = x.float()
            bands = [
                mod(self._bandpass(x32, self.masks[i]))
                for i, mod in enumerate(self.modulators)
            ]
        fused = torch.cat([b.to(x.dtype) for b in bands], dim=1)
        return x + self.fuse(fused)


class HierarchicalCapsuleRouting(nn.Module):
    def __init__(self, in_dim, out_dim, num_caps=8, routings=3):
        super().__init__()
        self.num_caps = num_caps
        self.routings = routings
        self.W        = nn.Parameter(torch.randn(1, num_caps, in_dim, out_dim) * 0.02)

    def forward(self, x):
        B, N, D  = x.shape
        x_in     = x.unsqueeze(1).unsqueeze(-1)
        W        = self.W.expand(B, -1, -1, -1)
        u_hat    = torch.matmul(W, x_in).squeeze(-1)
        b        = torch.zeros(B, self.num_caps, N, device=x.device)
        for _ in range(self.routings):
            c = torch.softmax(b, dim=1)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=2)
            v = s / (s.norm(dim=-1, keepdim=True) + 1e-8)
            b = b + (u_hat * v.unsqueeze(2)).sum(-1)
        return v


class MemoryAugmentedAttention(nn.Module):
    def __init__(self, dim, mem_slots=64):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_slots, dim) * 0.02)
        self.scale  = dim ** -0.5
        self.norm   = nn.LayerNorm(dim)

    def forward(self, x):
        attn    = torch.softmax(x @ self.memory.T * self.scale, dim=-1)
        mem_out = attn @ self.memory
        return self.norm(x + mem_out)


class MetaAdaptiveNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        mu  = x.mean(dim=-1, keepdim=True)
        sig = x.std(dim=-1, keepdim=True) + 1e-8
        return self.gamma * (x - mu) / sig + self.beta


class CrossDimensionalAttention(nn.Module):
    def __init__(self, channels, seq_len):
        super().__init__()
        self.ch_gate  = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels, channels), nn.Sigmoid(),
        )
        self.tmp_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels, seq_len), nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        ac = self.ch_gate(x).unsqueeze(-1) * x
        at = self.tmp_gate(x).unsqueeze(1) * x
        return x + self.proj(torch.cat([ac, at], dim=1))


class DynamicGraphConvolution(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.edge_net   = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels),
        )
        self.graph_conv = nn.Linear(channels, channels)
        self.norm       = nn.LayerNorm(channels)

    def forward(self, x):
        xt  = x.permute(0, 2, 1)
        adj = torch.softmax(xt @ xt.transpose(-1, -2) / (xt.size(-1) ** 0.5), dim=-1)
        out = self.graph_conv(adj @ xt)
        return self.norm(xt + out).permute(0, 2, 1)


class AdaptiveFeaturePyramid(nn.Module):
    def __init__(self, channels, levels=4):
        super().__init__()
        self.levels = levels
        self.fuse   = nn.Sequential(
            nn.Conv1d(channels * levels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        T     = x.size(-1)
        parts = []
        for l in range(self.levels):
            factor = 2 ** l
            down   = F.avg_pool1d(x, kernel_size=factor, stride=factor, padding=0)
            up     = F.interpolate(down, size=T, mode="linear", align_corners=False)
            parts.append(up)
        return x + self.fuse(torch.cat(parts, dim=1))


class ConditionalPositionalEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cond_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels, channels), nn.GELU(),
        )
        self.mod = nn.Linear(channels, channels)

    def forward(self, x):
        cond = self.cond_net(x)
        pe   = self.mod(cond).unsqueeze(-1)
        return x + pe


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, shortcut):
        if not self.training or self.drop_prob == 0.0:
            return x + shortcut
        keep = torch.rand(x.size(0), 1, 1, device=x.device) > self.drop_prob
        return x * keep.float() + shortcut


class MultiScaleBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.k3  = nn.Conv1d(ch, ch, 3,  padding=1, groups=ch)
        self.k7  = nn.Conv1d(ch, ch, 7,  padding=3, groups=ch)
        self.k15 = nn.Conv1d(ch, ch, 15, padding=7, groups=ch)
        self.mix = nn.Sequential(
            nn.Conv1d(ch * 3, ch, 1, bias=False),
            nn.BatchNorm1d(ch),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.mix(torch.cat([self.k3(x), self.k7(x), self.k15(x)], dim=1))


class OscillatoryModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depth = nn.Sequential(
            nn.Conv1d(channels, channels, 1, groups=channels),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.depth(x)


class NeuroFormer(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        C  = cfg["n_channels"]
        nc = cfg["n_classes"]

        self.proj = nn.Sequential(
            nn.Conv1d(C, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.spec_gate = SpectralGate(64, cfg["seq_len"])
        self.osc       = OscillatoryModule(64)

        self.enc1 = nn.Sequential(
            nn.Conv1d(64, 128, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.ms1 = MultiScaleBlock(128)

        self.enc2 = nn.Sequential(
            nn.Conv1d(128, 256, 5, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.ms2 = MultiScaleBlock(256)

        self.attn    = nn.MultiheadAttention(256, 4, dropout=0.1, batch_first=True)
        self.attn_ln = nn.LayerNorm(256)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, nc),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.spec_gate(x)
        x = self.osc(x)
        x = self.ms1(self.enc1(x))
        x = self.ms2(self.enc2(x))
        t      = x.permute(0, 2, 1)
        t_a, _ = self.attn(t, t, t)
        x      = self.attn_ln(t + t_a).permute(0, 2, 1)
        return self.head(self.pool(x))


def save_model_h5(model, filepath):
    with h5py.File(filepath, "w") as f:
        for key, val in model.state_dict().items():
            f.create_dataset(key, data=val.detach().cpu().numpy())
    print(f"  Checkpoint saved -> {filepath}")


def load_model_h5(model, filepath, device=None):
    device = device or DEVICE
    with h5py.File(filepath, "r") as f:
        sd = {k: torch.tensor(np.array(f[k])) for k in f.keys()}
    model.load_state_dict(sd, strict=True)
    return model.to(device)


def train_one_epoch(model, loader, opt, crit, scaler, alpha):
    model.train()
    total_loss = total_corr = n = 0
    for xb, yb in loader:
        xb, yb           = xb.to(DEVICE), yb.to(DEVICE)
        xm, ya, yb2, lam = mixup_batch(xb, yb, alpha)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            out  = model(xm)
            loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb2)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        preds       = out.argmax(1)
        total_corr += (lam * (preds == ya).float() + (1 - lam) * (preds == yb2).float()).sum().item()
        total_loss += loss.item() * xb.size(0)
        n          += xb.size(0)
    return total_loss / n, total_corr / n


@torch.no_grad()
def evaluate(model, loader, crit=None):
    model.eval()
    preds_all, labs_all, probs_all = [], [], []
    total_loss = n = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.cuda.amp.autocast():
            out = model(xb)
            if crit is not None:
                total_loss += crit(out, yb).item() * xb.size(0)
                n          += xb.size(0)
        pr = F.softmax(out.float(), dim=-1)
        preds_all.extend(pr.argmax(1).cpu().numpy())
        labs_all .extend(yb.cpu().numpy())
        probs_all.extend(pr.cpu().numpy())
    return (np.array(preds_all), np.array(labs_all),
            np.array(probs_all), total_loss / max(n, 1))


def compute_metrics(labs, preds, probs, nc):
    acc   = accuracy_score(labs, preds)
    f1    = f1_score(labs, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(labs, preds)
    mcc   = matthews_corrcoef(labs, preds)
    try:
        lb  = label_binarize(labs, classes=list(range(nc)))
        auc = roc_auc_score(lb, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0
    return dict(acc=acc, f1=f1, kappa=kappa, mcc=mcc, auc=auc)


def print_confusion(labs, preds, names):
    cm = confusion_matrix(labs, preds)
    print("\n  Confusion Matrix (rows=True, cols=Predicted):")
    print("       " + "".join(f"{n:>8s}" for n in names))
    for i, row in enumerate(cm):
        print(f"  {names[i]:5s}: " + "".join(f"{v:8d}" for v in row))


if __name__ == "__main__":
    sep = "=" * 60
    print(sep)
    print("  NeuroFormer (Tiny) — Alzheimer EEG Classification")
    print(sep)
    print(f"  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU    : {props.name}  |  VRAM {props.total_memory/1e9:.1f} GB")

    print(f"\n[1] Loading data...")
    X, y = load_dataset(CFG["data_root"])
    X    = normalize(X)
    print(f"  Total : {len(X)}  shape : {X.shape[1:]}")
    for i, nm in enumerate(CLASS_NAMES):
        print(f"  {nm} : {(y == i).sum()}")

    print("\n[2] Splitting (70/15/15) ...")
    Xtr, Xval, Xte, ytr, yval, yte = split_and_save(X, y, CFG)
    print(f"  Train {len(Xtr)} | Val {len(Xval)} | Test {len(Xte)}")

    tr_ld = DataLoader(EEGDataset(Xtr,  ytr,  augment=True),
                       batch_size=CFG["batch_size"], shuffle=True,
                       num_workers=2, pin_memory=True, drop_last=True)
    va_ld = DataLoader(EEGDataset(Xval, yval),
                       batch_size=CFG["batch_size"] * 2, shuffle=False,
                       num_workers=2, pin_memory=True)
    te_ld = DataLoader(EEGDataset(Xte,  yte),
                       batch_size=CFG["batch_size"] * 2, shuffle=False,
                       num_workers=2, pin_memory=True)

    model    = NeuroFormer(CFG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[3] Model ready  |  params : {n_params:,}")

    opt    = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"], eta_min=1e-5)
    crit   = LabelSmoothCE(CFG["n_classes"], CFG["label_smooth"])
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    best_state   = None
    patience_ctr = 0

    print(f"\n[4] Training ...")
    print(f"  {'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  {'VaAcc':>7}  "
          f"{'VaF1':>7}  {'kappa':>7}  {'AUC':>7}  {'LR':>9}")
    print("  " + "-" * 72)

    for ep in range(1, CFG["epochs"] + 1):
        tr_loss, _           = train_one_epoch(model, tr_ld, opt, crit, scaler, CFG["mixup_alpha"])
        vp, vl, vpr, va_loss = evaluate(model, va_ld, crit)
        vm                   = compute_metrics(vl, vp, vpr, CFG["n_classes"])
        sched.step()

        print(f"  {ep:4d}  {tr_loss:8.4f}  {va_loss:8.4f}  "
              f"{vm['acc']:7.4f}  {vm['f1']:7.4f}  "
              f"{vm['kappa']:7.4f}  {vm['auc']:7.4f}  "
              f"{opt.param_groups[0]['lr']:9.2e}")

        if vm["acc"] > best_val_acc:
            best_val_acc = vm["acc"]
            best_state   = copy.deepcopy(model.state_dict())
            save_model_h5(model, CFG["best_model_path"])
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= CFG["patience"]:
            print(f"\n  Early stopping at epoch {ep}")
            break

    print("\n[5] Final test evaluation ...")
    model.load_state_dict(best_state)
    tp, tl, tpr, _ = evaluate(model, te_ld)
    tm = compute_metrics(tl, tp, tpr, CFG["n_classes"])

    print(f"\n{sep}")
    print("  FINAL TEST RESULTS")
    print(sep)
    print(f"  Best Val Acc : {best_val_acc * 100:.2f}%")
    print(f"  Test Acc     : {tm['acc'] * 100:.2f}%")
    print(f"  Test F1      : {tm['f1']:.4f}")
    print(f"  Cohen's k    : {tm['kappa']:.4f}")
    print(f"  MCC          : {tm['mcc']:.4f}")
    print(f"  AUC-ROC      : {tm['auc']:.4f}")
    print_confusion(tl, tp, CLASS_NAMES)
    print(sep)
