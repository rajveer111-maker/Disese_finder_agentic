import os
import sys
import warnings
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score,
    cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

MODE        = "test_split"
MODEL_PATH  = "best_model.h5"
INPUT_PATH  = None
DATA_ROOT   = ("/kaggle/input/datasets/nikhilkushwaha2529/"
               "alzheimer-data/Alzheimer - Copy/SegmentedData")
SPLIT_IDX   = "split_test_idx.npy"
BATCH_SIZE  = 256
DEVICE_STR  = None

N_CHANNELS  = 19
SEQ_LEN     = 2500
N_CLASSES   = 3
CLASS_NAMES = ["AD", "CN", "FTD"]
LABEL_MAP   = {"AD": 0, "CN": 1, "FTD": 2, "HC": 1}
DEVICE      = torch.device(DEVICE_STR) if DEVICE_STR else torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, n_channels=N_CHANNELS, seq_len=SEQ_LEN, n_classes=N_CLASSES):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(n_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.spec_gate = SpectralGate(64, seq_len)
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
            nn.Linear(128, n_classes),
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


def load_model_h5(filepath, device):
    model = NeuroFormer()
    with h5py.File(filepath, "r") as f:
        sd = {k: torch.from_numpy(np.array(f[k])) for k in f.keys()}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def normalize(X):
    return (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)


class InferDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


@torch.no_grad()
def run_inference(model, X, batch_size=BATCH_SIZE, device=DEVICE):
    loader = DataLoader(InferDataset(X), batch_size=batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)
    preds_all, probs_all = [], []
    for xb in loader:
        xb = xb.to(device)
        with torch.cuda.amp.autocast():
            logits = model(xb)
        pr = F.softmax(logits.float(), dim=-1)
        preds_all.extend(pr.argmax(1).cpu().numpy())
        probs_all.extend(pr.cpu().numpy())
    return np.array(preds_all, dtype=np.int64), np.array(probs_all)


def print_metrics(labs, preds, probs):
    acc   = accuracy_score(labs, preds)
    f1    = f1_score(labs, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(labs, preds)
    mcc   = matthews_corrcoef(labs, preds)
    try:
        lb  = label_binarize(labs, classes=list(range(N_CLASSES)))
        auc = roc_auc_score(lb, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0
    sep = "-" * 42
    print(f"\n{sep}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Macro F1      : {f1:.4f}")
    print(f"  Cohen's k     : {kappa:.4f}")
    print(f"  MCC           : {mcc:.4f}")
    print(f"  AUC-ROC       : {auc:.4f}")
    print(sep)
    cm = confusion_matrix(labs, preds)
    print("\n  Confusion Matrix (rows=True, cols=Predicted):")
    print("       " + "".join(f"{n:>8s}" for n in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:5s}: " + "".join(f"{v:8d}" for v in row))


def mode_single(model):
    arr = np.load(INPUT_PATH).astype(np.float32)
    if arr.shape == (SEQ_LEN, N_CHANNELS):
        arr = arr.T
    if arr.shape != (N_CHANNELS, SEQ_LEN):
        raise ValueError(f"Expected ({N_CHANNELS},{SEQ_LEN}), got {arr.shape}")
    X            = normalize(arr[np.newaxis])
    preds, probs = run_inference(model, X, batch_size=1)
    label_idx    = preds[0]
    print(f"\n  Input      : {INPUT_PATH}")
    print(f"  Predicted  : {CLASS_NAMES[label_idx]}  (class {label_idx})")
    print(f"  Probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(probs[0, i] * 30)
        print(f"    {name:5s}: {probs[0, i]:.4f}  {bar}")


def mode_folder(model):
    files = sorted(f for f in os.listdir(INPUT_PATH) if f.endswith(".npy"))
    if not files:
        raise RuntimeError(f"No .npy files in {INPUT_PATH!r}")
    X_list, fnames = [], []
    for f in files:
        arr = np.load(os.path.join(INPUT_PATH, f)).astype(np.float32)
        if arr.shape == (SEQ_LEN, N_CHANNELS):
            arr = arr.T
        if arr.shape != (N_CHANNELS, SEQ_LEN):
            print(f"  [skip] {f} — shape {arr.shape}")
            continue
        X_list.append(arr)
        fnames.append(f)
    if not X_list:
        raise RuntimeError("No valid segments found.")
    X            = normalize(np.stack(X_list))
    preds, probs = run_inference(model, X)
    print(f"\n  {'File':<35} {'Pred':>6}  {'P(AD)':>7}  {'P(CN)':>7}  {'P(FTD)':>7}")
    print("  " + "-" * 68)
    for fname, pred, prob in zip(fnames, preds, probs):
        print(f"  {fname:<35} {CLASS_NAMES[pred]:>6}  "
              f"{prob[0]:>7.4f}  {prob[1]:>7.4f}  {prob[2]:>7.4f}")
    print(f"\n  Total : {len(preds)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(preds == i).sum()} ({(preds == i).mean() * 100:.1f}%)")


def mode_test_split(model):
    print("  Loading full dataset ...")
    X_all, y_all = [], []
    for cls in sorted(os.listdir(DATA_ROOT)):
        path = os.path.join(DATA_ROOT, cls)
        if not os.path.isdir(path) or cls not in LABEL_MAP:
            continue
        for f in sorted(os.listdir(path)):
            if not f.endswith(".npy"):
                continue
            arr = np.load(os.path.join(path, f)).astype(np.float32)
            if arr.shape == (N_CHANNELS, SEQ_LEN):
                X_all.append(arr)
                y_all.append(LABEL_MAP[cls])
    X_all = normalize(np.stack(X_all))
    y_all = np.array(y_all, dtype=np.int64)
    idx   = np.load(SPLIT_IDX)
    X_te, y_te = X_all[idx], y_all[idx]
    print(f"  Test set : {len(X_te)} segments")
    preds, probs = run_inference(model, X_te)
    print_metrics(y_te, preds, probs)


if __name__ == "__main__":
    print("=" * 50)
    print("  NeuroFormer — Inference")
    print("=" * 50)
    print(f"  Device : {DEVICE}")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Mode   : {MODE}")

    if not os.path.isfile(MODEL_PATH):
        sys.exit(f"[ERROR] Model file not found: {MODEL_PATH}")

    model = load_model_h5(MODEL_PATH, DEVICE)
    print(f"  Params : {sum(p.numel() for p in model.parameters()):,}")

    if MODE == "single":
        mode_single(model)
    elif MODE == "folder":
        mode_folder(model)
    elif MODE == "test_split":
        mode_test_split(model)
    else:
        sys.exit(f"[ERROR] Unknown MODE: {MODE!r}. Choose single / folder / test_split")
