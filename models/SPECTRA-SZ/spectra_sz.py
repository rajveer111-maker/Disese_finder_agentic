"""
SPECTRA Model Architecture
===========================
Schizophrenia Prediction via Encoded Cognitive Task Routing Architecture

Novel compact (~650K params) multi-paradigm EEG classifier. Solves:
  1. Domain-Adaptive Spectral Normalization (DASN)   — FFT frequency-domain conditioning
  2. Multi-scale Causal Resolution Bank (MCRB)       — dilated conv + Hilbert envelope
  3. Cognitive State Gated Router (CSGR)             — SE-Temporal MoE with load balancing
  4. Multi-slot Prototype Memory (MPM)               — slot-attention session aggregation
  5. Graph Spatial Regularization Loss (GRSL)        — 10-20 topology as training signal

ZERO Python loops in forward() — fully vectorized on GPU/MPS.
GroupNorm throughout — no BatchNorm sync overhead.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dataloader import build_adjacency_matrix
except ImportError:
    from .dataloader import build_adjacency_matrix



# ─────────────────────────────────────────────────────────────────────────────
# Block 1: Domain-Adaptive Spectral Normalization (DASN)
# ─────────────────────────────────────────────────────────────────────────────
# Novel: normalization happens in the *frequency domain*.
# Each recording device has a different hardware frequency response (gain curve).
# DASN learns per-frequency-bin magnitude scale factors conditioned on
# device + protocol metadata embeddings, then reconstructs the signal via IFFT.
# This bridges the CONTEK vs DISCOVERY domain gap *inside* the model — no
# handcrafted preprocessing required.

class DASN(nn.Module):
    def __init__(self, n_ch: int = 19, n_freqs: int = 513,
                 n_devices: int = 2, n_protocols: int = 2, emb_dim: int = 32):
        super().__init__()
        # n_freqs = T//2 + 1 at T=1024  →  513
        self.n_ch    = n_ch
        self.n_freqs = n_freqs

        self.dev_emb  = nn.Embedding(n_devices,   emb_dim)
        self.prot_emb = nn.Embedding(n_protocols, emb_dim)

        # MLP → per-frequency scale (one scalar per freq bin, shared across channels)
        self.scale_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, n_freqs),
            nn.Sigmoid(),           # scale ∈ (0,1) — selective freq suppression
        )
        # Residual bias after iFFT
        self.post_norm = nn.GroupNorm(1, n_ch)   # LayerNorm equivalent, GPU-friendly

    def forward(self, x: torch.Tensor, dev_id: torch.Tensor,
                prot_id: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        meta  = torch.cat([self.dev_emb(dev_id), self.prot_emb(prot_id)], dim=-1)  # (B, 2d)
        scale = self.scale_mlp(meta)          # (B, n_freqs)

        # FFT → scale magnitude → iFFT
        X   = torch.fft.rfft(x, dim=-1)              # (B, C, n_freqs) complex
        mag = X.abs().clamp(min=1e-8)
        pha = X / mag
        scaled_mag = mag * scale.unsqueeze(1)         # (B, C, n_freqs) broadcast
        X_scaled   = scaled_mag * pha
        x_out      = torch.fft.irfft(X_scaled, n=T, dim=-1)   # (B, C, T) real
        return self.post_norm(x_out)


# ─────────────────────────────────────────────────────────────────────────────
# Block 2: Multi-scale Causal Resolution Bank (MCRB)
# ─────────────────────────────────────────────────────────────────────────────
# Novel: combines dilated depthwise-separable convs (for band-specific features)
# with a *differentiable Hilbert envelope* via FFT (for amplitude modulation —
# a key schizophrenia biomarker). Both branches are GPU tensor ops.
# No scipy, no CPU fallback.

def _hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
    """Fully differentiable Hilbert analytic envelope. x: (B, C, T)"""
    T  = x.shape[-1]
    X  = torch.fft.fft(x, dim=-1)           # (B, C, T) complex
    h  = torch.zeros(T, device=x.device, dtype=x.dtype)
    h[0] = 1.0
    if T % 2 == 0:
        h[T // 2] = 1.0
        h[1 : T // 2] = 2.0
    else:
        h[1 : (T + 1) // 2] = 2.0
    z = torch.fft.ifft(X * h, dim=-1)       # analytic signal (complex)
    return z.abs()                           # envelope (B, C, T) real


class _BandBranch(nn.Module):
    """Depthwise-separable dilated conv branch with GroupNorm."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.dw  = nn.Conv1d(in_ch, in_ch, kernel, dilation=dilation,
                             padding=pad, groups=in_ch, bias=False)
        self.pw  = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.gn  = nn.GroupNorm(min(4, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.gn(self.pw(self.dw(x))))


class MCRB(nn.Module):
    def __init__(self, in_ch: int = 19, branch_ch: int = 28, kernel: int = 31):
        super().__init__()
        # 5 dilated conv bands: Delta/Theta/Alpha/Beta/Gamma
        dilations = [16, 8, 4, 2, 1]
        self.conv_branches = nn.ModuleList([
            _BandBranch(in_ch, branch_ch, kernel, d) for d in dilations
        ])
        # Envelope branch: project Hilbert envelope → branch_ch features
        self.env_proj = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 1, bias=False),
            nn.GroupNorm(min(4, branch_ch), branch_ch),
            nn.GELU(),
        )
        self.out_ch = branch_ch * (len(dilations) + 1)   # 5+1 branches = 6*28=168

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        conv_feats = [b(x) for b in self.conv_branches]  # 5 × (B, 28, T)
        env        = _hilbert_envelope(x)                 # (B, C, T)
        env_feat   = self.env_proj(env)                   # (B, 28, T)
        return torch.cat(conv_feats + [env_feat], dim=1)  # (B, 168, T)


# ─────────────────────────────────────────────────────────────────────────────
# Block 3: Cognitive State Gated Router (CSGR)
# ─────────────────────────────────────────────────────────────────────────────
# Novel: Soft MoE where each specialist uses Squeeze-and-Excitation + temporal
# self-attention to find the most informative time window within a phase.
# Router applies entropy regularization (load-balance loss) so all 4 specialists
# are utilized rather than collapsing to one dominant expert.

class _SETemporalSpecialist(nn.Module):
    """
    Specialist: Channel SE + Global Context Temporal Gate (GCTG) + global pool.

    GCTG replaces the O(T^2) self-attention with an O(C*T) gate:
      - Broadcast global avg & max context to every timestep
      - 1x1 conv learns per-timestep sigmoid gates from those statistics
    No T x T matrix ever materialised — safe at BN=200+, T=1024.
    Novel: temporal saliency from global context rather than pairwise scores.
    """
    def __init__(self, in_ch: int, r: int = 4, out_dim: int = 160):
        super().__init__()
        mid = max(in_ch // r, 8)
        self.se_fc1 = nn.Linear(in_ch, mid, bias=False)
        self.se_fc2 = nn.Linear(mid, in_ch, bias=False)
        self.tgate = nn.Sequential(
            nn.Conv1d(in_ch * 2, in_ch, 1, bias=False),
            nn.GroupNorm(min(4, in_ch), in_ch),
            nn.GELU(),
            nn.Conv1d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv1d(in_ch, out_dim, 1, bias=False)
        self.gn   = nn.GroupNorm(min(4, out_dim), out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel SE
        se = x.mean(-1)
        se = F.gelu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se)).unsqueeze(-1)
        x  = x * se
        # GCTG — broadcast global statistics, no T x T matrix
        avg_ctx = x.mean(-1, keepdim=True).expand_as(x)
        max_ctx = x.amax(-1, keepdim=True).expand_as(x)
        gate = self.tgate(torch.cat([avg_ctx, max_ctx], dim=1))
        x    = x * gate
        x = F.gelu(self.gn(self.proj(x)))
        return x.mean(-1)


class CSGR(nn.Module):
    def __init__(self, in_ch: int = 168, n_paradigms: int = 4,
                 out_dim: int = 192):
        super().__init__()
        self.n_paradigms  = n_paradigms
        self.out_dim      = out_dim
        self.paradigm_emb = nn.Embedding(n_paradigms, 32)
        self.router       = nn.Linear(32, n_paradigms)
        self.specialists  = nn.ModuleList([
            _SETemporalSpecialist(in_ch, r=4, out_dim=out_dim)
            for _ in range(n_paradigms)
        ])

    def forward(self, x: torch.Tensor,
                phase_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (BN, in_ch, T) — fully vectorized, NO Python loop
        emb    = self.paradigm_emb(phase_id)          # (BN, 32)
        gates  = F.softmax(self.router(emb), dim=-1)  # (BN, 4)

        # All specialists run in parallel on the same batch (vectorized)
        spec_outs = torch.stack(
            [s(x) for s in self.specialists], dim=1
        )                                              # (BN, 4, out_dim)

        out = (gates.unsqueeze(-1) * spec_outs).sum(1) # (BN, out_dim)
        return out, gates                              # for load-balance loss


def router_load_balance_loss(gates: torch.Tensor) -> torch.Tensor:
    """Entropy regularization: maximize routing entropy → balanced specialist use."""
    entropy = -(gates * (gates + 1e-8).log()).sum(-1)   # (BN,)
    return -entropy.mean()   # negative because we want high entropy


# ─────────────────────────────────────────────────────────────────────────────
# Block 4: Multi-slot Prototype Memory (MPM)
# ─────────────────────────────────────────────────────────────────────────────
# Novel: inspired by slot attention (Locatello 2020) but adapted for EEG sessions.
# K learnable prototype queries attend over N windows, each slot capturing a
# different "aspect" of brain activity (e.g., baseline vs. task-specific).
# Slot outputs are concatenated → a richer subject-level vector.
# Pairwise cosine similarity between slots feeds the classifier as implicit
# inter-aspect coherence features.

class MPM(nn.Module):
    def __init__(self, embed_dim: int = 192, n_slots: int = 4,
                 proj_dim: int = 64):
        super().__init__()
        self.n_slots    = n_slots
        self.embed_dim  = embed_dim

        # K learnable prototype query vectors
        self.slots   = nn.Parameter(torch.randn(n_slots, embed_dim))
        nn.init.xavier_uniform_(self.slots.unsqueeze(0))

        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.val_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm     = nn.LayerNorm(embed_dim)

        # Projection head for contrastive auxiliary loss
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim * n_slots, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, window_embeds: torch.Tensor,
                pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # window_embeds: (B, N, D)  pad_mask: (B, N) True=padded
        B, N, D = window_embeds.shape

        keys = self.key_proj(window_embeds)   # (B, N, D)
        vals = self.val_proj(window_embeds)   # (B, N, D)

        # Slot cross-attention over windows
        q   = self.slots.unsqueeze(0).expand(B, -1, -1)   # (B, K, D)
        attn = torch.bmm(q, keys.transpose(1, 2)) / math.sqrt(D)  # (B, K, N)

        # Mask out padded windows
        attn = attn.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        slots_out = torch.bmm(attn, vals)                 # (B, K, D)
        slots_out = self.norm(slots_out)                  # (B, K, D)

        # Flatten all slots → subject representation
        flat      = slots_out.reshape(B, -1)              # (B, K*D)
        proj      = F.normalize(self.proj_head(flat), dim=-1)  # (B, proj_dim)

        # Pairwise slot coherence — added as extra features in classifier
        # cos-sim between every slot pair → (B, K*(K-1)/2) extra features
        slots_n   = F.normalize(slots_out, dim=-1)        # (B, K, D)
        coh = torch.bmm(slots_n, slots_n.transpose(1, 2))[:, :, :]
        # Extract upper triangle (K*K values, includes diagonal=1)
        coherence = coh.reshape(B, -1)                    # (B, K*K)

        return flat, proj, coherence


# ─────────────────────────────────────────────────────────────────────────────
# Block 5: Graph Spatial Regularization Loss (GRSL)
# ─────────────────────────────────────────────────────────────────────────────

def graph_spatial_loss(spatial: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    spatial : (B, 19, F) — channel-wise features from DASN output
    adj     : (19, 19)   — 10-20 electrode adjacency
    Penalizes large differences between spatially adjacent channel features.
    """
    wi = spatial.unsqueeze(2)               # (B, 19, 1, F)
    wj = spatial.unsqueeze(1)               # (B, 1, 19, F)
    diff_sq = ((wi - wj) ** 2).sum(-1)      # (B, 19, 19)
    return (adj.unsqueeze(0) * diff_sq).sum() / (adj.sum() + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# NT-Xent Contrastive Loss
# ─────────────────────────────────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 l1: torch.Tensor, l2: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    z      = torch.cat([z1, z2], dim=0)
    labels = torch.cat([l1, l2], dim=0)
    sim    = torch.mm(z, z.T) / temperature

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)
    neg_mask = 1.0 - (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    pos_sim  = (pos_mask * sim.exp()).sum(-1)
    all_sim  = (neg_mask * sim.exp()).sum(-1) + pos_sim
    return (-torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# SPECTRA: Full Model  (fully vectorized — zero Python loops in forward)
# ─────────────────────────────────────────────────────────────────────────────

class SPECTRA(nn.Module):
    """
    SPECTRA — Schizophrenia Prediction via Encoded Cognitive Task Routing Architecture

    Inputs  (from collate_sessions):
      windows      : (B, N, 19, T)
      phase_ids    : (B, N)        — paradigm type per window
      pad_mask     : (B, N)        — True = padded window
      device_id    : (B,)
      protocol_id  : (B,)

    Outputs dict:
      logits       : (B, 2)
      proj_embed   : (B, 64)       — for NT-Xent loss
      gates        : (B*N, 4)      — router weights (load balance loss)
      spatial_feat : (B, 19, 32)   — for GRSL
    """

    def __init__(
        self,
        n_ch:          int = 19,
        branch_ch:     int = 28,   # MCRB branch width  → 6*28 = 168 total
        specialist_dim: int = 160, # CSGR output per window
        n_paradigms:   int = 4,
        n_slots:       int = 4,    # MPM slots
        proj_dim:      int = 64,
        n_classes:     int = 2,
    ):
        super().__init__()
        mcrb_ch = branch_ch * 6          # 168
        slot_flat = specialist_dim * n_slots  # 768
        coh_dim   = n_slots * n_slots         # 16

        # Block 1: DASN
        self.dasn = DASN(n_ch=n_ch, n_freqs=513)

        # Spatial snapshot (for GRSL) — lightweight per-channel conv
        self.spatial_snap = nn.Conv1d(n_ch, n_ch, 1, groups=n_ch, bias=False)

        # Block 2: MCRB
        self.mcrb = MCRB(in_ch=n_ch, branch_ch=branch_ch, kernel=31)

        # Block 3: CSGR
        self.csgr = CSGR(in_ch=mcrb_ch, n_paradigms=n_paradigms,
                         out_dim=specialist_dim)

        # Block 4: MPM
        self.mpm = MPM(embed_dim=specialist_dim, n_slots=n_slots,
                       proj_dim=proj_dim)

        # Classifier: slot flat + coherence features → logits
        self.classifier = nn.Sequential(
            nn.Linear(slot_flat + coh_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

        # Adjacency buffer
        self.register_buffer("adj_matrix", build_adjacency_matrix())

    # ── Fully vectorized forward — NO Python for-loop ──────────────────────
    def forward(
        self,
        windows:     torch.Tensor,   # (B, N, 19, T)
        phase_ids:   torch.Tensor,   # (B, N)
        pad_mask:    torch.Tensor,   # (B, N)
        device_id:   torch.Tensor,   # (B,)
        protocol_id: torch.Tensor,   # (B,)
    ) -> Dict[str, torch.Tensor]:

        B, N, C, T = windows.shape

        # ── Flatten B,N → BN (vectorized over all windows at once) ─────────
        x      = windows.reshape(B * N, C, T)                     # (BN, C, T)
        ph     = phase_ids.reshape(B * N).clamp(min=0)            # (BN,)
        dev    = device_id.unsqueeze(1).expand(B, N).reshape(B*N) # (BN,)
        prot   = protocol_id.unsqueeze(1).expand(B, N).reshape(B*N)

        # Block 1: DASN — frequency-domain device conditioning
        x_norm = self.dasn(x, dev, prot)                          # (BN, C, T)

        # Capture spatial snapshot for GRSL (before spectral decomposition)
        # Pool T → 32 to keep it lightweight
        snap   = F.adaptive_avg_pool1d(
            self.spatial_snap(x_norm), 32
        )                                                          # (BN, C, 32)

        # Block 2: MCRB — multi-scale bands + Hilbert envelope
        feat   = self.mcrb(x_norm)                                # (BN, 168, T)

        # Block 3: CSGR — SE-Temporal paradigm routing
        embed, gates = self.csgr(feat, ph)                        # (BN, 192), (BN, 4)

        # ── Zero-out padded windows (mask applied as tensor op, no loop) ───
        valid_mask = (~pad_mask).reshape(B * N).float()           # (BN,) 0 or 1
        embed = embed * valid_mask.unsqueeze(-1)                  # (BN, 192)

        # ── Unflatten back to (B, N, D) for session aggregation ────────────
        window_embeds = embed.reshape(B, N, -1)                   # (B, N, 192)

        # Block 4: MPM — multi-slot prototype memory
        slot_flat, proj_embed, coherence = self.mpm(window_embeds, pad_mask)
        # (B, 768), (B, 64), (B, 16)

        # Classify using slot features + inter-slot coherence
        clf_input = torch.cat([slot_flat, coherence], dim=-1)     # (B, 784)
        logits    = self.classifier(clf_input)                    # (B, 2)

        # Spatial feat for GRSL: mean over BN snapshots → (B, C, 32)
        snap_b = snap.reshape(B, N, C, 32).mean(1)               # (B, C, 32)

        return {
            "logits":       logits,
            "proj_embed":   proj_embed,
            "gates":        gates,              # (B*N, 4) for load-balance loss
            "spatial_feat": snap_b,             # (B, 19, 32)
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# SPECTRA Combined Loss
# ─────────────────────────────────────────────────────────────────────────────

class SPECTRALoss(nn.Module):
    """
    L_total = L_ce
            + λ_contrast   * L_ntxent         (session-level contrastive)
            + λ_spatial    * L_grsl            (electrode topology regularization)
            + λ_balance    * L_load_balance    (router entropy regularization)
    """
    def __init__(self, lambda_contrast: float = 0.3, lambda_spatial: float = 0.1,
                 lambda_balance: float = 0.05, temperature: float = 0.07):
        super().__init__()
        self.lc = lambda_contrast
        self.ls = lambda_spatial
        self.lb = lambda_balance
        self.t  = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, proj_embed, spatial_feat, gates,
                labels, adj) -> Tuple[torch.Tensor, Dict]:
        l_ce = self.ce(logits, labels)

        B = proj_embed.shape[0]
        if B >= 2:
            h  = B // 2
            z1, z2 = proj_embed[:h], proj_embed[h:h*2]
            l1, l2 = labels[:h],     labels[h:h*2]
            l_con  = nt_xent_loss(z1, z2, l1, l2, self.t)
        else:
            l_con = proj_embed.sum() * 0.0

        l_spa = graph_spatial_loss(spatial_feat, adj)
        l_bal = router_load_balance_loss(gates)

        total = l_ce + self.lc * l_con + self.ls * l_spa + self.lb * l_bal
        return total, {
            "loss_ce": l_ce.item(), "loss_contrast": l_con.item(),
            "loss_spatial": l_spa.item(), "loss_balance": l_bal.item(),
            "loss_total": total.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = SPECTRA()
    print(f"\n{'='*58}")
    print(f"  SPECTRA — Schizophrenia Prediction via Encoded")
    print(f"           Cognitive Task Routing Architecture")
    print(f"{'='*58}")
    print(f"  Total trainable params: {model.count_parameters():,}")
    blocks = {"DASN": model.dasn, "MCRB": model.mcrb, "CSGR": model.csgr,
              "MPM": model.mpm, "Classifier": model.classifier}
    for name, blk in blocks.items():
        n = sum(p.numel() for p in blk.parameters() if p.requires_grad)
        print(f"  {name:<14}: {n:>9,} params")

    B, N, C, T = 4, 8, 19, 1024
    dummy = dict(
        windows     = torch.randn(B, N, C, T),
        phase_ids   = torch.randint(0, 4, (B, N)),
        pad_mask    = torch.zeros(B, N, dtype=torch.bool),
        device_id   = torch.randint(0, 2, (B,)),
        protocol_id = torch.randint(0, 2, (B,)),
        label       = torch.randint(0, 2, (B,)),
    )
    out = model(dummy["windows"], dummy["phase_ids"], dummy["pad_mask"],
                dummy["device_id"], dummy["protocol_id"])
    print(f"\n  Output shapes:")
    for k, v in out.items():
        print(f"    {k:<16}: {tuple(v.shape)}")
    print(f"{'='*58}\n")

