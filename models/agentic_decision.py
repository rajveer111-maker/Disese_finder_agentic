"""
ANDI: Agentic Neurological Disorder Identifier
Master Orchestration Agent Policy Architecture -- A_top

Implements the agentic decision-policy formulation:
    A_top = <S, A, K, pi_top, C_CMO>

This module implements:
  - Telemetry feature extraction and spectral entropy gate Phi(t)
  - Policy Utility Score U(m|s_t) with four sub-rules R1..R4  [w = 0.25,0.25,0.30,0.20]
  - Sigmoid lambda gate for RAG / Heuristic branch mixing
  - Diagnostic uncertainty score delta_conflict for Virtual CMO
  - 5-class action space: {M_HC, M_PD, M_AD, M_SZ, empty_unroutable}

Reference: ANDI Architecture Specification (Manuscript Under Peer Review)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants matching paper formulation
# ─────────────────────────────────────────────────────────────────────────────

VALID_CHANNEL_COUNTS = {8, 16, 19, 32, 40, 64}
C_MAX = 64          # R4 normalisation denominator

SIGMA_MIN   = 0.01  # µV flatline threshold
H_SPEC_MIN  = 0.22  # Minimum spectral entropy for valid signal
H_SPEC_HIGH = 0.70  # Healthy broadband floor

W1, W2, W3, W4 = 0.25, 0.25, 0.30, 0.20  # Heuristic branch weights
SIGMOID_K  = 7.0
SIGMOID_A0 = 0.50

DELTA_CONFLICT_THRESHOLD = 0.40

M_HC   = "healthy_control"
M_PD   = "nhrn_pd"
M_AD   = "neuroformer"
M_SZ   = "spectra_sz"
M_NULL = "uncertain"

CLINICAL_KEYWORDS: Dict[str, List[str]] = {
    M_HC: ["healthy", "control", "normal", "cognitively normal", "cn", "baseline", "routine"],
    M_PD: ["parkinson", "pd", "tremor", "beta", "basal ganglia", "dopaminergic", "motor", "resonance"],
    M_AD: ["alzheimer", "ad", "dementia", "ftd", "frontotemporal", "theta", "synaptic", "cognitive decline"],
    M_SZ: ["schizophrenia", "sz", "psychosis", "gamma", "phase coupling", "auditory", "hallucination"],
}

C_TARGET: Dict[str, int] = {M_HC: 19, M_PD: 22, M_AD: 19, M_SZ: 19}

FORMAT_SUPPORT: Dict[str, List[str]] = {
    M_HC: ["csv","txt","npy","edf","bdf"],
    M_PD: ["csv","txt","npy","edf","bdf"],
    M_AD: ["csv","txt","npy","edf","bdf"],
    M_SZ: ["csv","txt","npy","edf","bdf"],
}

ALL_MODELS = [M_HC, M_PD, M_AD, M_SZ]


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_telemetry(data: np.ndarray, fs: float = 250.0) -> Dict[str, Any]:
    """Compute telemetry vector t = [D, sigma, n_hat, H_spec, C_active]."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[0] < data.shape[1]:
        data = data.T  # (channels, samples) -> (samples, channels)
    n_samples, n_channels = data.shape
    duration  = n_samples / fs
    sigma     = float(np.std(data))
    noise_hat = float(np.mean(np.sum(np.abs(np.diff(data, axis=0)), axis=1))) if n_samples > 1 else 0.0
    h_spec    = _compute_spectral_entropy(data, fs)
    c_active  = _resolve_channel_count(n_channels)
    return dict(duration=duration, sigma=sigma, noise_hat=noise_hat,
                h_spec=h_spec, c_active=c_active, shape=data.shape)


def _compute_spectral_entropy(data: np.ndarray, fs: float) -> float:
    """H_spec = -sum_f P(f) log2 P(f) / log2(F)  averaged over channels."""
    try:
        n = data.shape[0]
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        band  = (freqs >= 0.5) & (freqs <= 50.0)
        if not np.any(band):
            return 0.0
        psd = np.abs(np.fft.rfft(data, axis=0))[band, :] ** 2
        entropies = []
        for ch in range(psd.shape[1]):
            p = psd[:, ch]; total = p.sum()
            if total < 1e-12: entropies.append(0.0); continue
            p = np.clip(p / total, 1e-12, 1.0)
            F = len(p)
            entropies.append(float(np.clip(-np.sum(p * np.log2(p)) / np.log2(F), 0.0, 1.0)))
        return float(np.mean(entropies)) if entropies else 0.0
    except Exception as e:
        logger.warning(f"H_spec failed: {e}"); return 0.0


def _resolve_channel_count(n: int) -> int:
    return n if n in VALID_CHANNEL_COUNTS else min(VALID_CHANNEL_COUNTS, key=lambda c: abs(c - n))


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry Quality Gate  Phi(t)
# ─────────────────────────────────────────────────────────────────────────────

def quality_gate(telemetry: Dict[str, Any]) -> int:
    """Phi(t) = I(sigma > 0.01) * I(H_spec >= 0.22). Returns 1=valid, 0=reject."""
    return 1 if (telemetry["sigma"] > SIGMA_MIN and telemetry["h_spec"] >= H_SPEC_MIN) else 0


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic Policy Sub-Rules R1..R4
# ─────────────────────────────────────────────────────────────────────────────

def _r1(model: str, file_type: str) -> float:
    ft = file_type.lower().lstrip(".")
    if ft in ("edf","bdf","csv","npy"): return 1.0
    if ft == "txt": return 0.8
    return 0.6 if ft in FORMAT_SUPPORT.get(model, []) else 0.0

def _r2(model: str, c_active: int) -> float:
    return float(min(1.0, c_active / C_TARGET.get(model, 19)))

def _r3(model: str, referral: Any) -> float:
    if isinstance(referral, dict):
        text = " ".join(str(v) for v in referral.values()).lower()
    else:
        text = str(referral or "").lower()
    kws = CLINICAL_KEYWORDS.get(model, [])
    if not kws:
        return 0.0
    matched = [k for k in kws if k in text]
    if len(matched) >= 2:
        return 1.0
    elif len(matched) == 1:
        return 0.90
    return 0.0

def _r4(h_spec: float, c_active: int) -> float:
    return float(min(1.0, max(0.0, 0.60 * h_spec + 0.40 * (c_active / C_MAX))))


def compute_utility_score(model: str, file_type: str, c_active: int,
                           h_spec: float, referral: str) -> Tuple[float, Dict]:
    """U(m|s_t) = W1*R1 + W2*R2 + W3*R3 + W4*R4."""
    r1,r2,r3,r4 = _r1(model,file_type), _r2(model,c_active), _r3(model,referral), _r4(h_spec,c_active)
    score = float(np.clip(W1*r1 + W2*r2 + W3*r3 + W4*r4, 0.0, 1.0))
    return score, {"w1_R1": round(W1*r1,4), "w2_R2": round(W2*r2,4),
                   "w3_R3": round(W3*r3,4), "w4_R4": round(W4*r4,4)}


def heuristic_routing(file_type: str, telemetry: Dict, referral: str,
                      threshold: float = 0.50) -> Tuple[str, float, Dict]:
    """pi_top^Heur — returns (best_model, score, all_scores)."""
    scores = {}
    for m in ALL_MODELS:
        u, _ = compute_utility_score(m, file_type, telemetry["c_active"], telemetry["h_spec"], referral)
        scores[m] = u
    best = max(scores, key=lambda m: scores[m])
    return (M_NULL, scores[best], scores) if scores[best] < threshold else (best, scores[best], scores)


# ─────────────────────────────────────────────────────────────────────────────
# Sigmoid Lambda Gate
# ─────────────────────────────────────────────────────────────────────────────

def lambda_rag(alpha_ambig: float) -> float:
    """lambda(alpha) = 1 / (1 + exp(7*(alpha - 0.5)))  in [0,1]."""
    return float(1.0 / (1.0 + np.exp(SIGMOID_K * (alpha_ambig - SIGMOID_A0))))


# ─────────────────────────────────────────────────────────────────────────────
# Virtual CMO Diagnostic Uncertainty Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_delta_conflict(probs: Dict[str, float], epsilon: float = 1e-8) -> float:
    """delta_conflict = 1 - max(p_m) / (sum(p_m) + eps)  ~= 1 - max(p_m)."""
    if not probs: return 1.0
    v = np.array(list(probs.values()), dtype=float)
    return float(1.0 - v.max() / (v.sum() + epsilon))


# ─────────────────────────────────────────────────────────────────────────────
# AgenticDecisionSystem — Main Class
# ─────────────────────────────────────────────────────────────────────────────

class AgenticDecisionSystem:
    """
    Master Orchestration Agent  A_top = <S, A, K, pi_top, C_CMO>

    Routing policy:
        pi_top(a_t|s_t) = lambda(alpha)*P_RAG + (1-lambda(alpha))*P_Heur

    In offline/no-Bedrock mode use_rag=False: lambda forced to 0 (heuristic only).
    """

    def __init__(self, use_rag: bool = False):
        self.use_rag = use_rag
        logger.info(f"AgenticDecisionSystem | RAG={'ON' if use_rag else 'HEURISTIC-ONLY'}")

    def decide_model(self, data: np.ndarray, file_type: str,
                     referral_text: str = "", alpha_ambig: float = 0.50,
                     fs: float = 250.0) -> Dict[str, Any]:
        """Full A_top pipeline. Returns routing decision dict."""

        # Step 1 — Telemetry
        telem = extract_telemetry(data, fs=fs)
        logger.info(f"σ={telem['sigma']:.4f}  H_spec={telem['h_spec']:.3f}  C={telem['c_active']}  D={telem['duration']:.1f}s")

        # Step 2 — Quality gate Phi(t)
        phi = quality_gate(telem)
        if phi == 0:
            reason = (f"REJECTED by Phi(t)=0: sigma={telem['sigma']:.4f}<=0.01 "
                      f"or H_spec={telem['h_spec']:.3f}<0.22. gamma_guard=1.")
            logger.warning(reason)
            data_chars = {
                'shape': list(data.shape),
                'channels': int(telem['c_active']),
                'samples': int(data.shape[0]),
                'duration_estimate': float(telem['duration']),
                'signal_quality': 'low (rejected by Phi(t))'
            }
            return {"selected_model": M_NULL, "tau_route": 0.0, "gamma_guard": 1,
                    "telemetry": telem, "phi_gate": 0, "lambda_rag": lambda_rag(alpha_ambig),
                    "utility_scores": {}, "reasoning": reason, "confidence": 0.0, "all_scores": {},
                    "data_characteristics": data_chars}

        # Step 3 — Lambda mixing weight
        lam = lambda_rag(alpha_ambig)
        lam_h = 1.0 - lam

        # Step 4 — Heuristic branch
        h_model, h_score, utility_scores = heuristic_routing(file_type, telem, referral_text)

        # Step 5 — RAG branch (stub when Bedrock unavailable)
        if self.use_rag and lam > 0.05:
            r_model, r_conf = self._rag_stub(referral_text, telem)
        else:
            r_model, r_conf = h_model, h_score

        # Step 6 — Mixture decision
        if r_model == h_model:
            selected = r_model
            tau = float(lam * r_conf + lam_h * h_score)
        else:
            selected, tau = (r_model, r_conf) if lam >= lam_h else (h_model, h_score)

        gamma = 0
        if selected == M_NULL or tau < 0.50:
            selected, gamma = M_NULL, 1

        reasoning = (f"Routing->{selected.upper()} tau={tau:.3f} | "
                     f"sigma={telem['sigma']:.4f} H_spec={telem['h_spec']:.3f} "
                     f"C={telem['c_active']} | alpha={alpha_ambig:.2f} "
                     f"lam_RAG={lam:.3f} lam_Heur={lam_h:.3f} | "
                     f"U({selected})={utility_scores.get(selected,0):.4f}")

        data_chars = {
            'shape': list(data.shape),
            'channels': int(telem['c_active']),
            'samples': int(data.shape[0]),
            'duration_estimate': float(telem['duration']),
            'signal_quality': 'high' if telem['h_spec'] >= 0.35 and telem['sigma'] > 0.01 else 'medium'
        }
        return {"selected_model": selected, "tau_route": round(tau,4),
                "gamma_guard": gamma, "telemetry": telem, "phi_gate": phi,
                "lambda_rag": round(lam,4), "utility_scores": utility_scores,
                "reasoning": reasoning, "confidence": round(tau,4),
                "all_scores": {k: round(v,4) for k,v in utility_scores.items()},
                "data_characteristics": data_chars}

    def cmo_synthesis(self, predictions: Dict[str, Dict[str, float]],
                      telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Virtual CMO delta_conflict scoring and conflict detection."""
        per_model_conf, per_model_pred = {}, {}
        for mk, cp in predictions.items():
            if not cp: continue
            bc = max(cp, key=cp.get)
            per_model_conf[mk] = float(cp[bc])
            per_model_pred[mk] = bc
        delta = compute_delta_conflict(per_model_conf)
        gamma = 1 if (delta > DELTA_CONFLICT_THRESHOLD or telemetry.get("h_spec",1.0) < H_SPEC_MIN) else 0
        dom = max(per_model_conf, key=lambda m: per_model_conf[m]) if per_model_conf else M_NULL
        return {"delta_conflict": round(delta,4), "gamma_guard": gamma,
                "dominant_model": dom, "dominant_class": per_model_pred.get(dom,"Uncertain"),
                "confidence": round(per_model_conf.get(dom,0.0),4),
                "per_model_conf": {k:round(v,4) for k,v in per_model_conf.items()},
                "per_model_pred": per_model_pred}

    def _rag_stub(self, referral: str, telem: Dict) -> Tuple[str, float]:
        """Stub for f_Claude(e_q, G, t). Replace with Bedrock call in production."""
        scores = {m: _r3(m, referral) for m in ALL_MODELS}
        best = max(scores, key=lambda m: scores[m])
        return (best, scores[best]) if scores[best] > 0 else (M_HC, 0.5)

    # Legacy compatibility
    def get_model_recommendations(self, use_case: str) -> List[Dict[str, Any]]:
        return [{"model": m, "suitability": "high", "confidence_threshold": 0.01}
                for m, kws in CLINICAL_KEYWORDS.items()
                if any(use_case.lower() in kw for kw in kws)]