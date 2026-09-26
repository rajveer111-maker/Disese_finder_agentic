# ANDI: Agentic Neurological Disorder Identifier

Official implementation and reference architecture for **ANDI: Agentic Neurological Disorder Identifier**, submitted to *IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)*.

---

## 🔬 System Architecture

ANDI introduces a multi-tier agentic decision architecture for autonomous, robust routing and diagnosis of neurological disorders from resting-state and task-induced electroencephalography (EEG):

$$\mathcal{A}_{\text{top}} = \langle \mathcal{S}, \mathcal{A}, \mathcal{K}, \pi_{\text{top}}, \mathcal{C}_{\text{CMO}} \rangle$$

```
                         Incoming Multichannel EEG Record
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │    Signal Telemetry Extraction    │
                      │  Duration, Channels, Noise σ, PSD │
                      └─────────────────┬─────────────────┘
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │    Spectral Quality Gate Φ(t)     │
                      │   σ > 0.01 μV  and  H_spec ≥ 0.22 │
                      └───────┬───────────────────┬───────┘
                     Passed   │                   │ Rejected
                              ▼                   ▼
     ┌────────────────────────────────────┐    ┌───────────────────────────┐
     │ Dual-Branch Soft Routing Policy    │    │ Out-of-Distribution Guard │
     │  Branch A: Clinical RAG Policy     │    │    γ_guard = 1            │
     │  Branch B: Normalized Heuristic    │    │    Action = M_null        │
     └─────────────────┬──────────────────┘    └───────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Specialized Neural Engines  │
        ├──────────────┬───────────────┤
        │ NHRN-PD      │ Parkinson's   │
        │ Neuroformer  │ Alzheimer's   │
        │ SPECTRA-SZ   │ Schizophrenia │
        │ Baseline     │ Healthy Ctrl  │
        └──────────────┴───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Virtual CMO Consensus      │
        │   Δ_conflict Evaluation      │
        └──────────────────────────────┘
```

---

## 📐 Mathematical Formulation

### 1. Spectral Quality Gate $\Phi(\mathbf{t})$
Degraded or ungrounded recordings are filtered prior to deep inference:
$$\Phi(\mathbf{t}) = \mathbb{I}(\sigma > 0.01\,\mu\text{V}) \cdot \mathbb{I}(H_{\text{spec}} \ge 0.22)$$
where $H_{\text{spec}} = -\frac{1}{\ln K}\sum_{k=1}^K p_k \ln p_k$ represents broadband spectral entropy derived from Welch power spectral density.

### 2. Dual-Branch Soft Routing Policy $\pi_{\text{top}}$
Continuous arbitration between the clinical RAG policy ($\pi_{\text{top}}^{\text{RAG}}$) and normalized heuristic engine ($\pi_{\text{top}}^{\text{Heur}}$) is modulated by referral note ambiguity $\alpha_{\text{ambig}} \in [0, 1]$:
$$\lambda(\alpha_{\text{ambig}}) = \frac{1}{1 + \exp\left(7(\alpha_{\text{ambig}} - 0.50)\right)}$$

The heuristic utility score $U(m \mid \mathbf{s}_t)$ evaluates candidate models across four calibrated rules:
$$U(m \mid \mathbf{s}_t) = \sum_{j=1}^4 w_j R_j(m, \mathbf{s}_t)$$
with fixed clinical hyperparameter weights $\mathbf{w} = [0.25, 0.25, 0.30, 0.20]^\top$:
* **$R_1(m, \mathbf{s}_t) \in [0, 1]$**: Format support (1.0 for EDF/BDF/CSV/NPY, 0.8 for TXT).
* **$R_2(m, \mathbf{s}_t) = \min(1.0, C_{\text{active}} / C_{\text{target}}^{(m)})$**: Lead dimension compliance ($C_{\text{target}} \in \{19, 22\}$).
* **$R_3(m, \mathbf{s}_t) \in [0, 1]$**: Semantic clinical keyword and diagnosis matching.
* **$R_4(H_{\text{spec}}, C_{\text{active}}) = \min(1.0, 0.60 H_{\text{spec}} + 0.40(C_{\text{active}} / 64))$**: Spectral complexity and electrode density.

### 3. Virtual CMO Disagreement Resolution $\mathcal{C}_{\text{CMO}}$
Inter-classifier predictive divergence across the ensemble is quantified by:
$$\Delta_{\text{conflict}} = \max_{m} p(y \mid \mathbf{x}, m) - \min_{m} p(y \mid \mathbf{x}, m)$$
When $\Delta_{\text{conflict}} > 0.40$ or $H_{\text{spec}} < 0.35$, the safety guardrail flag $\gamma_{\text{guard}} = 1$ is activated, prompting clinical review.

---

## 🧠 Specialized Diagnostic Engines

| Model | Target Disorder | Key Neural Mechanism | Nominal Input |
|---|---|---|---|
| **NHRN-PD** | Parkinson's Disease | Neuromorphic Hierarchical Resonance Network; basal ganglia beta-band coupling | 40 × 1024 / 22-ch |
| **Neuroformer** | Alzheimer's Disease | Temporal sequence self-attention modeling synaptic decoupling | 19 × 2500 |
| **SPECTRA-SZ** | Schizophrenia | Cognitive task phase-locking in task-induced gamma oscillations | 19 × 1024 |
| **Healthy Control** | Baseline Physiology | Broadband spectral regularity and stable background rhythms | 19 × 2500 |

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/rajveer111-maker/Disese_finder_agentic.git
cd Disese_finder_agentic

# Install dependencies
pip install -r requirements.txt
```

### Running the API Server
```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

### Diagnostic Ingestion Example (Python)
```python
import numpy as np
from models.agentic_decision import AgenticDecisionSystem

# Initialize master agent
agent = AgenticDecisionSystem()

# Synthetic 19-channel EEG epoch (10 seconds @ 250 Hz)
eeg_epoch = np.random.randn(2500, 19)

# Execute agentic routing
decision = agent.decide_model(
    data=eeg_epoch,
    file_type="edf",
    referral_text="Resting tremor, suspected basal ganglia pathology",
    alpha_ambig=0.30
)

print(f"Selected Model: {decision['selected_model']}")
print(f"Confidence (tau): {decision['tau_route']}")
print(f"Quality Gate Phi(t): {decision['phi_gate']}")
print(f"Guardrail Flag gamma_guard: {decision['gamma_guard']}")
print(f"Broadband Spectral Entropy H_spec: {decision['telemetry']['h_spec']:.4f}")
```

---

## 📄 IEEE Reference & Citation

If you use this codebase or architecture in your research, please cite:
```bibtex
@article{andi_ieee_tnsre_2026,
  title={Autonomous Ingestion, Dynamic Routing, and Clinical Consensus in Multi-Disorder Neurological EEG Analysis: An Agentic System},
  author={ANDI Research Consortium},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  year={2026},
  note={Under Peer Review}
}
```
