# ANDI: Agentic Neurological Disorder Identifier

An open-source research implementation of an agentic decision-orchestration and multi-disorder diagnostic routing architecture for electroencephalography (EEG) analysis.

---

## 🔬 Overview & Key Concept

ANDI implements a multi-tier agentic architecture designed to autonomously ingest raw multichannel EEG recordings, evaluate signal integrity via spectral telemetry, and dynamically select optimal neural classifiers using clinical guideline vector retrieval and LLM reasoning.

> **Core Formulation:** A Master Orchestration Agent ($\mathcal{A}_{\text{top}}$) running on Amazon Bedrock selects the optimal SageMaker classifier by evaluating EEG telemetry, an automated spectral-entropy quality gate, and the semantic similarity between the physician's referral note and clinical guidelines retrieved via a vector search engine.

$$\mathcal{A}_{\text{top}} = \langle \mathcal{S}, \mathcal{A}, \mathcal{K}, \pi_{\text{top}}, \mathcal{C}_{\text{CMO}} \rangle$$

```
                         Incoming Multichannel EEG Record
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │    Signal Telemetry Extraction    │
                      │  Duration, Channels, Noise σ, PSD │
                      │   t = [D, σ, n̂, H_spec, C_active] │
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
     │  Branch A: Bedrock RAG Policy      │    │    γ_guard = 1            │
     │    (AAN, IWG, WHO Guidelines)      │    │    Action = M_null        │
     │  Branch B: Normalized Heuristic    │    │    Status: REJECTED       │
     │    (Format, Bounds, Keywords, PSD) │    └───────────────────────────┘
     └─────────────────┬──────────────────┘
                       │  Continuous Gate: λ(α_ambig)
                       ▼
        ┌────────────────────────────────────────────────────────┐
        │       Specialized SageMaker Classifier Endpoints       │
        ├─────────────────┬─────────────────┬────────────────────┤
        │ NHRN-PD         │ Neuroformer     │ SPECTRA-SZ         │
        │ Parkinson's     │ Alzheimer's     │ Schizophrenia      │
        │ 40×1024 / 22-ch │ 19×2500 Leads   │ 19×1024 Leads      │
        │ Beta Resonance  │ Synaptic Decoup │ Gamma Phase-Lock   │
        └─────────────────┴────────┬────────┴────────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │        Virtual CMO Meta-Governance Consensus           │
        │        Δ_conflict Resolution & Safety Guardrails       │
        └────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Formulation

### 1. State Space & Continuous Telemetry $\mathcal{S}$
The observation state vector $\mathbf{s}_t = [\mathbf{t}^\top, \mathbf{e}_q^\top, \alpha_{\text{ambig}}]^\top$ integrates:
* Continuous telemetry $\mathbf{t} = [D, \sigma, \hat{n}, H_{\text{spec}}, C_{\text{active}}]^\top$:
  * $D$: Recording duration in seconds.
  * $\sigma$: Amplitude standard deviation across channels.
  * $\hat{n}$: Noise floor level in $\mu\text{V}$.
  * $H_{\text{spec}}$: Welch broadband spectral entropy ($-\frac{1}{\ln K}\sum p_k \ln p_k$).
  * $C_{\text{active}}$: Active electrode channel count ($8, 16, 19, 22, 32, 64$).
* $\mathbf{e}_q$: Query embedding of physician referral note.
* $\alpha_{\text{ambig}} \in [0, 1]$: Referral semantic ambiguity index.

### 2. Spectral Quality Gate $\Phi(\mathbf{t})$
Signals corrupted by electrode detachments, ground disconnection, flatlines, or heavy powerline interference are automatically intercepted prior to deep model inference:
$$\Phi(\mathbf{t}) = \mathbb{I}(\sigma > 0.01\,\mu\text{V}) \cdot \mathbb{I}(H_{\text{spec}} \ge 0.22)$$
* When $\Phi(\mathbf{t}) = 0$, the recording is routed to $\emptyset_{\text{unroutable}}$ and the safety flag $\gamma_{\text{guard}} = 1$ is activated.

### 3. Vector Search Engine over Clinical Guidelines $\mathcal{K}$
The knowledge base $\mathcal{K}$ indexes consensus clinical criteria:
* **AAN Practice Guidelines (PD):** Basal ganglia sensorimotor resting-state beta-band synchrony and resonance (13–30 Hz).
* **IWG & NIA-AA Criteria (AD):** Progressive episodic memory impairment, temporal-parietal spectral slowing, theta elevation, and functional synaptic decoupling.
* **WHO & APA DSM-5 Criteria (SZ):** Cognitive dysfunction and task-induced gamma-band phase-locking coherence deficits (30–80 Hz).

*Vector engine connects to Pinecone in production or utilizes the built-in normalized cosine similarity engine for local offline execution.*

### 4. Dual-Branch Soft Routing Policy $\pi_{\text{top}}$
Continuous arbitration between Branch A (Bedrock semantic RAG policy) and Branch B (normalized heuristic engine) is governed by a logistic sigmoid transition:
$$\lambda(\alpha_{\text{ambig}}) = \frac{1}{1 + \exp\left(7(\alpha_{\text{ambig}} - 0.50)\right)}$$

The fallback heuristic engine evaluates the 4-part normalized Policy Utility score:
$$U(m \mid \mathbf{s}_t) = \sum_{j=1}^4 w_j R_j(m, \mathbf{s}_t)$$
with fixed clinical weights $\mathbf{w} = [0.25, 0.25, 0.30, 0.20]^\top$ ($\sum w_j = 1.0$):
* **$R_1(m, \mathbf{s}_t) \in [0, 1]$**: Format support (1.0 for EDF/BDF/CSV/NPY, 0.8 for TXT).
* **$R_2(m, \mathbf{s}_t) = \min(1.0, C_{\text{active}} / C_{\text{target}}^{(m)})$**: Channel dimension compliance.
* **$R_3(m, \mathbf{s}_t) \in [0, 1]$**: Semantic clinical terminology keyword matching.
* **$R_4(H_{\text{spec}}, C_{\text{active}}) = \min(1.0, 0.60 H_{\text{spec}} + 0.40(C_{\text{active}} / 64))$**: Lead density & spectral complexity.

### 5. Virtual CMO Consensus & Conflict Metric $\mathcal{C}_{\text{CMO}}$
Predictive divergence across the ensemble is quantified by:
$$\Delta_{\text{conflict}} = 1 - \frac{\max_{m} p_m}{\sum_{m=1}^{M} p_m + \epsilon}$$
When $\Delta_{\text{conflict}} > 0.40$ or $H_{\text{spec}} < 0.22$, the safety guardrail flag $\gamma_{\text{guard}} = 1$ is activated, escalating the case to clinician review.

---

## 🧠 Specialized Diagnostic Classifiers

| Model | Target Pathology | Key Neural Mechanism | Nominal Input | Threshold |
|---|---|---|---|---|
| **NHRN-PD** | Parkinson's Disease | Neuromorphic Hierarchical Resonance Network; basal ganglia beta-band coupling | 40 × 1024 (22 leads active) | 0.01 |
| **Neuroformer** | Alzheimer's Disease | Temporal sequence self-attention modeling synaptic decoupling | 19 × 2500 | 0.01 |
| **SPECTRA-SZ** | Schizophrenia | Cognitive task phase-locking in task-induced gamma oscillations | 19 × 1024 | 0.01 |
| **Healthy Baseline** | Normal Control | Regular posterior dominant alpha rhythm with stable reactivity | 19 × 2500 | 0.01 |

---

## 💻 Codebase Structure

```
├── models/
│   ├── agentic_decision.py   # Master Agent A_top, Phi(t) gate, Bedrock & Vector RAG, CMO
│   ├── model_manager.py      # Unified model loader (Keras h5 & PyTorch pt)
│   ├── nhrn_model.py         # NHRN neuromorphic architecture layers
│   ├── custom_layers.py      # MultiScaleFeatureFusion, SqueezeExcitation
│   └── simple_model_loader.py# Lightweight model wrapper
├── deployment/
│   ├── deploy_sagemaker_endpoints.py # AWS SageMaker model endpoints deployment
│   ├── deploy_ecs.py                 # AWS ECS/Fargate container orchestration
│   └── check_models.py               # Endpoint verification script
├── api.py                    # FastAPI server exposing /api/diagnose with telemetry
├── config.py                 # Model parameters, thresholds, and data configurations
└── requirements.txt          # Python dependencies
```

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/rajveer111-maker/Disese_finder_agentic.git
cd Disese_finder_agentic

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Diagnostic Server
```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Diagnostic Ingestion Example (Python)
```python
import numpy as np
from models.agentic_decision import AgenticDecisionSystem

# Initialize Master Agent (supports both local vector engine and AWS Bedrock)
agent = AgenticDecisionSystem()

# Synthetic 19-channel EEG epoch (10 seconds @ 250 Hz)
eeg_epoch = np.random.randn(2500, 19)

# Execute agentic decision pipeline
decision = agent.decide_model(
    data=eeg_epoch,
    file_type="edf",
    referral_text="Elderly female presenting with progressive memory decline and spatial disorientation",
    alpha_ambig=0.30
)

print(f"Selected Model: {decision['selected_model']}")
print(f"Routing Confidence (tau): {decision['tau_route']}")
print(f"Quality Gate Phi(t): {decision['phi_gate']}")
print(f"Safety Guard Flag gamma_guard: {decision['gamma_guard']}")
print(f"Broadband Spectral Entropy H_spec: {decision['telemetry']['h_spec']:.4f}")
print(f"Reasoning: {decision['reasoning']}")
```

### 4. Running With Cloud Integration (AWS Bedrock & Pinecone)
To enable live Amazon Bedrock LLM routing and Pinecone vector search, configure your environment variables:
```bash
export AWS_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export BEDROCK_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
export PINECONE_API_KEY="your_pinecone_key"
export PINECONE_INDEX_NAME="andi-clinical-guidelines"
```
*(If cloud credentials are not configured, the system automatically runs the local high-precision vector similarity engine and algorithmic heuristic policy).*

---

## 📄 Peer Review & Anonymity Notice

This codebase accompanies a manuscript currently under formal peer review. Bibliographic details, citations, and benchmark checkpoints will be released upon completion of peer review.
