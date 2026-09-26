import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# Set page config
st.set_page_config(
    page_title="Agentic Disease Finder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports for logic
from models.model_manager import ModelManager
from models.agentic_decision import AgenticDecisionSystem
from utils.preprocessing import EEGPreprocessor
# from utils.visualization import VisualizationHelper # Assuming this exists or we mock it if needed

# --- Header & Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    /* Let Streamlit's config.toml handle global backgrounds and text colors */
    
    .glass-panel {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-panel:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Premium Button overrides */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        background-size: 200% auto;
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.6rem 1.5rem;
        transition: 0.5s;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        width: 100%;
        height: 3.5em;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton>button:hover {
        background-position: right center;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6);
        color: white;
    }
    /* Elegant Sidebar & Navigation Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    
    /* Hide the default radio button circles */
    div[role="radiogroup"] > label > div:first-of-type {
        display: none !important;
    }
    
    /* Style the radio labels as modern sidebar buttons */
    div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.03);
        padding: 12px 15px !important;
        border-radius: 12px !important;
        margin-bottom: 8px;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.05);
        cursor: pointer;
        width: 100%;
    }
    
    div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        transform: translateX(4px);
    }
    
    /* Highlight the actively selected sidebar tab */
    div[role="radiogroup"] > label[data-baseweb="radio"] input:checked + div {
        background: transparent !important;
    }
    
    div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.2), rgba(129, 140, 248, 0.2)) !important;
        border-left: 4px solid #38bdf8 !important;
        border-radius: 8px !important;
    }
    
    div[role="radiogroup"] p {
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        margin: 0 !important;
    }
    
    /* Custom metric card */
    .metric-card {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border-left: 4px solid #818cf8;
        height: 100%;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 8px;
    }
    
    /* Probability Bars */
    .prob-container {
        margin-top: 12px;
    }
    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .prob-bar-bg {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        height: 14px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .prob-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        border-radius: 6px;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem; padding-top: 1rem;">
    <h1 class='gradient-text' style='font-size: 3.5rem; margin-bottom: 0.5rem;'>🧠 Agentic Disease Finder</h1>
    <h3 style='color: #94a3b8; font-weight: 400; margin-top: 0; letter-spacing: 1px;'>Agent-Driven Multi-Modal Diagnostic Suite</h3>
</div>
<div class="glass-panel" style="margin-bottom: 2.5rem; padding: 24px; border-left: 4px solid #38bdf8; background: rgba(56, 189, 248, 0.05);">
    <p style="color: #f8fafc; font-size: 1.1rem; line-height: 1.7; margin: 0; font-weight: 300;">
        Welcome to the next generation of clinical analysis. This system utilizes an <strong>Agentic Architecture</strong> that functions as a virtual Chief Medical Officer. Rather than relying on a single diagnostic algorithm, the agent concurrently processes your raw clinical data across multiple specialized neural networks—ranging from Deep Sequence temporal modeling to Spatial Morphology CNNs—to synthesize a comprehensive, holistic medical report.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/color/96/000000/brain--v1.png", width=80)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to:", ["🩺 Clinical Diagnosis", "🧪 Sample Data Analysis", "📈 Model Performance", "ℹ️ System Info"])

# --- Helper Functions ---
@st.cache_resource
def load_system():
    model_manager = ModelManager()
    agentic_system = AgenticDecisionSystem()
    preprocessor = EEGPreprocessor()
    return model_manager, agentic_system, preprocessor

try:
    model_manager, agentic_system, preprocessor = load_system()
except Exception as e:
    st.error(f"Failed to load system components: {e}")
    st.stop()

def process_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    
    if name.endswith('.csv'):
        # Assume EEG data
        df = pd.read_csv(uploaded_file)
        if 'Time' in df.columns:
            df = df.drop(columns=['Time'])
        return df.values, 'csv'
        
    elif name.endswith('.txt'):
        try:
            data = np.loadtxt(uploaded_file)
            return data, 'txt'
        except Exception:
            return None, None
            
    elif name.endswith('.npy'):
        try:
            data = np.load(uploaded_file)
            return data, 'npy'
        except Exception:
            return None, None
            
    elif name.endswith(('.edf', '.bdf')):
        try:
            import mne
            import tempfile
            import os
            # MNE requires a file path
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Read EDF/BDF and extract numpy array
            raw = mne.io.read_raw(tmp_path, preload=True, verbose=False)
            data = raw.get_data().T  # MNE returns (channels, time) -> transpose to (time, channels)
            
            os.unlink(tmp_path)
            return data, 'edf' if name.endswith('.edf') else 'bdf'
        except Exception as e:
            st.error(f"Error parsing EDF/BDF: {e}")
            return None, None
            
    return None, None

def run_agentic_analysis(data, file_name, model_manager, agentic_system=None):
    with st.spinner("Agent is running multi-model clinical analysis..."):
        import time
        time.sleep(1) # simulate thinking
        
        file_ext = os.path.splitext(file_name)[1].lstrip('.').lower() if file_name else 'csv'
        context = {'filename': file_name} if file_name else {}
        
        # 1. Execute Agentic Routing Policy Decision
        if agentic_system is not None:
            decision = agentic_system.decide_model(data, file_ext, context)
            selected_model = decision.get('selected_model', 'nhrn_pd')
            agent_conf = decision.get('confidence', 0.85)
            reasoning_summary = decision.get('reasoning', ['Validated signal quality and duration metrics.'])
            
            st.markdown(f"""
            <div class="glass-panel" style="border-left: 4px solid #10b981; background: rgba(16, 185, 129, 0.08); margin-bottom: 24px;">
                <div style="color: #10b981; font-size: 0.85rem; text-transform: uppercase; font-weight: 800; letter-spacing: 1px; margin-bottom: 4px;">
                    🧠 MASTER ORCHESTRATION AGENT ROUTING POLICY CORE (A_top)
                </div>
                <div style="font-size: 1.4rem; font-weight: bold; color: #f8fafc; margin-bottom: 4px;">
                    Optimal Model Endpoint Selected: <span style="color: #34d399;">{selected_model.upper()}</span>
                </div>
                <div style="color: #a7f3d0; font-size: 1.05rem; font-weight: 600; margin-bottom: 10px;">
                    Agent Routing Confidence score τ_route = {agent_conf:.2%}
                </div>
                <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">
                    <b>Agent Reasoning Chain:</b><br>
                    • {'<br>• '.join(reasoning_summary[:3])}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Execute all models including MRI
        active_models = model_manager.get_available_models()
        results = {}
        
        st.markdown("<h3 class='gradient-text' style='margin-top: 20px;'>Multi-Modal Neural Ensemble Diagnostics</h3>", unsafe_allow_html=True)
        
        cols = st.columns(len(active_models))
        
        for idx, model_key in enumerate(active_models):
            try:
                prediction = model_manager.predict(data, model_key)
                results[model_key] = prediction
                
                pred_label = prediction['prediction']
                prob = prediction['probability']
                model_name = prediction.get('model_used', model_key)
                
                with cols[idx]:
                    st.markdown(f"""
                    <div class="glass-panel" style="padding: 16px; margin-bottom: 16px; border-top: 4px solid #818cf8; min-height: 250px;">
                        <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 8px;">{model_name}</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #f8fafc; margin-bottom: 4px;">{pred_label}</div>
                        <div style="color: #38bdf8; font-size: 1rem; font-weight: 600;">{prob:.2%} Confidence</div>
                        <div style='height: 10px;'></div>
                    """, unsafe_allow_html=True)
                    
                    if 'class_probabilities' in prediction:
                        prob_html = "<div class='prob-container'>"
                        for cls, p_val in prediction['class_probabilities'].items():
                            prob_html += f"<div style='margin-bottom: 8px;'><div class='prob-label' style='font-size: 0.75rem; margin-bottom: 2px;'><span>{cls}</span><span>{p_val:.1%}</span></div><div class='prob-bar-bg' style='height: 6px;'><div class='prob-bar-fill' style='width: {p_val*100}%;'></div></div></div>"
                        prob_html += "</div>"
                        st.markdown(prob_html, unsafe_allow_html=True)
                        
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                with cols[idx]:
                    st.error(f"Error in {model_key}: {str(e)}")
                    
        # Agentic Comprehensive Patient Analysis
        st.markdown("<h3 class='gradient-text'>🧠 Virtual CMO Synthesized Clinical Report</h3>", unsafe_allow_html=True)
        
        # Logic to synthesize results
        analysis_parts = []
        for model_key, res in results.items():
            pred_label = res.get('prediction', 'Unknown')
            prob = res.get('probability', 0)
            
            if pred_label == 'Error' or pred_label == 'Unknown':
                continue
                
            if model_key == 'bci2a_crdae':
                analysis_parts.append(f"**Motor Function (BCI):** Cortical mapping shows a {prob:.1%} correlation with **{pred_label}** activation patterns.")
            elif model_key == 'eeg_pd':
                if pred_label != "Healthy":
                    analysis_parts.append(f"**Neurological Biomarkers:** Detected potential **Parkinsonian resting-state oscillations** with {prob:.1%} confidence, requiring clinical follow-up.")
                else:
                    analysis_parts.append(f"**Neurological Biomarkers:** Basal ganglia oscillations are within **normal healthy ranges** ({prob:.1%} confidence).")
            elif model_key == 'neuroformer':
                desc = {"AD": "Alzheimer's Disease signatures", "CN": "Cognitively Normal temporal dynamics", "FTD": "Frontotemporal Dementia patterns"}
                analysis_parts.append(f"**Cognitive State:** Deep sequence modeling identified **{desc.get(pred_label, pred_label)}** with {prob:.1%} certainty.")
            elif model_key == 'brain_tumor_mri':
                analysis_parts.append(f"**Structural Morphology (MRI mapping):** By converting the signal into a spatial matrix, the spatial CNN detected **{pred_label}** morphological features ({prob:.1%} confidence).")
            elif model_key == 'spectra_sz':
                analysis_parts.append(f"**Psychiatric Evaluation:** SPECTRA multi-scale routing analysis detected **{pred_label}** cognitive state signatures ({prob:.1%} confidence).")
            elif model_key == 'nhrn_pd':
                if pred_label != "Healthy":
                    analysis_parts.append(f"**Neuromorphic Resonance Analysis:** The advanced NHRN model detected signs of **Parkinson's Disease** from dynamic cortical resonance patterns ({prob:.1%} confidence).")
                else:
                    analysis_parts.append(f"**Neuromorphic Resonance Analysis:** NHRN cortical resonance patterns are within **healthy baseline bounds** ({prob:.1%} confidence).")
                
        # Combine
        final_analysis = "Based on the comprehensive multi-modal agentic evaluation of the provided signal:<br><br>" + "<br><br>".join(analysis_parts)
        final_analysis += "<br><br><i style='color: #94a3b8;'>This is an AI-generated synthesis and does not substitute a formal medical diagnosis.</i>"
        
        st.markdown(f"""
        <div class="glass-panel" style="border-left: 4px solid #c084fc; background: rgba(139, 92, 246, 0.05);">
            <div style="color: #e2e8f0; font-size: 1.05rem; line-height: 1.6;">
                {final_analysis}
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Main App Logic ---

if app_mode == "🩺 Clinical Diagnosis":
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Upload Clinical Data")
        uploaded_file = st.file_uploader("Upload Signal Data (CSV, TXT, NPY, EDF, BDF)", type=['csv', 'txt', 'npy', 'edf', 'bdf'])
        
        data = None
        data_type = None
        file_name = None
        
        if uploaded_file:
            data, data_type = process_uploaded_file(uploaded_file)
            st.success(f"Loaded {data_type.upper()} File: {uploaded_file.name}")
            file_name = uploaded_file.name
            
    with col2:
        st.subheader("2. Data Visualization")
        if data is not None:
            if data_type == 'csv':
                st.line_chart(data[:, :3]) # Show first 3 channels
                st.caption("Displaying first 3 channels")
        else:
            st.info("Upload data to see visualization.")

    # Analysis Section
    if data is not None:
        st.markdown("---")
        st.subheader("3. Agentic Analysis")
        
        if st.button("Start Diagnosis"):
            run_agentic_analysis(data, file_name, model_manager, agentic_system)

elif app_mode == "🧪 Sample Data Analysis":
    st.markdown("---")
    st.subheader("Simulate Clinical Diagnosis with Sample Data")
    st.write("Select one of the pre-generated synthetic datasets below to test the multi-modal pipeline without needing to upload your own files.")
    
    import os
    sample_files = []
    if os.path.exists("sample_data"):
        sample_files = [f for f in os.listdir("sample_data") if f.endswith(('.csv', '.npy'))]
        
    if not sample_files:
        st.warning("No sample data found! Please run `generate_test_data.py` first.")
    else:
        selected_file = st.selectbox("Select Sample Dataset:", sample_files)
        
        data = None
        data_type = None
        if selected_file:
            file_path = os.path.join("sample_data", selected_file)
            if selected_file.endswith('.csv'):
                df = pd.read_csv(file_path)
                data = df.values
                data_type = 'csv'
            elif selected_file.endswith('.npy'):
                data = np.load(file_path)
                data_type = 'npy'
                
            st.success(f"Loaded {selected_file}")
            
            st.markdown("---")
            if st.button("Run Multi-Modal Analysis on Sample"):
                run_agentic_analysis(data, selected_file, model_manager, agentic_system)

elif app_mode == "📈 Model Performance":
    st.header("Agentic Model Performance & Benchmarks")
    
    st.markdown("### 🏛️ System Architecture & Orchestration Flow")
    if os.path.exists("springer_paper/top_agent_block_diagram.png"):
        st.image("springer_paper/top_agent_block_diagram.png", caption="Master Orchestration Agent Policy Architecture & Dynamic Pipeline", use_column_width=True)
    
    st.markdown("---")
    st.markdown("### 📊 Multi-Domain Confusion Matrices & Evaluation Metrics")
    
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Unified 3-Domain Confusion Matrix**")
        if os.path.exists("springer_paper/unified_confusion_matrix.png"):
            st.image("springer_paper/unified_confusion_matrix.png", use_column_width=True)
        elif os.path.exists("outputs/bci2a_confusion_matrix_enhanced.png"):
            st.image("outputs/bci2a_confusion_matrix_enhanced.png", use_column_width=True)
            
    with cols[1]:
        st.markdown("**Master Agent Score Analysis**")
        if os.path.exists("springer_paper/top_agent_score_analysis.png"):
            st.image("springer_paper/top_agent_score_analysis.png", use_column_width=True)
        elif os.path.exists("outputs/eeg_pd_confusion_matrix_enhanced.png"):
            st.image("outputs/eeg_pd_confusion_matrix_enhanced.png", use_column_width=True)
            
    cols2 = st.columns(2)
    with cols2[0]:
        st.markdown("**Classification Performance Across Disorders**")
        if os.path.exists("springer_paper/performance_comparison.png"):
            st.image("springer_paper/performance_comparison.png", use_column_width=True)
            
    with cols2[1]:
        st.markdown("**Transaction Execution Latency Comparison**")
        if os.path.exists("springer_paper/latency_comparison.png"):
            st.image("springer_paper/latency_comparison.png", use_column_width=True)
            
    st.markdown("---")
    st.markdown("### 📝 Clinical Research Evaluation Summary")
    if os.path.exists("outputs/PAPER_ANALYSIS_SECTION.md"):
        with open("outputs/PAPER_ANALYSIS_SECTION.md", "r") as f:
            analysis_text = f.read()
            st.markdown(analysis_text)
    elif os.path.exists("AGENTIC_SYSTEM_EVALUATION_REPORT.md"):
        with open("AGENTIC_SYSTEM_EVALUATION_REPORT.md", "r") as f:
            st.markdown(f.read())

elif app_mode == "ℹ️ System Info":
    st.markdown("<h2 class='gradient-text'>System Architecture</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-panel" style="margin-bottom: 20px; border-left: 4px solid #38bdf8;">
        <h4 style="color: #f8fafc; margin-top: 0;">Multi-Modal Agentic Framework</h4>
        <p style="color: #cbd5e1; line-height: 1.6;">
            The <b>Agentic Disease Finder</b> represents a paradigm shift from traditional single-model classifiers. 
            Instead of forcing the user to manually select the correct model for their data, the system utilizes an overarching 
            <b>Virtual Chief Medical Officer (CMO)</b> agent. When raw data (in any format: EDF, BDF, NPY, CSV, TXT) is ingested, 
            the Agent concurrently normalizes the signal into both 1D temporal sequences and 2D spatial matrices, routing it through all available specialized neural networks simultaneously.
        </p>
    </div>
    
    <div class="glass-panel" style="margin-bottom: 20px; border-left: 4px solid #818cf8;">
        <h4 style="color: #f8fafc; margin-top: 0;">Active Clinical Neural Networks</h4>
        <ul style="color: #cbd5e1; line-height: 1.8;">
            <li>🧠 <b>Neuroformer (Alzheimer's/FTD):</b> A deep temporal sequence transformer designed to identify subtle cognitive decline signatures over long-range EEG epochs.</li>
            <li>🌊 <b>Parkinson's Detector (EEG-PD):</b> An oscillatory frequency analyzer screening resting-state signals for classic Parkinsonian theta/beta anomalies.</li>
            <li>⚡ <b>Motor Function (BCI2A):</b> A spatial-temporal mapping network for decoding cortical motor imagery intents (Left/Right Hand, Foot, Tongue).</li>
            <li>🖼️ <b>Structural Morphology (Brain MRI):</b> An open-source <i>MobileNetV2</i> transfer-learning CNN that analyzes spatial matrices to identify intracranial tumors (Glioma, Meningioma, Pituitary).</li>
            <li>🧬 <b>SPECTRA (Schizophrenia):</b> A multi-scale cognitive task routing architecture that detects complex psychiatric signatures in frequency and spatial domains.</li>
            <li>🌀 <b>NHRN (Parkinson's Disease):</b> A Neuromorphic Hierarchical Resonance Network that evaluates complex cortical resonance patterns across 10 specialized neural levels.</li>
        </ul>
    </div>
    
    <div class="glass-panel" style="border-left: 4px solid #c084fc;">
        <h4 style="color: #f8fafc; margin-top: 0;">Technology Stack</h4>
        <p style="color: #cbd5e1; line-height: 1.6; margin-bottom: 0;">
            <b>Core UI/UX:</b> Streamlit, Custom Glassmorphism CSS<br>
            <b>Data Ingestion:</b> MNE-Python (EDF/BDF), NumPy, Pandas<br>
            <b>Deep Learning Backends:</b> TensorFlow 2.x, Keras (MobileNetV2), PyTorch<br>
            <b>Inference Mode:</b> CPU (TensorFlow-DirectML / WSL2 compatible for GPU Acceleration)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #f8fafc; margin-top: 30px; margin-bottom: 15px;'>Live Model Status Monitor</h3>", unsafe_allow_html=True)
    status = model_manager.get_model_status()
    
    cols = st.columns(4)
    for idx, (model_name, is_ready) in enumerate(status.items()):
        color = "#10b981" if is_ready else "#ef4444"
        icon = "✅ ONLINE" if is_ready else "❌ OFFLINE"
        with cols[idx % 4]:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; text-align: center; border-bottom: 3px solid {color}; border-top: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; margin-bottom: 8px;">{model_name}</div>
                <div style="color: {color}; font-weight: 800; letter-spacing: 1px;">{icon}</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Agentic Disease Finder v1.0 | Research Prototype")
