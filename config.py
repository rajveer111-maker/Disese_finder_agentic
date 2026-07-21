"""
Configuration file for Agentic Disease Finder
"""

# Model configuration
MODEL_CONFIG = {
    'bci2a_crdae': {
        'name': 'BCI2A CRDAE (Motor Imagery)',
        'input_shape': (1000, 22),
        'output_classes': ['Left Hand', 'Right Hand', 'Foot', 'Tongue'],
        'description': 'Motor imagery classification for BCI applications',
        'confidence_threshold': 0.01,
        'file_path': 'models/bci2a_crdae_gtaa_best_weights.weights.h5'
    },
    'eeg_pd': {
        'name': 'EEG Parkinson\'s Disease Detection',
        'input_shape': (40, 1024),
        'output_classes': ['Healthy', 'Parkinson\'s Disease'],
        'description': 'Parkinson\'s disease detection from EEG signals',
        'confidence_threshold': 0.01,
        'file_path': 'models/Parkinson_Model.py'
    },
    'brain_tumor_mri': {
        'name': 'Adaptive Multi-Scale Fusion (Brain MRI)',
        'input_shape': (128, 128, 1),
        'output_classes': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
        'description': 'Brain tumor classification using adaptive multi-scale feature fusion on MRI scans',
        'confidence_threshold': 0.01,
        'file_path': 'models/adaptive_multi_scale_fusion_network.h5'
    },
    'neuroformer': {
        'name': 'Neuroformer (Alzheimer\'s Disease)',
        'input_shape': (19, 2500),
        'output_classes': ['AD', 'CN', 'FTD'],
        'description': 'Alzheimer\'s disease classification from EEG',
        'confidence_threshold': 0.01,
        'file_path': 'models/neuroformer/best_model.h5'
    },
    'spectra_sz': {
        'name': 'SPECTRA-SZ (Schizophrenia)',
        'input_shape': (19, 1024),
        'output_classes': ['Healthy', 'Schizophrenia'],
        'description': 'Schizophrenia Prediction via Encoded Cognitive Task Routing Architecture',
        'confidence_threshold': 0.01,
        'file_path': 'models/SPECTRA-SZ/spectra_run/best_model.pt'
    },
    'nhrn_pd': {
        'name': 'NHRN Parkinson\'s Disease Detection',
        'input_shape': (40, 1024),
        'output_classes': ['Healthy', 'Parkinson\'s Disease'],
        'description': 'Advanced Neuromorphic Hierarchical Resonance Network for Parkinson\'s Detection',
        'confidence_threshold': 0.01,
        'file_path': 'models/best_nhrn_model.h5'
    }
}

# Data processing configuration
DATA_CONFIG = {
    'sampling_rate': 250,  # Hz
    'standard_channels': 22,
    'filter_freqs': [1.0, 40.0],  # Bandpass filter frequencies
    'normalize': True,
    'remove_dc': True
}

# UI configuration
UI_CONFIG = {
    'page_title': 'Agentic Disease Finder',
    'page_icon': '🧠',
    'layout': 'wide',
    'theme': 'light',
    'color_primary': '#667eea',
    'color_secondary': '#764ba2'
}

# Visualization configuration
VIZ_CONFIG = {
    'max_channels_display': 10,
    'max_spectrum_channels': 6,
    'figure_height': 400,
    'color_palette': 'Set3'
}

# File upload configuration
UPLOAD_CONFIG = {
    'max_file_size': 50 * 1024 * 1024,  # 50MB
    'allowed_extensions': {
        'eeg': ['csv', 'txt', 'edf'],
        'image': ['png', 'jpg', 'jpeg']
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'agentic_disease_finder.log'
}

