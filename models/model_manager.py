import tensorflow as tf
import numpy as np
import os
from typing import Dict, Any, Optional
import logging
from .simple_model_loader import SimpleModelLoader
from .custom_layers import (
    DepthwiseSeparableConvBlock,
    MultiScaleFeatureFusion,
    SqueezeExcitation,
)
from .nhrn_model import (
    FractalConvolutionBlock,
    SynapticGateMechanism,
    NeuralOscillationExtractor,
    AdaptiveReceptiveFieldModule,
    TemporalBifurcationNetwork,
    CognitiveLoadBalancer,
    MemoryConsolidationGate,
    SpectralTemporalBridge,
    EmergentPatternSynthesizer,
    HierarchicalFeatureCrystallizer,
)
from config import MODEL_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages loading and inference for different disease detection models.
    """
    
    def __init__(self):
        self.models = {}
        self.model_paths = {
            'bci2a_crdae': 'models/bci2a_crdae_gtaa_best_weights.weights.h5',
            'eeg_pd': 'models/best_eeg_pd_model.h5',
            'brain_tumor_mri': 'models/adaptive_multi_scale_fusion_network.h5',
            'neuroformer': 'models/neuroformer/best_model.h5',
            'spectra_sz': 'models/SPECTRA-SZ/spectra_run/best_model.pt',
            'nhrn_pd': 'models/best_nhrn_model.h5',
        }
        self.model_info = {
            'bci2a_crdae': {
                'name': 'BCI2A CRDAE (Motor Imagery)',
                'input_shape': (1000, 22),  # 1000 samples, 22 channels
                'output_classes': ['Left Hand', 'Right Hand', 'Foot', 'Tongue'],
                'description': 'Motor imagery classification for BCI applications'
            },
            'eeg_pd': {
                'name': 'EEG Parkinson\'s Disease Detection',
                'input_shape': (1000, 22),  # 1000 samples, 22 channels
                'output_classes': ['Healthy', 'Parkinson\'s Disease'],
                'description': 'Parkinson\'s disease detection from EEG signals'
            },
            'brain_tumor_mri': {
                'name': 'Adaptive Multi-Scale Fusion (Brain MRI)',
                'input_shape': (128, 128, 1),
                'output_classes': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
                'description': 'Brain tumor classification using adaptive multi-scale feature fusion on MRI scans'
            },
            'neuroformer': {
                'name': 'Neuroformer (Alzheimer\'s Disease)',
                'input_shape': (19, 2500),  # 19 channels, 2500 sequence length
                'output_classes': ['AD', 'CN', 'FTD'],
                'description': 'Alzheimer\'s disease classification using Neural Oscillatory networks from EEG'
            },
            'spectra_sz': {
                'name': 'SPECTRA-SZ (Schizophrenia)',
                'input_shape': (19, 1024),
                'output_classes': ['Healthy', 'Schizophrenia'],
                'description': 'Schizophrenia prediction via encoded cognitive task routing architecture'
            },
            'nhrn_pd': {
                'name': 'NHRN Parkinson\'s Disease Detection',
                'input_shape': (40, 1024),
                'output_classes': ['Healthy', 'Parkinson\'s Disease'],
                'description': 'Advanced Neuromorphic Hierarchical Resonance Network for Parkinson\'s Detection'
            },
        }
        self.simple_loader = SimpleModelLoader()
        self.custom_objects = {
            'DepthwiseSeparableConvBlock': DepthwiseSeparableConvBlock,
            'MultiScaleFeatureFusion': MultiScaleFeatureFusion,
            'SqueezeExcitation': SqueezeExcitation,
            'FractalConvolutionBlock': FractalConvolutionBlock,
            'SynapticGateMechanism': SynapticGateMechanism,
            'NeuralOscillationExtractor': NeuralOscillationExtractor,
            'AdaptiveReceptiveFieldModule': AdaptiveReceptiveFieldModule,
            'TemporalBifurcationNetwork': TemporalBifurcationNetwork,
            'CognitiveLoadBalancer': CognitiveLoadBalancer,
            'MemoryConsolidationGate': MemoryConsolidationGate,
            'SpectralTemporalBridge': SpectralTemporalBridge,
            'EmergentPatternSynthesizer': EmergentPatternSynthesizer,
            'HierarchicalFeatureCrystallizer': HierarchicalFeatureCrystallizer,
        }
        self._load_models()
    
    def _load_models(self):
        """Load all available models."""
        for model_key, model_path in self.model_paths.items():
            try:
                if os.path.exists(model_path):
                    if model_key == 'neuroformer':
                        try:
                            import torch
                            from .neuroformer.neuroformer_infer import load_model_h5
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            self.models[model_key] = load_model_h5(model_path, device)
                            logger.info(f"✅ Loaded {model_key} model successfully")
                        except Exception as e:
                            logger.error(f"❌ Error loading {model_key}: {e}")
                        continue
                        
                    if model_key == 'spectra_sz':
                        try:
                            import sys
                            # Add SPECTRA-SZ dir to path temporarily so it can import its modules if needed
                            sys.path.insert(0, os.path.abspath('models/SPECTRA-SZ'))
                            from spectra_sz import SPECTRA
                            import torch
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            model = SPECTRA(n_classes=2)
                            checkpoint = torch.load(model_path, map_location=device)
                            if 'model_state_dict' in checkpoint:
                                model.load_state_dict(checkpoint['model_state_dict'])
                            elif 'model_state' in checkpoint:
                                model.load_state_dict(checkpoint['model_state'])
                            else:
                                model.load_state_dict(checkpoint)
                            model.to(device)
                            model.eval()
                            self.models[model_key] = model
                            sys.path.pop(0)
                            logger.info(f"✅ Loaded {model_key} model successfully")
                        except Exception as e:
                            logger.error(f"❌ Error loading {model_key}: {e}")
                            if 'sys' in locals():
                                sys.path.pop(0)
                        continue

                    # Try to load as weights first, then as full model
                    try:
                        # For weights file, we need to reconstruct the model architecture
                        model = self._reconstruct_model(model_key)
                        model.load_weights(model_path)
                        self.models[model_key] = model
                        logger.info(f"✅ Loaded {model_key} model successfully")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load {model_key} as weights: {e}")
                        # Try loading as full model
                        try:
                            model = tf.keras.models.load_model(
                                model_path,
                                custom_objects=self.custom_objects
                            )
                            self.models[model_key] = model
                            logger.info(f"✅ Loaded {model_key} as full model")
                        except Exception as e2:
                            logger.error(f"❌ Failed to load {model_key}: {e2}")
                else:
                    logger.warning(f"⚠️ Model file not found: {model_path}")
            except Exception as e:
                logger.error(f"❌ Error loading {model_key}: {e}")
    
    def _reconstruct_model(self, model_key: str) -> tf.keras.Model:
        """Reconstruct model architecture based on model type."""
        if model_key == 'bci2a_crdae':
            return self._create_bci2a_crdae_model()
        elif model_key == 'eeg_pd':
            return self._create_eeg_pd_model()
        elif model_key == 'nhrn_pd':
            from .nhrn_model import create_neuromorphic_hierarchical_resonance_network
            return create_neuromorphic_hierarchical_resonance_network(
                input_shape=self.model_info['nhrn_pd']['input_shape'],
                num_classes=len(self.model_info['nhrn_pd']['output_classes'])
            )
        else:
            raise ValueError(f"Unknown model key: {model_key}")
    
    def _create_bci2a_crdae_model(self) -> tf.keras.Model:
        """Create BCI2A CRDAE model architecture."""
        input_shape = self.model_info['bci2a_crdae']['input_shape']
        
        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Reshape for CNN
        x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
        
        # Convolutional layers - simplified architecture
        x = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(4, activation='softmax')(x)  # 4 classes for motor imagery
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _create_eeg_pd_model(self) -> tf.keras.Model:
        """Create EEG Parkinson's Disease model architecture."""
        input_shape = self.model_info['eeg_pd']['input_shape']
        
        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Reshape for CNN
        x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
        
        # Convolutional layers
        x = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(2, activation='softmax')(x)  # 2 classes: Healthy/PD
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def predict(self, data: np.ndarray, model_key: str) -> Dict[str, Any]:
        """
        Make prediction using specified model.
        
        Args:
            data: Input data array
            model_key: Key identifying which model to use
            
        Returns:
            Dictionary containing prediction results
        """
        # Try to use the original model first
        if model_key in self.models:
            model = self.models[model_key]
            model_info = self.model_info[model_key]
            
            # Preprocess data to match model input shape
            processed_data = self._preprocess_data(data, model_key)
            
            # Make prediction in chunks to prevent Out-Of-Memory (OOM) errors on large files
            try:
                chunk_size = 16
                probs_list = []
                
                if model_key == 'neuroformer':
                    import torch
                    import torch.nn.functional as F
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    for i in range(0, processed_data.shape[0], chunk_size):
                        chunk = processed_data[i:i+chunk_size]
                        xb = torch.from_numpy(chunk).float().to(device)
                        with torch.cuda.amp.autocast():
                            logits = model(xb)
                        chunk_probs = F.softmax(logits.float(), dim=-1).detach().cpu().numpy()
                        probs_list.append(chunk_probs)
                    prediction_probs = np.vstack(probs_list)
                    
                elif model_key == 'spectra_sz':
                    import torch
                    import torch.nn.functional as F
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    for i in range(0, processed_data.shape[0], chunk_size):
                        chunk = processed_data[i:i+chunk_size]
                        xb = torch.from_numpy(chunk).float().to(device)
                        
                        batch_size = xb.shape[0]
                        phase_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
                        pad_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
                        device_id = torch.zeros((batch_size,), dtype=torch.long, device=device)
                        protocol_id = torch.zeros((batch_size,), dtype=torch.long, device=device)
                        
                        with torch.no_grad():
                            out = model(xb, phase_ids, pad_mask, device_id, protocol_id)
                            logits = out['logits']
                        chunk_probs = F.softmax(logits.float(), dim=-1).detach().cpu().numpy()
                        probs_list.append(chunk_probs)
                    prediction_probs = np.vstack(probs_list)
                    
                else:
                    for i in range(0, processed_data.shape[0], chunk_size):
                        chunk = processed_data[i:i+chunk_size]
                        chunk_probs = model.predict(chunk, verbose=0)
                        probs_list.append(chunk_probs)
                    prediction_probs = np.vstack(probs_list)
            except Exception as e:
                logger.error(f"Prediction error for {model_key}: {e}")
                return {
                    'prediction': 'Error',
                    'probability': 0.0,
                    'class_probabilities': {},
                    'model_used': model_info['name'],
                    'error': str(e)
                }

            # Average predictions across all segments
            if prediction_probs.shape[0] > 1:
                avg_prediction_probs = np.mean(prediction_probs, axis=0, keepdims=True)
            else:
                avg_prediction_probs = prediction_probs

            prediction_class_idx = int(np.argmax(avg_prediction_probs[0]))
            prediction_class = model_info['output_classes'][prediction_class_idx]
            confidence = float(avg_prediction_probs[0][prediction_class_idx])

            # Apply confidence threshold
            try:
                threshold = MODEL_CONFIG.get(model_key, {}).get('confidence_threshold', 0.5)
            except Exception:
                threshold = 0.5
            
            # Create class probabilities dictionary
            class_probabilities = {
                class_name: float(prob) 
                for class_name, prob in zip(model_info['output_classes'], avg_prediction_probs[0])
            }
            
            below_threshold = confidence < threshold

            result = {
                'prediction': 'Uncertain' if below_threshold else prediction_class,
                'probability': confidence,
                'class_probabilities': class_probabilities,
                'model_used': model_info['name'],
                'raw_predictions': avg_prediction_probs[0].tolist(),
                'below_threshold': below_threshold,
                'threshold': float(threshold),
                'raw_predicted_class': prediction_class,
            }

            if below_threshold:
                logger.info(
                    f"Prediction below threshold for {model_key}: {prediction_class} ({confidence:.2f} < {threshold:.2f})"
                )

            return result
        else:
            # Fallback to simple model
            logger.warning(f"Using simple model for {model_key} (original model not available)")
            return self.simple_loader.predict(data, model_key)
            
    def _preprocess_data(self, data: np.ndarray, model_key: str) -> np.ndarray:
        """Preprocess data to match model input requirements."""
        if model_key == 'brain_tumor_mri':
            target_shape = self.model_info[model_key]['input_shape']
            return self._preprocess_image_data(data, target_shape)
            
        # For all EEG models, slice into 3-second segments and take 5 optimal channels
        segments = self._segment_eeg_data(data)
        target_shape = self.model_info[model_key]['input_shape']
        
        preprocessed_segments = []
        for segment in segments:
            prep_seg = self._preprocess_eeg_segment(segment, model_key, target_shape)
            preprocessed_segments.append(prep_seg)
            
        return np.array(preprocessed_segments)

    def _segment_eeg_data(self, data: np.ndarray) -> list:
        """Slice raw EEG (Samples, Channels) into non-overlapping 3-second segments (750 samples)."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        segment_len = 750
        total_samples = data.shape[0]
        
        if total_samples <= segment_len:
            pad_width = segment_len - total_samples
            padded_data = np.pad(data, ((0, pad_width), (0, 0)), mode='constant')
            return [padded_data]
            
        num_segments = total_samples // segment_len
        segments = []
        for i in range(num_segments):
            start_idx = i * segment_len
            end_idx = start_idx + segment_len
            segments.append(data[start_idx:end_idx, :])
        return segments

    def _preprocess_eeg_segment(self, segment: np.ndarray, model_key: str, target_shape: tuple) -> np.ndarray:
        """Preprocess a single 3-second segment for the target model using 5 optimal channels."""
        num_input_chans = segment.shape[1]
        chans_to_use = min(5, num_input_chans)
        segment_5ch = segment[:, :chans_to_use]
        
        if model_key in ['neuroformer', 'spectra_sz', 'nhrn_pd']:
            segment_t = segment_5ch.T
            
            target_chans = target_shape[0]
            if segment_t.shape[0] < target_chans:
                pad_chans = target_chans - segment_t.shape[0]
                segment_t = np.pad(segment_t, ((0, pad_chans), (0, 0)), mode='constant')
            elif segment_t.shape[0] > target_chans:
                segment_t = segment_t[:target_chans, :]
                
            target_samples = target_shape[1]
            if segment_t.shape[1] < target_samples:
                pad_samples = target_samples - segment_t.shape[1]
                segment_t = np.pad(segment_t, ((0, 0), (0, pad_samples)), mode='constant')
            elif segment_t.shape[1] > target_samples:
                segment_t = segment_t[:, :target_samples]
                
            mean = np.mean(segment_t, axis=1, keepdims=True)
            std = np.std(segment_t, axis=1, keepdims=True) + 1e-8
            segment_norm = (segment_t - mean) / std
            
            if model_key == 'spectra_sz':
                return np.expand_dims(segment_norm, axis=0).astype(np.float32)
            else:
                return segment_norm.astype(np.float32)
        else:
            target_samples = target_shape[0]
            target_chans = target_shape[1]
            
            if segment_5ch.shape[1] < target_chans:
                pad_chans = target_chans - segment_5ch.shape[1]
                segment_5ch = np.pad(segment_5ch, ((0, 0), (0, pad_chans)), mode='constant')
            elif segment_5ch.shape[1] > target_chans:
                segment_5ch = segment_5ch[:, :target_chans]
                
            if segment_5ch.shape[0] < target_samples:
                pad_samples = target_samples - segment_5ch.shape[0]
                segment_5ch = np.pad(segment_5ch, ((0, pad_samples), (0, 0)), mode='constant')
            elif segment_5ch.shape[0] > target_samples:
                segment_5ch = segment_5ch[:target_samples, :]
                
            return segment_5ch.astype(np.float32)

    def _preprocess_image_data(self, data: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Helper to preprocess image inputs for MRI classification."""
        def _process_image(sample: np.ndarray) -> np.ndarray:
            img = np.array(sample, dtype=np.float32)

            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)

            if img.ndim == 3 and img.shape[-1] > target_shape[-1]:
                img = img[..., :target_shape[-1]]
            elif img.ndim == 3 and img.shape[-1] < target_shape[-1]:
                padding = target_shape[-1] - img.shape[-1]
                img = np.pad(img, ((0, 0), (0, 0), (0, padding)), mode='constant')

            if img.shape[0] != target_shape[0] or img.shape[1] != target_shape[1]:
                img = tf.image.resize(img, target_shape[:2], method='bilinear').numpy()

            if np.max(img) > 1.0:
                img = img / 255.0

            return img.astype(np.float32)

        if data.ndim == 4:
            processed = np.array([_process_image(sample) for sample in data])
        else:
            processed = np.expand_dims(_process_image(data), axis=0)

        return processed
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of all models."""
        # Always return True since we have simple fallbacks
        return {key: True for key in self.model_paths.keys()}
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if model_key not in self.model_info:
            raise ValueError(f"Unknown model key: {model_key}")
        
        info = self.model_info[model_key].copy()
        info['loaded'] = model_key in self.models
        return info
    
    def get_available_models(self) -> list:
        """Get list of available (loaded) models."""
        return list(self.model_paths.keys())
