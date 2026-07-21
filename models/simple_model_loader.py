"""
Simple model loader that creates basic models for demonstration purposes
when the original complex models can't be loaded.
Adds confidence thresholding and basic logging for reliability.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any
import logging
from config import MODEL_CONFIG

logger = logging.getLogger(__name__)

class SimpleModelLoader:
    """Simple model loader for demonstration purposes."""
    
    def __init__(self):
        self.models = {}
        self._create_simple_models()
    
    def _create_simple_models(self):
        """Create simple models for demonstration."""
        
        # EEGNet BCI2A model
        bci2a_model = self._create_eegnet_model(nb_classes=4)
        self.models['bci2a_crdae'] = bci2a_model
        
        # EEGNet PD model
        eeg_pd_model = self._create_eegnet_model(nb_classes=2)
        self.models['eeg_pd'] = eeg_pd_model

        # EEGNet Autism (ASD) model
        autism_model = self._create_eegnet_model(nb_classes=2)
        self.models['autism_asd'] = autism_model

        # EEGNet NHRN PD model (40 channels, 1024 samples)
        nhrn_pd_model = self._create_eegnet_model(nb_classes=2, Chans=40, Samples=1024)
        self.models['nhrn_pd'] = nhrn_pd_model
    
    def _create_eegnet_model(self, nb_classes=4, Chans=22, Samples=1000, 
                             dropoutRate=0.5, kernLength=64, F1=8, 
                             D=2, F2=16, norm_rate=0.25, dropoutType='Dropout') -> tf.keras.Model:
        """
        Create EEGNet model.
        Adapted from Lawhern et al., 2018.
        """
        from tensorflow.keras.constraints import max_norm
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense, Dropout, SeparableConv2D, DepthwiseConv2D, Reshape
        
        # Input Shape: (Samples, Channels) -> we need (1, Channels, Samples) or (Samples, Channels, 1)
        # Our preprocessing gives (Samples, Channels). We will treat it as (Samples, 1, Channels) or just Reshape inside.
        # Let's target (Channels, Samples, 1) usually, but typically EEGNet takes (Channels, Samples, 1).
        # However, our data is (Samples=1000, Channels=22). 
        # Let's reshape to (1, Channels, Samples) inside the model for standard EEGNet topology
        
        input1   = Input(shape=(Samples, Chans))
        
        # Reshape layer to match EEGNet expectations: (Chans, Samples, 1)
        # Note: Our data is (Samples, Chans). Let's permute to (Chans, Samples) then add dim.
        x = Reshape((Chans, Samples, 1))(input1)

        ##################################################################
        # Block 1 : Conv2D + BatchNormalization + DepthwiseConv2D + ...
        ##################################################################
        
        # Conv2D (Temporal Convolution)
        x = Conv2D(F1, (1, kernLength), padding='same',
                   input_shape=(Chans, Samples, 1),
                   use_bias=False)(x)
        x = BatchNormalization()(x)
        
        # DepthwiseConv2D (Spatial Filter)
        x = DepthwiseConv2D((Chans, 1), use_bias=False, 
                            depth_multiplier=D,
                            depthwise_constraint=max_norm(1.))(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = AveragePooling2D((1, 4))(x)
        x = Dropout(dropoutRate)(x)
        
        ##################################################################
        # Block 2 : SeparableConv2D + ...
        ##################################################################
        x = SeparableConv2D(F2, (1, 16),
                            use_bias=False, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = AveragePooling2D((1, 8))(x)
        x = Dropout(dropoutRate)(x)
        
        flatten = Flatten(name='flatten')(x)
        
        dense = Dense(nb_classes, name='dense', 
                      kernel_constraint=max_norm(norm_rate))(flatten)
        softmax = Activation('softmax', name='softmax')(dense)
        
        return tf.keras.Model(inputs=input1, outputs=softmax)

    def predict(self, data: np.ndarray, model_key: str) -> Dict[str, Any]:
        """Make prediction using simple models."""
        if model_key not in self.models:
            if model_key == 'neuroformer':
                return {
                    'prediction': 'Uncertain',
                    'probability': 0.0,
                    'class_probabilities': {'AD': 0.33, 'CN': 0.33, 'FTD': 0.33},
                    'model_used': 'Simple Neuroformer (Dummy)',
                    'raw_predictions': [0.33, 0.33, 0.33],
                    'below_threshold': True,
                    'threshold': 0.75,
                    'raw_predicted_class': 'CN',
                }
            elif model_key == 'brain_tumor_mri':
                return {
                    'prediction': 'No Tumor',
                    'probability': 0.45,
                    'class_probabilities': {'Glioma': 0.15, 'Meningioma': 0.20, 'No Tumor': 0.45, 'Pituitary': 0.20},
                    'model_used': 'Simple Brain MRI (Dummy)',
                    'raw_predictions': [0.15, 0.20, 0.45, 0.20],
                    'below_threshold': False,
                    'threshold': 0.01,
                    'raw_predicted_class': 'No Tumor',
                }
            raise ValueError(f"Model {model_key} not available in simple loader")
        
        model = self.models[model_key]
        processed_data = self._preprocess_data(data, model_key)
        
        try:
            prediction_probs = model.predict(processed_data, verbose=0)
        except Exception as e:
            logger.error(f"Prediction error for {model_key}: {e}")
            return {
                'prediction': 'Error',
                'probability': 0.0,
                'class_probabilities': {},
                'model_used': f'Simple {model_key}',
                'error': str(e)
            }
            
        prediction_class_idx = int(np.argmax(prediction_probs[0]))

        if model_key == 'bci2a_crdae':
            class_names = MODEL_CONFIG[model_key]['output_classes']
        elif model_key == 'eeg_pd':
            class_names = MODEL_CONFIG[model_key]['output_classes']
        elif model_key == 'autism_asd':
            class_names = MODEL_CONFIG[model_key]['output_classes']
        elif model_key == 'nhrn_pd':
            class_names = MODEL_CONFIG[model_key]['output_classes']
        else:
            class_names = [f"Class_{i}" for i in range(prediction_probs.shape[1])]
        
        prediction_class = class_names[prediction_class_idx]
        confidence = float(prediction_probs[0][prediction_class_idx])

        # Apply confidence threshold
        threshold = MODEL_CONFIG.get(model_key, {}).get('confidence_threshold', 0.5)
        
        # Create class probabilities dictionary
        class_probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(class_names, prediction_probs[0])
        }
        
        below_threshold = confidence < threshold

        result = {
            'prediction': 'Uncertain' if below_threshold else prediction_class,
            'probability': confidence,
            'class_probabilities': class_probabilities,
            'model_used': f'Simple {model_key} (Demo)',
            'raw_predictions': prediction_probs[0].tolist(),
            'below_threshold': below_threshold,
            'threshold': float(threshold),
            'raw_predicted_class': prediction_class,
        }

        if below_threshold:
            logger.info(
                f"Prediction below threshold for {model_key}: {prediction_class} ({confidence:.2f} < {threshold:.2f})"
            )

        return result
    
    def _preprocess_data(self, data: np.ndarray, model_key: str) -> np.ndarray:
        """Preprocess data to match model input requirements."""
        target_shape = MODEL_CONFIG.get(model_key, {}).get('input_shape', (1000, 22))

        if len(target_shape) == 2:
            # Ensure data is 2D
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            # Pad or truncate to target length
            if data.shape[0] > target_shape[0]:
                data = data[:target_shape[0]]
            elif data.shape[0] < target_shape[0]:
                padding = np.zeros((target_shape[0] - data.shape[0], data.shape[1]))
                data = np.vstack([data, padding])

            # Adjust number of channels
            if data.shape[1] > target_shape[1]:
                data = data[:, :target_shape[1]]
            elif data.shape[1] < target_shape[1]:
                padding = np.zeros((data.shape[0], target_shape[1] - data.shape[1]))
                data = np.hstack([data, padding])

            data = np.expand_dims(data, axis=0)
            return data.astype(np.float32)

            data = np.expand_dims(data, axis=0)
            return data.astype(np.float32)

        return data.astype(np.float32) # Fallback, should likely be covered above
    
    def get_available_models(self) -> list:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of all models."""
        return {key: True for key in self.models.keys()}
