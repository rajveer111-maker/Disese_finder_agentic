import numpy as np
import pandas as pd
from PIL import Image
import io
from typing import Union, Tuple, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGPreprocessor:
    """
    Preprocessing utilities for EEG data.
    """
    
    def __init__(self):
        self.sampling_rate = 250  # Default sampling rate
        self.standard_channels = 22  # Standard number of channels for BCI data
    
    def load_eeg_data(self, file) -> np.ndarray:
        """
        Load EEG data from various file formats.
        
        Args:
            file: File object or path
            
        Returns:
            EEG data as numpy array
        """
        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            return self._load_csv_eeg(file)
        elif file_extension == 'txt':
            return self._load_txt_eeg(file)
        elif file_extension == 'edf':
            return self._load_edf_eeg(file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_csv_eeg(self, file) -> np.ndarray:
        """Load EEG data from CSV file."""
        try:
            # Try to read as CSV
            df = pd.read_csv(file, header=None)
            data = df.values
            
            # If data is 1D, reshape it
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            logger.info(f"Loaded CSV EEG data: {data.shape}")
            return data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    def _load_txt_eeg(self, file) -> np.ndarray:
        """Load EEG data from TXT file."""
        try:
            # Read file content
            content = file.read().decode('utf-8')
            
            # Split into lines and convert to numbers
            lines = content.strip().split('\n')
            data = []
            
            for line in lines:
                if line.strip():
                    # Split by whitespace or comma
                    values = line.replace(',', ' ').split()
                    try:
                        row = [float(v) for v in values]
                        data.append(row)
                    except ValueError:
                        continue
            
            if not data:
                raise ValueError("No valid data found in TXT file")
            
            data = np.array(data, dtype=np.float32)
            
            # If data is 1D, reshape it
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            logger.info(f"Loaded TXT EEG data: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading TXT file: {e}")
            raise
    
    def _load_edf_eeg(self, file) -> np.ndarray:
        """Load EEG data from EDF file."""
        try:
            # For EDF files, we'll use a simplified approach
            # In a real implementation, you'd use pyedflib or mne
            logger.warning("EDF file loading is simplified. Consider using MNE for full EDF support.")
            
            # For now, treat as binary data and try to extract numeric values
            file.seek(0)
            raw_data = file.read()
            
            # Convert to numpy array (this is a simplified approach)
            data = np.frombuffer(raw_data, dtype=np.float32)
            
            # Reshape to 2D (assume 22 channels)
            if len(data) % self.standard_channels == 0:
                data = data.reshape(-1, self.standard_channels)
            else:
                # Pad or truncate to fit standard channels
                target_length = (len(data) // self.standard_channels) * self.standard_channels
                data = data[:target_length].reshape(-1, self.standard_channels)
            
            logger.info(f"Loaded EDF EEG data: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading EDF file: {e}")
            raise
    
    def preprocess_eeg(self, data: np.ndarray, 
                      apply_filter: bool = True,
                      apply_normalization: bool = True) -> np.ndarray:
        """
        Preprocess EEG data.
        
        Args:
            data: Raw EEG data
            apply_filter: Whether to apply bandpass filter
            apply_normalization: Whether to normalize data
            
        Returns:
            Preprocessed EEG data
        """
        processed_data = data.copy()
        
        # Remove DC offset
        processed_data = processed_data - np.mean(processed_data, axis=0, keepdims=True)
        
        # Apply bandpass filter (simplified)
        if apply_filter:
            processed_data = self._apply_bandpass_filter(processed_data)
        
        # Normalize data
        if apply_normalization:
            processed_data = self._normalize_data(processed_data)
        
        return processed_data
    
    def _apply_bandpass_filter(self, data: np.ndarray, 
                              low_freq: float = 1.0, 
                              high_freq: float = 40.0) -> np.ndarray:
        """Apply bandpass filter to EEG data."""
        # Simplified bandpass filter implementation
        # In a real implementation, you'd use scipy.signal or mne
        
        # Calculate frequency domain
        fft_data = np.fft.fft(data, axis=0)
        freqs = np.fft.fftfreq(data.shape[0], 1/self.sampling_rate)
        
        # Create filter mask
        filter_mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
        filter_mask = filter_mask.reshape(-1, 1)
        
        # Apply filter
        fft_data = fft_data * filter_mask
        
        # Convert back to time domain
        filtered_data = np.real(np.fft.ifft(fft_data, axis=0))
        
        return filtered_data
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize EEG data."""
        # Z-score normalization
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        normalized_data = (data - mean) / std
        return normalized_data
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from EEG data."""
        features = []
        
        # Statistical features
        features.extend([
            np.mean(data, axis=0),
            np.std(data, axis=0),
            np.var(data, axis=0),
            np.max(data, axis=0),
            np.min(data, axis=0)
        ])
        
        # Spectral features (simplified)
        fft_data = np.abs(np.fft.fft(data, axis=0))
        features.extend([
            np.mean(fft_data, axis=0),
            np.std(fft_data, axis=0)
        ])
        
        # Combine all features
        feature_vector = np.concatenate(features)
        return feature_vector.reshape(1, -1)

class ImagePreprocessor:
    """
    Preprocessing utilities for medical images.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess medical image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Apply basic preprocessing
        image = self._enhance_contrast(image)
        image = self._normalize_image(image)
        
        return image
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Enhance image contrast."""
        from PIL import ImageEnhance
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Increase contrast by 20%
        
        return image
    
    def _normalize_image(self, image: Image.Image) -> Image.Image:
        """Normalize image values."""
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to 0-1 range
        img_array = img_array / 255.0
        
        # Convert back to PIL Image
        img_array = (img_array * 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        
        return image
    
    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        """Extract features from medical image."""
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Basic statistical features
        features = [
            np.mean(img_array),
            np.std(img_array),
            np.var(img_array),
            np.max(img_array),
            np.min(img_array)
        ]
        
        # Histogram features
        hist, _ = np.histogram(img_array.flatten(), bins=32, range=(0, 255))
        hist = hist / np.sum(hist)  # Normalize histogram
        features.extend(hist)
        
        return np.array(features).reshape(1, -1)
    
    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image characteristics."""
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray_array = np.mean(img_array, axis=2)
        else:
            gray_array = img_array
        
        # Calculate image characteristics
        brightness = np.mean(gray_array)
        contrast = np.std(gray_array)
        
        # Calculate complexity (using edge detection)
        from scipy import ndimage
        sobel_x = ndimage.sobel(gray_array, axis=1)
        sobel_y = ndimage.sobel(gray_array, axis=0)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        complexity = np.mean(edge_magnitude)
        
        # Edge density
        edge_density = np.sum(edge_magnitude > np.percentile(edge_magnitude, 90)) / edge_magnitude.size
        
        # Color variance (if color image)
        if len(img_array.shape) == 3:
            color_variance = np.var(img_array, axis=(0, 1)).mean()
        else:
            color_variance = np.var(gray_array)
        
        # Determine image type based on characteristics
        if brightness < 50:
            image_type = "X-ray/CT Scan"
        elif brightness > 200:
            image_type = "Ultrasound"
        elif complexity > 50:
            image_type = "EEG/Spectrogram"
        elif edge_density > 0.1:
            image_type = "Medical Chart"
        else:
            image_type = "General Medical Image"
        
        return {
            'image_type': image_type,
            'brightness': float(brightness),
            'contrast': float(contrast),
            'complexity': float(complexity),
            'edge_density': float(edge_density),
            'color_variance': float(color_variance)
        }
    
    def analyze_medical_image(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze medical image and provide diagnosis suggestions."""
        # Get image analysis
        analysis = self.analyze_image(image)
        
        # Create mock diagnosis based on image characteristics
        image_type = analysis['image_type']
        brightness = analysis['brightness']
        contrast = analysis['contrast']
        complexity = analysis['complexity']
        
        # Simple rule-based analysis
        if image_type == "X-ray/CT Scan":
            if brightness < 30:
                prediction = "Possible abnormality detected"
                confidence = 0.75
                reasoning = "Low brightness suggests potential pathology"
            else:
                prediction = "Normal appearance"
                confidence = 0.85
                reasoning = "Normal brightness and contrast levels"
        
        elif image_type == "Ultrasound":
            if complexity > 40:
                prediction = "Complex structure detected"
                confidence = 0.70
                reasoning = "High complexity suggests detailed anatomical structure"
            else:
                prediction = "Clear ultrasound image"
                confidence = 0.80
                reasoning = "Good image quality with clear structures"
        
        elif image_type == "EEG/Spectrogram":
            if contrast > 50:
                prediction = "Abnormal EEG pattern"
                confidence = 0.65
                reasoning = "High contrast suggests unusual brain activity"
            else:
                prediction = "Normal EEG pattern"
                confidence = 0.75
                reasoning = "Normal contrast levels in EEG signal"
        
        elif image_type == "Medical Chart":
            prediction = "Vital signs within normal range"
            confidence = 0.90
            reasoning = "Medical chart shows stable vital signs"
        
        else:
            prediction = "Image requires further analysis"
            confidence = 0.60
            reasoning = "General medical image - detailed analysis needed"
        
        # Create class probabilities
        class_probabilities = {
            'Normal': 1.0 - confidence if 'Normal' in prediction else 0.3,
            'Abnormal': confidence if 'Abnormal' in prediction else 0.2,
            'Uncertain': 0.1
        }
        
        return {
            'prediction': prediction,
            'probability': confidence,
            'confidence': confidence,  # Add confidence field for compatibility
            'class_probabilities': class_probabilities,
            'model_used': f'Image Analysis ({image_type})',
            'reasoning': reasoning,
            'raw_predictions': [class_probabilities['Normal'], class_probabilities['Abnormal'], class_probabilities['Uncertain']]
        }
