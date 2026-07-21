#!/usr/bin/env python3
"""
Test script for image analysis functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import ImagePreprocessor
from PIL import Image

def test_image_analysis():
    """Test the image analysis functionality."""
    print("Testing Image Analysis Functionality")
    print("=" * 50)
    
    # Initialize image preprocessor
    preprocessor = ImagePreprocessor()
    
    # Test with sample images
    test_images = [
        "sample_images/eeg_spectrogram.png",
        "sample_images/brain_mri.png",
        "sample_images/chest_xray.png",
        "test_images/eeg_waveform.png",
        "test_images/vital_signs.png"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\nAnalyzing: {image_path}")
            print("-" * 30)
            
            try:
                # Load image
                image = Image.open(image_path)
                
                # Analyze image
                analysis = preprocessor.analyze_image(image)
                print(f"Image Type: {analysis['image_type']}")
                print(f"Brightness: {analysis['brightness']:.2f}")
                print(f"Contrast: {analysis['contrast']:.2f}")
                print(f"Complexity: {analysis['complexity']:.2f}")
                print(f"Edge Density: {analysis['edge_density']:.2f}")
                print(f"Color Variance: {analysis['color_variance']:.2f}")
                
                # Medical analysis
                medical_analysis = preprocessor.analyze_medical_image(image)
                print(f"Prediction: {medical_analysis['prediction']}")
                print(f"Confidence: {medical_analysis['probability']:.2%}")
                print(f"Reasoning: {medical_analysis['reasoning']}")
                
            except Exception as e:
                print(f"Error analyzing {image_path}: {e}")
        else:
            print(f"Image not found: {image_path}")
    
    print("\n" + "=" * 50)
    print("Image analysis test completed!")

if __name__ == "__main__":
    test_image_analysis()
