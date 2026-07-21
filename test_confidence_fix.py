#!/usr/bin/env python3
"""
Test script to verify confidence field fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import ImagePreprocessor
from PIL import Image

def test_confidence_fix():
    """Test that confidence field is properly handled."""
    print("Testing Confidence Field Fix")
    print("=" * 40)
    
    # Initialize image preprocessor
    preprocessor = ImagePreprocessor()
    
    # Test with a sample image
    image_path = "test_images/eeg_waveform.png"
    
    if os.path.exists(image_path):
        print(f"Testing with: {image_path}")
        
        # Load image
        image = Image.open(image_path)
        
        # Analyze image
        prediction = preprocessor.analyze_medical_image(image)
        
        print("\nPrediction fields:")
        for key, value in prediction.items():
            print(f"  {key}: {value}")
        
        # Test confidence access
        print(f"\nConfidence access tests:")
        print(f"  prediction.get('confidence'): {prediction.get('confidence', 'NOT_FOUND')}")
        print(f"  prediction.get('probability'): {prediction.get('probability', 'NOT_FOUND')}")
        
        # Test the app logic
        confidence_value = prediction.get('confidence', prediction.get('probability', 0))
        print(f"  App logic result: {confidence_value}")
        
        if confidence_value > 0:
            print("PASS: Confidence field fix working correctly!")
        else:
            print("FAIL: Confidence field fix failed!")
    
    else:
        print(f"Image not found: {image_path}")
    
    print("\n" + "=" * 40)
    print("Confidence field test completed!")

if __name__ == "__main__":
    test_confidence_fix()
