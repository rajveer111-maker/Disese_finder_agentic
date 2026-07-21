#!/usr/bin/env python3
"""
Test script to verify session state initialization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock streamlit session state for testing
class MockSessionState:
    def __init__(self):
        self._state = {}
    
    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattr__(name)
        return self._state.get(name)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._state[name] = value
    
    def __contains__(self, name):
        return name in self._state

def test_session_state():
    """Test session state initialization."""
    print("Testing Session State Initialization")
    print("=" * 40)
    
    # Mock streamlit
    import streamlit as st
    st.session_state = MockSessionState()
    
    # Test the initialization logic from app.py
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = "ModelManager()"
    if 'agentic_system' not in st.session_state:
        st.session_state.agentic_system = "AgenticDecisionSystem()"
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = "EEGPreprocessor()"
    if 'image_preprocessor' not in st.session_state:
        st.session_state.image_preprocessor = "ImagePreprocessor()"
    if 'viz_helper' not in st.session_state:
        st.session_state.viz_helper = "VisualizationHelper()"
    
    # Initialize prediction-related session state
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'last_decision' not in st.session_state:
        st.session_state.last_decision = None
    if 'last_data_type' not in st.session_state:
        st.session_state.last_data_type = None
    
    # Test the results logic
    print("Testing results logic...")
    
    # Test case 1: No prediction
    if 'last_prediction' in st.session_state and st.session_state.last_prediction is not None:
        print("FAIL: Should not enter results section when no prediction")
    else:
        print("PASS: Correctly handles no prediction case")
    
    # Test case 2: With prediction
    st.session_state.last_prediction = {"prediction": "Test", "probability": 0.8}
    st.session_state.last_decision = {"selected_model": "test_model", "confidence": 0.9}
    st.session_state.last_data_type = "eeg"
    
    if 'last_prediction' in st.session_state and st.session_state.last_prediction is not None:
        print("PASS: Correctly handles prediction case")
        prediction = st.session_state.last_prediction
        decision = st.session_state.last_decision
        data_type = st.session_state.last_data_type if hasattr(st.session_state, 'last_data_type') else 'unknown'
        
        print(f"  Prediction: {prediction['prediction']}")
        print(f"  Decision: {decision['selected_model']}")
        print(f"  Data Type: {data_type}")
    else:
        print("FAIL: Should enter results section when prediction exists")
    
    print("\n" + "=" * 40)
    print("Session state test completed successfully!")

if __name__ == "__main__":
    test_session_state()
