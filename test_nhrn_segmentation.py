jimport os
import numpy as np
import h5py
from models.model_manager import ModelManager

def test_segmentation():
    print("Initializing ModelManager...")
    manager = ModelManager()
    
    # 1. Test Segmentation Logic directly
    print("\nTesting _segment_eeg_data directly...")
    # 3000 samples, 8 channels (12 seconds of recording at 250Hz)
    dummy_input = np.random.randn(3000, 8)
    segments = manager._segment_eeg_data(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Number of sliced segments (expected 4): {len(segments)}")
    assert len(segments) == 4, f"Expected 4 segments, got {len(segments)}"
    print("Segment shape (expected (750, 8)):", segments[0].shape)
    assert segments[0].shape == (750, 8), f"Expected (750, 8), got {segments[0].shape}"
    
    # 2. Test EEG Segment Preprocessing
    print("\nTesting _preprocess_eeg_segment directly for nhrn_pd...")
    target_shape = manager.model_info['nhrn_pd']['input_shape'] # (40, 1024)
    prep_segment = manager._preprocess_eeg_segment(segments[0], 'nhrn_pd', target_shape)
    print(f"Preprocessed segment shape (expected (40, 1024)): {prep_segment.shape}")
    assert prep_segment.shape == (40, 1024), f"Expected (40, 1024), got {prep_segment.shape}"
    
    # Verify channel selection (first 5 channels used, the remaining 35 channels should be padded with zeros)
    # The padded channels are from index 5 to 39
    padded_channels_std = np.std(prep_segment[5:, :], axis=1)
    print("Standard deviation of padded channels (expected 0.0):", padded_channels_std)
    assert np.allclose(padded_channels_std, 0.0), "Padded channels are not zero-padded"
    
    # Verify active channels std (should be normalized, std = 1.0)
    active_channels_std = np.std(prep_segment[:5, :], axis=1)
    print("Standard deviation of active channels (expected 1.0):", active_channels_std)
    assert np.allclose(active_channels_std, 1.0), "Active channels are not normalized correctly"
    
    # 3. Test Ingress Prediction Pipeline
    print("\nRunning full predict pipeline on dummy 12-second signal...")
    result = manager.predict(dummy_input, 'nhrn_pd')
    print("Prediction result:")
    for k, v in result.items():
        if k != 'class_probabilities':
            print(f"  {k}: {v}")
        else:
            print(f"  class_probabilities: {v}")
            
    # 4. Test real dataset Rec_21april.h5 (852720 samples, 8 channels)
    dataset_path = 'sample_data/Rec_21april.h5'
    if os.path.exists(dataset_path):
        print(f"\nLoading real EEG recording from {dataset_path}...")
        with h5py.File(dataset_path, 'r') as f:
            real_data = f['data'][:].T
            print(f"Real data shape: {real_data.shape}")
            
        print("Running prediction with real data...")
        result_real = manager.predict(real_data, 'nhrn_pd')
        print("Real prediction result:")
        for k, v in result_real.items():
            if k != 'class_probabilities':
                print(f"  {k}: {v}")
            else:
                print(f"  class_probabilities: {v}")
    else:
        print(f"\n[WARNING] Real dataset {dataset_path} not found.")

if __name__ == '__main__':
    test_segmentation()
