#!/usr/bin/env python3
"""
Sample data generator for Agentic Disease Finder
Generates synthetic EEG data for testing and demonstration purposes.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

class SampleDataGenerator:
    """Generate sample EEG data for testing."""
    
    def __init__(self, sampling_rate=250, n_channels=22):
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.channel_names = [f'Channel_{i+1:02d}' for i in range(n_channels)]
    
    def generate_motor_imagery_data(self, duration=4, condition='left_hand'):
        """
        Generate motor imagery EEG data.
        
        Args:
            duration: Duration in seconds
            condition: 'left_hand', 'right_hand', 'foot', 'tongue'
        """
        n_samples = int(duration * self.sampling_rate)
        time = np.arange(n_samples) / self.sampling_rate
        
        # Create base signal with different characteristics for each condition
        base_freqs = {
            'left_hand': [8, 12, 20],  # Alpha, beta
            'right_hand': [10, 14, 22],
            'foot': [6, 10, 18],
            'tongue': [9, 13, 21]
        }
        
        base_amplitudes = {
            'left_hand': [15, 8, 5],
            'right_hand': [12, 10, 6],
            'foot': [18, 6, 4],
            'tongue': [14, 9, 7]
        }
        
        freqs = base_freqs[condition]
        amplitudes = base_amplitudes[condition]
        
        data = np.zeros((n_samples, self.n_channels))
        
        for i in range(self.n_channels):
            # Create signal with multiple frequency components
            signal = np.zeros(n_samples)
            
            for freq, amp in zip(freqs, amplitudes):
                # Add some randomness
                actual_freq = freq + np.random.normal(0, 0.5)
                actual_amp = amp + np.random.normal(0, 2)
                phase = np.random.uniform(0, 2*np.pi)
                
                signal += actual_amp * np.sin(2 * np.pi * actual_freq * time + phase)
            
            # Add channel-specific characteristics
            if i < 8:  # Frontal channels - more alpha
                signal += 5 * np.sin(2 * np.pi * 10 * time + np.random.uniform(0, 2*np.pi))
            elif i < 16:  # Central channels - more beta
                signal += 3 * np.sin(2 * np.pi * 20 * time + np.random.uniform(0, 2*np.pi))
            else:  # Posterior channels - more theta
                signal += 4 * np.sin(2 * np.pi * 6 * time + np.random.uniform(0, 2*np.pi))
            
            # Add noise
            noise = np.random.normal(0, 3, n_samples)
            data[:, i] = signal + noise
        
        return data, time
    
    def generate_parkinsons_data(self, duration=10, has_pd=True):
        """
        Generate Parkinson's disease EEG data.
        
        Args:
            duration: Duration in seconds
            has_pd: Whether to simulate Parkinson's disease characteristics
        """
        n_samples = int(duration * self.sampling_rate)
        time = np.arange(n_samples) / self.sampling_rate
        
        data = np.zeros((n_samples, self.n_channels))
        
        for i in range(self.n_channels):
            if has_pd:
                # Parkinson's characteristics: reduced beta, increased theta
                # Reduced beta (13-30 Hz)
                beta_freq = 20 + np.random.normal(0, 2)
                beta_amp = 3 + np.random.normal(0, 1)  # Reduced amplitude
                
                # Increased theta (4-8 Hz)
                theta_freq = 6 + np.random.normal(0, 1)
                theta_amp = 12 + np.random.normal(0, 3)  # Increased amplitude
                
                # Reduced alpha (8-12 Hz)
                alpha_freq = 10 + np.random.normal(0, 1)
                alpha_amp = 8 + np.random.normal(0, 2)  # Reduced amplitude
                
                # Increased delta (0.5-4 Hz)
                delta_freq = 2 + np.random.normal(0, 0.5)
                delta_amp = 6 + np.random.normal(0, 2)  # Increased amplitude
                
            else:
                # Healthy characteristics: normal frequency distribution
                beta_freq = 20 + np.random.normal(0, 2)
                beta_amp = 8 + np.random.normal(0, 2)
                
                theta_freq = 6 + np.random.normal(0, 1)
                theta_amp = 6 + np.random.normal(0, 2)
                
                alpha_freq = 10 + np.random.normal(0, 1)
                alpha_amp = 15 + np.random.normal(0, 3)
                
                delta_freq = 2 + np.random.normal(0, 0.5)
                delta_amp = 3 + np.random.normal(0, 1)
            
            # Create signal
            signal = (beta_amp * np.sin(2 * np.pi * beta_freq * time + np.random.uniform(0, 2*np.pi)) +
                     theta_amp * np.sin(2 * np.pi * theta_freq * time + np.random.uniform(0, 2*np.pi)) +
                     alpha_amp * np.sin(2 * np.pi * alpha_freq * time + np.random.uniform(0, 2*np.pi)) +
                     delta_amp * np.sin(2 * np.pi * delta_freq * time + np.random.uniform(0, 2*np.pi)))
            
            # Add noise
            noise = np.random.normal(0, 2, n_samples)
            data[:, i] = signal + noise
        
        return data, time

    def generate_alzheimers_data(self, duration=5, has_ad=True):
        """
        Generate Alzheimer's disease EEG data (19 channels).
        
        Args:
            duration: Duration in seconds
            has_ad: Whether to simulate AD characteristics
        """
        n_samples = int(duration * self.sampling_rate)
        time = np.arange(n_samples) / self.sampling_rate
        n_chan_ad = 19
        data = np.zeros((n_samples, n_chan_ad))
        
        for i in range(n_chan_ad):
            if has_ad:
                # AD characteristics: Slowing of EEG, increased theta/delta, decreased alpha/beta
                beta_amp = 3 + np.random.normal(0, 1)
                theta_amp = 15 + np.random.normal(0, 3)
                alpha_amp = 5 + np.random.normal(0, 2)
                delta_amp = 12 + np.random.normal(0, 2)
            else:
                beta_amp = 10 + np.random.normal(0, 2)
                theta_amp = 5 + np.random.normal(0, 1)
                alpha_amp = 15 + np.random.normal(0, 3)
                delta_amp = 4 + np.random.normal(0, 1)
            
            signal = (beta_amp * np.sin(2 * np.pi * 20 * time + np.random.uniform(0, 2*np.pi)) +
                     theta_amp * np.sin(2 * np.pi * 6 * time + np.random.uniform(0, 2*np.pi)) +
                     alpha_amp * np.sin(2 * np.pi * 10 * time + np.random.uniform(0, 2*np.pi)) +
                     delta_amp * np.sin(2 * np.pi * 2 * time + np.random.uniform(0, 2*np.pi)))
            
            noise = np.random.normal(0, 2, n_samples)
            data[:, i] = signal + noise
            
        return data, time
    
    def save_to_csv(self, data, filename, include_time=True):
        """Save data to CSV file."""
        if include_time:
            time = np.arange(data.shape[0]) / self.sampling_rate
            df_data = np.column_stack([time, data])
            columns = ['Time'] + self.channel_names
        else:
            df_data = data
            columns = self.channel_names
        
        df = pd.DataFrame(df_data, columns=columns)
        df.to_csv(filename, index=False)
        print(f"Saved data to {filename}")
    
    def generate_all_samples(self, output_dir="sample_data"):
        """Generate all sample data files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Motor imagery samples
        print("Generating motor imagery samples...")
        conditions = ['left_hand', 'right_hand', 'foot', 'tongue']
        for condition in conditions:
            data, time = self.generate_motor_imagery_data(duration=4, condition=condition)
            filename = os.path.join(output_dir, f"motor_imagery_{condition}.csv")
            self.save_to_csv(data, filename)
        
        # Parkinson's disease samples
        print("Generating Parkinson's disease samples...")
        for has_pd in [True, False]:
            condition = "parkinsons" if has_pd else "healthy"
            data, time = self.generate_parkinsons_data(duration=10, has_pd=has_pd)
            filename = os.path.join(output_dir, f"eeg_{condition}.csv")
            self.save_to_csv(data, filename)

        # Alzheimer's disease samples
        print("Generating Alzheimer's disease samples...")
        for has_ad in [True, False]:
            condition = "alzheimers" if has_ad else "cn"
            data, time = self.generate_alzheimers_data(duration=5, has_ad=has_ad)
            
            # Temporary override channel names for 19 channels
            original_names = self.channel_names
            self.channel_names = [f'Channel_{i+1:02d}' for i in range(19)]
            filename = os.path.join(output_dir, f"eeg_{condition}.csv")
            self.save_to_csv(data, filename)
            self.channel_names = original_names
        
        print(f"All sample data generated in {output_dir}/")
        print("   Files created:")
        for file in os.listdir(output_dir):
            if file.endswith('.csv'):
                print(f"   - {file}")

def main():
    """Main function to generate sample data."""
    print("Sample Data Generator for Agentic Disease Finder")
    print("=" * 60)
    
    generator = SampleDataGenerator()
    generator.generate_all_samples()
    
    print("\nSample data files can be used to test the application:")
    print("   1. Upload the CSV files in the Streamlit app")
    print("   2. The agentic system will automatically select the appropriate model")
    print("   3. View the results and visualizations")

if __name__ == "__main__":
    main()
