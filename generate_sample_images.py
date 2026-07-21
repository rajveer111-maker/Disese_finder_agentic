#!/usr/bin/env python3
"""
Sample Medical Image Generator for Agentic Disease Finder
Generates various types of medical images for testing purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime

class MedicalImageGenerator:
    """Generate sample medical images for testing."""
    
    def __init__(self, output_dir="sample_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_eeg_spectrogram(self, filename="eeg_spectrogram.png"):
        """Generate EEG spectrogram image."""
        # Create synthetic EEG data
        fs = 250  # Sampling frequency
        t = np.linspace(0, 10, 10 * fs)  # 10 seconds
        
        # Generate different frequency components
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t)  # Alpha waves
        beta = 0.3 * np.sin(2 * np.pi * 20 * t)   # Beta waves
        theta = 0.4 * np.sin(2 * np.pi * 6 * t)   # Theta waves
        delta = 0.2 * np.sin(2 * np.pi * 2 * t)   # Delta waves
        
        # Combine signals
        eeg_signal = alpha + beta + theta + delta + 0.1 * np.random.randn(len(t))
        
        # Create spectrogram
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time series plot
        ax1.plot(t, eeg_signal, 'b-', linewidth=0.5)
        ax1.set_title('EEG Signal - 10 seconds', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude (μV)')
        ax1.grid(True, alpha=0.3)
        
        # Spectrogram
        Pxx, freqs, bins, im = ax2.specgram(eeg_signal, Fs=fs, NFFT=512, noverlap=256, cmap='viridis')
        ax2.set_title('EEG Spectrogram', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_ylim(0, 50)
        plt.colorbar(im, ax=ax2, label='Power (dB)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {filename}")
    
    def generate_brain_mri(self, filename="brain_mri.png"):
        """Generate synthetic brain MRI image."""
        # Create a synthetic brain MRI-like image
        size = 256
        img = np.zeros((size, size))
        
        # Create brain-like structure
        center = size // 2
        y, x = np.ogrid[:size, :size]
        
        # Brain outline
        brain_mask = ((x - center) ** 2 + (y - center) ** 2) < (center - 20) ** 2
        brain_mask = brain_mask & ((x - center) ** 2 + (y - center) ** 2) > (center - 80) ** 2
        
        # Add some structure
        img[brain_mask] = 0.3
        
        # Add ventricles
        ventricle_mask = ((x - center) ** 2 + (y - center) ** 2) < (center - 40) ** 2
        ventricle_mask = ventricle_mask & ((x - center) ** 2 + (y - center) ** 2) > (center - 60) ** 2
        img[ventricle_mask] = 0.8
        
        # Add some noise and texture
        noise = np.random.normal(0, 0.1, (size, size))
        img += noise
        img = np.clip(img, 0, 1)
        
        # Convert to PIL Image
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        
        # Add text overlay
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "Brain MRI - Axial View", fill=255, font=font)
        draw.text((10, 30), "Generated for Testing", fill=200, font=font)
        
        img_pil.save(os.path.join(self.output_dir, filename))
        print(f"Generated: {filename}")
    
    def generate_ecg_trace(self, filename="ecg_trace.png"):
        """Generate ECG trace image."""
        # Create synthetic ECG data
        fs = 1000  # Sampling frequency
        duration = 5  # seconds
        t = np.linspace(0, duration, duration * fs)
        
        # Generate ECG-like signal
        ecg = np.zeros_like(t)
        
        # Heart rate (beats per minute)
        hr = 75
        beat_interval = 60 / hr  # seconds between beats
        
        for i in range(int(duration / beat_interval)):
            beat_time = i * beat_interval
            if beat_time < duration:
                # QRS complex
                qrs_start = int(beat_time * fs)
                qrs_end = min(qrs_start + int(0.1 * fs), len(ecg))
                if qrs_start < len(ecg):
                    ecg[qrs_start:qrs_end] += 1.0 * np.exp(-((t[qrs_start:qrs_end] - beat_time) / 0.05) ** 2)
        
        # Add baseline wander and noise
        baseline = 0.1 * np.sin(2 * np.pi * 0.5 * t)
        noise = 0.05 * np.random.randn(len(t))
        ecg += baseline + noise
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t, ecg, 'b-', linewidth=1)
        ax.set_title('ECG Trace - 5 seconds', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, duration)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {filename}")
    
    def generate_medical_chart(self, filename="medical_chart.png"):
        """Generate medical chart with multiple signals."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Generate time axis
        t = np.linspace(0, 10, 1000)
        
        # Blood pressure
        bp_systolic = 120 + 10 * np.sin(2 * np.pi * 0.1 * t) + 2 * np.random.randn(len(t))
        bp_diastolic = 80 + 5 * np.sin(2 * np.pi * 0.1 * t) + 1 * np.random.randn(len(t))
        
        axes[0].plot(t, bp_systolic, 'r-', label='Systolic', linewidth=2)
        axes[0].plot(t, bp_diastolic, 'b-', label='Diastolic', linewidth=2)
        axes[0].set_title('Blood Pressure Monitoring', fontweight='bold')
        axes[0].set_ylabel('Pressure (mmHg)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Heart rate
        hr = 70 + 5 * np.sin(2 * np.pi * 0.2 * t) + 3 * np.random.randn(len(t))
        axes[1].plot(t, hr, 'g-', linewidth=2)
        axes[1].set_title('Heart Rate', fontweight='bold')
        axes[1].set_ylabel('BPM')
        axes[1].grid(True, alpha=0.3)
        
        # Temperature
        temp = 98.6 + 0.5 * np.sin(2 * np.pi * 0.05 * t) + 0.1 * np.random.randn(len(t))
        axes[2].plot(t, temp, 'm-', linewidth=2)
        axes[2].set_title('Body Temperature', fontweight='bold')
        axes[2].set_xlabel('Time (minutes)')
        axes[2].set_ylabel('Temperature (°F)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {filename}")
    
    def generate_xray_image(self, filename="chest_xray.png"):
        """Generate synthetic chest X-ray image."""
        # Create synthetic X-ray image
        size = 512
        img = np.zeros((size, size))
        
        # Create chest-like structure
        center_x, center_y = size // 2, size // 2
        
        # Lungs (darker areas)
        for i in range(2):
            lung_x = center_x + (i * 2 - 1) * size // 4
            lung_y = center_y
            lung_width = size // 6
            lung_height = size // 2
            
            y, x = np.ogrid[:size, :size]
            lung_mask = ((x - lung_x) / lung_width) ** 2 + ((y - lung_y) / lung_height) ** 2 < 1
            img[lung_mask] = 0.2
        
        # Heart shadow
        heart_x, heart_y = center_x, center_y + size // 8
        y, x = np.ogrid[:size, :size]
        heart_mask = ((x - heart_x) / (size // 8)) ** 2 + ((y - heart_y) / (size // 6)) ** 2 < 1
        img[heart_mask] = 0.6
        
        # Ribs
        for i in range(8):
            rib_y = center_y - size // 3 + i * size // 12
            if 0 <= rib_y < size:
                img[rib_y, center_x - size // 3:center_x + size // 3] = 0.4
        
        # Add noise and texture
        noise = np.random.normal(0, 0.05, (size, size))
        img += noise
        img = np.clip(img, 0, 1)
        
        # Invert for X-ray effect (darker = more dense)
        img = 1 - img
        
        # Convert to PIL Image
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        
        # Add text overlay
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((20, 20), "Chest X-Ray - PA View", fill=255, font=font)
        draw.text((20, 50), "Generated for Testing", fill=200, font=font)
        
        img_pil.save(os.path.join(self.output_dir, filename))
        print(f"Generated: {filename}")
    
    def generate_ultrasound_image(self, filename="ultrasound.png"):
        """Generate synthetic ultrasound image."""
        # Create synthetic ultrasound image
        size = 400
        img = np.zeros((size, size))
        
        # Create ultrasound-like patterns
        x = np.linspace(0, 4 * np.pi, size)
        y = np.linspace(0, 4 * np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        # Create wave patterns
        img = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2 * X) * np.cos(2 * Y)
        
        # Add some structures
        center_x, center_y = size // 2, size // 2
        y_coords, x_coords = np.ogrid[:size, :size]
        
        # Add circular structure (like a cyst or organ)
        structure_mask = ((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) < (size // 6) ** 2
        img[structure_mask] += 0.5
        
        # Add noise
        noise = np.random.normal(0, 0.1, (size, size))
        img += noise
        
        # Normalize and convert to grayscale
        img = (img - img.min()) / (img.max() - img.min())
        
        # Convert to PIL Image
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        
        # Add text overlay
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "Ultrasound Image", fill=255, font=font)
        draw.text((10, 30), "Generated for Testing", fill=200, font=font)
        
        img_pil.save(os.path.join(self.output_dir, filename))
        print(f"Generated: {filename}")
    
    def generate_all_samples(self):
        """Generate all sample medical images."""
        print("Generating sample medical images...")
        print("=" * 50)
        
        self.generate_eeg_spectrogram("eeg_spectrogram.png")
        self.generate_brain_mri("brain_mri.png")
        self.generate_ecg_trace("ecg_trace.png")
        self.generate_medical_chart("medical_chart.png")
        self.generate_xray_image("chest_xray.png")
        self.generate_ultrasound_image("ultrasound.png")
        
        print("\n" + "=" * 50)
        print(f"All sample images generated in {self.output_dir}/")
        print("Files created:")
        for file in os.listdir(self.output_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                print(f"  - {file}")
        
        print("\nYou can now upload these images to test the Agentic Disease Finder!")

def main():
    """Main function to generate sample images."""
    print("Medical Image Generator for Agentic Disease Finder")
    print("=" * 60)
    
    generator = MedicalImageGenerator()
    generator.generate_all_samples()
    
    print("\nSample images can be used to test the application:")
    print("  1. Upload the PNG files in the Streamlit app")
    print("  2. The agentic system will analyze the images")
    print("  3. View the results and visualizations")

if __name__ == "__main__":
    main()
