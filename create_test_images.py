#!/usr/bin/env python3
"""
Create additional test images for the Agentic Disease Finder
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_simple_test_images():
    """Create simple test images for upload testing."""
    output_dir = "test_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple EEG waveform image
    width, height = 800, 400
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple sine wave
    x_points = np.linspace(50, width-50, 200)
    y_points = height//2 + 50 * np.sin(2 * np.pi * x_points / 100)
    
    for i in range(len(x_points)-1):
        draw.line([(x_points[i], y_points[i]), (x_points[i+1], y_points[i+1])], fill='blue', width=2)
    
    # Add labels
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 20), "EEG Signal Test", fill='black', font=font)
    draw.text((50, height-30), "Time (s)", fill='black', font=font)
    
    img.save(os.path.join(output_dir, "eeg_waveform.png"))
    print("Created: eeg_waveform.png")
    
    # Create a medical chart image
    img2 = Image.new('RGB', (600, 400), 'white')
    draw2 = ImageDraw.Draw(img2)
    
    # Draw some bars representing medical data
    bar_data = [120, 80, 72, 98.6, 100, 95]
    bar_labels = ['BP Sys', 'BP Dia', 'HR', 'Temp', 'O2 Sat', 'RR']
    
    bar_width = 60
    bar_spacing = 20
    start_x = 50
    
    for i, (value, label) in enumerate(zip(bar_data, bar_labels)):
        x = start_x + i * (bar_width + bar_spacing)
        bar_height = int((value / 120) * 200)  # Scale to fit
        draw2.rectangle([x, 300-bar_height, x+bar_width, 300], fill='lightblue', outline='blue')
        draw2.text((x, 310), label, fill='black', font=font)
        draw2.text((x, 330), str(value), fill='black', font=font)
    
    draw2.text((50, 20), "Medical Vital Signs", fill='black', font=font)
    img2.save(os.path.join(output_dir, "vital_signs.png"))
    print("Created: vital_signs.png")
    
    # Create a simple brain scan image
    img3 = Image.new('RGB', (400, 400), 'black')
    draw3 = ImageDraw.Draw(img3)
    
    # Draw brain outline
    center_x, center_y = 200, 200
    draw3.ellipse([center_x-150, center_y-150, center_x+150, center_y+150], outline='white', width=3)
    
    # Draw some internal structures
    draw3.ellipse([center_x-100, center_y-100, center_x+100, center_y+100], outline='gray', width=2)
    draw3.ellipse([center_x-50, center_y-50, center_x+50, center_y+50], outline='lightgray', width=2)
    
    # Add labels
    draw3.text((50, 20), "Brain Scan", fill='white', font=font)
    draw3.text((50, 370), "Generated for Testing", fill='gray', font=font)
    
    img3.save(os.path.join(output_dir, "brain_scan.png"))
    print("Created: brain_scan.png")
    
    print(f"\nAll test images created in {output_dir}/")
    print("You can upload these to test the image analysis functionality!")

if __name__ == "__main__":
    create_simple_test_images()
