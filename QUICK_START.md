# 🚀 Quick Start Guide

## Agentic Disease Finder - Quick Start


You now have a complete **Agentic Disease Finder** application that can:

1. **Intelligently choose** between two specialized models based on input data
2. **Analyze EEG signals** for motor imagery classification and Parkinson's disease detection
3. **Provide a beautiful web interface** for easy interaction
4. **Generate visualizations** and detailed analysis reports

### 🎯 Key Features

- **Agentic Decision Making**: Automatically selects the best model for your data
- **Two Specialized Models**:
  - BCI2A CRDAE: Motor imagery classification (Left Hand, Right Hand, Foot, Tongue)
  - EEG Parkinson's Disease Detection: Clinical diagnosis support
- **Multi-format Support**: CSV, TXT, EDF files for EEG data
- **Real-time Analysis**: Fast processing with confidence scores
- **Interactive Visualizations**: Beautiful charts and graphs

### 🚀 Getting Started

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Run the Application
```bash
streamlit run app.py
```

#### 3. Open Your Browser
Navigate to: `http://localhost:8501`

### 📊 Test with Sample Data

Sample data has been generated for you! Use these files to test the application:

- `sample_data/motor_imagery_left_hand.csv` - Left hand motor imagery
- `sample_data/motor_imagery_right_hand.csv` - Right hand motor imagery  
- `sample_data/motor_imagery_foot.csv` - Foot motor imagery
- `sample_data/motor_imagery_tongue.csv` - Tongue motor imagery
- `sample_data/eeg_healthy.csv` - Healthy EEG data
- `sample_data/eeg_parkinsons.csv` - Parkinson's disease EEG data

### 🔧 How It Works

1. **Upload Data**: Go to "Upload & Analyze" tab and upload a CSV file
2. **Agentic Selection**: The system analyzes your data and chooses the best model
3. **Analysis**: The selected model processes your data and makes predictions
4. **Results**: View detailed results, confidence scores, and visualizations

### 🧠 Model Selection Logic

The agentic system considers:
- **File type compatibility** (CSV, TXT, EDF)
- **Data characteristics** (channels, duration, quality)
- **Use case context** (motor imagery vs. clinical diagnosis)
- **Confidence estimation** (predicted model performance)

### 📈 Understanding Results

- **Prediction**: The most likely class/condition
- **Confidence**: How certain the model is (0-100%)
- **Class Probabilities**: Breakdown of all possible outcomes
- **Model Reasoning**: Why this model was selected

### 🛠️ Customization

#### Adding Your Own Models
1. Place `.h5` model files in the `models/` directory
2. Update `ModelManager` class in `models/model_manager.py`
3. Add model capabilities to `AgenticDecisionSystem`

#### Modifying Preprocessing
1. Edit `EEGPreprocessor` in `utils/preprocessing.py`
2. Adjust filtering, normalization, or feature extraction
3. Update visualization functions as needed

### 🔍 Troubleshooting

#### Common Issues

1. **Model Loading Errors**
   - Ensure model files are in the correct location
   - Check file permissions
   - Verify TensorFlow installation

2. **Data Format Issues**
   - Ensure CSV files have numeric data
   - Check for proper channel separation
   - Verify sampling rate assumptions

3. **Memory Issues**
   - Reduce data size for large files
   - Close other applications
   - Consider data preprocessing

#### Getting Help

- Check the console output for error messages
- Review the model status in the sidebar
- Ensure all dependencies are installed correctly

### 📚 Next Steps

1. **Test with Real Data**: Upload your own EEG recordings
2. **Customize Models**: Add domain-specific models
3. **Extend UI**: Add new analysis features
4. **Deploy**: Consider cloud deployment for production use

### 🎉 Congratulations!

You now have a fully functional agentic disease finder that can intelligently analyze medical data and provide accurate predictions. The system demonstrates advanced AI capabilities including:

- Multi-model architecture
- Intelligent model selection
- Real-time data processing
- Interactive visualization
- Confidence-based decision making

Enjoy exploring your new AI-powered medical analysis tool! 🧠✨
