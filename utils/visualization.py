import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import pandas as pd

class VisualizationHelper:
    """
    Helper class for creating visualizations.
    """
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_eeg_signals(self, data: np.ndarray, 
                        sampling_rate: float = 250,
                        title: str = "EEG Signals") -> go.Figure:
        """
        Create interactive plot of EEG signals.
        
        Args:
            data: EEG data array (samples x channels)
            sampling_rate: Sampling rate in Hz
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_channels = min(data.shape[1], 10)  # Limit to 10 channels for readability
        time_axis = np.arange(data.shape[0]) / sampling_rate
        
        fig = make_subplots(
            rows=n_channels, 
            cols=1,
            subplot_titles=[f"Channel {i+1}" for i in range(n_channels)],
            vertical_spacing=0.02
        )
        
        for i in range(n_channels):
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=data[:, i],
                    mode='lines',
                    name=f'Channel {i+1}',
                    line=dict(color=self.color_palette[i % len(self.color_palette)]),
                    showlegend=False
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title=title,
            height=200 * n_channels,
            showlegend=False,
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (μV)"
        )
        
        return fig
    
    def plot_eeg_spectrum(self, data: np.ndarray,
                         sampling_rate: float = 250,
                         title: str = "EEG Power Spectrum") -> go.Figure:
        """
        Create power spectrum plot of EEG data.
        
        Args:
            data: EEG data array
            sampling_rate: Sampling rate in Hz
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_channels = min(data.shape[1], 6)  # Limit to 6 channels
        
        fig = make_subplots(
            rows=2, 
            cols=3,
            subplot_titles=[f"Channel {i+1}" for i in range(n_channels)],
            specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(2)]
        )
        
        for i in range(n_channels):
            row = i // 3 + 1
            col = i % 3 + 1
            
            # Calculate power spectrum
            fft_data = np.fft.fft(data[:, i])
            freqs = np.fft.fftfreq(len(data[:, i]), 1/sampling_rate)
            power = np.abs(fft_data) ** 2
            
            # Only plot positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_power = power[:len(power)//2]
            
            fig.add_trace(
                go.Scatter(
                    x=positive_freqs,
                    y=positive_power,
                    mode='lines',
                    name=f'Channel {i+1}',
                    line=dict(color=self.color_palette[i % len(self.color_palette)])
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=False,
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power"
        )
        
        return fig
    
    def plot_prediction_confidence(self, predictions: Dict[str, float],
                                 title: str = "Prediction Confidence") -> go.Figure:
        """
        Create bar chart of prediction confidences.
        
        Args:
            predictions: Dictionary of class names and their probabilities
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        classes = list(predictions.keys())
        probabilities = list(predictions.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=classes,
                y=probabilities,
                marker_color=self.color_palette[:len(classes)],
                text=[f"{p:.2%}" for p in probabilities],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Classes",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    def plot_model_comparison(self, model_scores: Dict[str, float],
                            title: str = "Model Comparison") -> go.Figure:
        """
        Create comparison chart of model scores.
        
        Args:
            model_scores: Dictionary of model names and their scores
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        models = list(model_scores.keys())
        scores = list(model_scores.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=scores,
                marker_color=self.color_palette[:len(models)],
                text=[f"{s:.2f}" for s in scores],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Models",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    def plot_data_quality_metrics(self, data: np.ndarray,
                                 title: str = "Data Quality Metrics") -> go.Figure:
        """
        Create visualization of data quality metrics.
        
        Args:
            data: EEG data array
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Calculate quality metrics
        metrics = {
            'Signal Power': np.var(data, axis=0),
            'Noise Level': np.mean(np.abs(np.diff(data, axis=0)), axis=0),
            'SNR': np.var(data, axis=0) / (np.mean(np.abs(np.diff(data, axis=0)), axis=0) + 1e-8)
        }
        
        fig = make_subplots(
            rows=1, 
            cols=3,
            subplot_titles=list(metrics.keys())
        )
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            fig.add_trace(
                go.Bar(
                    x=[f"Ch {j+1}" for j in range(len(values))],
                    y=values,
                    name=metric_name,
                    marker_color=self.color_palette[i]
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title=title,
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_dashboard(self, eeg_data: np.ndarray, 
                        predictions: Dict[str, Any],
                        model_scores: Dict[str, float]) -> go.Figure:
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Args:
            eeg_data: EEG data array
            predictions: Prediction results
            model_scores: Model comparison scores
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "EEG Signals", 
                "Power Spectrum",
                "Prediction Confidence", 
                "Model Comparison"
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # EEG Signals (top left)
        if len(eeg_data.shape) == 1:
            eeg_data = eeg_data.reshape(-1, 1)
        
        time_axis = np.arange(eeg_data.shape[0]) / 250
        for i in range(min(3, eeg_data.shape[1])):
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=eeg_data[:, i],
                    mode='lines',
                    name=f'Ch {i+1}',
                    line=dict(color=self.color_palette[i])
                ),
                row=1, col=1
            )
        
        # Power Spectrum (top right)
        if eeg_data.shape[1] > 0:
            fft_data = np.fft.fft(eeg_data[:, 0])
            freqs = np.fft.fftfreq(len(eeg_data[:, 0]), 1/250)
            power = np.abs(fft_data) ** 2
            positive_freqs = freqs[:len(freqs)//2]
            positive_power = power[:len(power)//2]
            
            fig.add_trace(
                go.Scatter(
                    x=positive_freqs,
                    y=positive_power,
                    mode='lines',
                    name='Power',
                    line=dict(color=self.color_palette[3])
                ),
                row=1, col=2
            )
        
        # Prediction Confidence (bottom left)
        if 'class_probabilities' in predictions:
            classes = list(predictions['class_probabilities'].keys())
            probabilities = list(predictions['class_probabilities'].values())
            
            fig.add_trace(
                go.Bar(
                    x=classes,
                    y=probabilities,
                    marker_color=self.color_palette[:len(classes)],
                    name='Confidence'
                ),
                row=2, col=1
            )
        
        # Model Comparison (bottom right)
        models = list(model_scores.keys())
        scores = list(model_scores.values())
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=scores,
                marker_color=self.color_palette[4:4+len(models)],
                name='Scores'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Agentic Disease Finder Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
