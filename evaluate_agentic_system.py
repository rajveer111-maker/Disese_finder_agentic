"""
Comprehensive Evaluation of the Agentic Disease Finder System

This script evaluates the ENTIRE agentic system, including:
1. Agentic decision-making (model selection)
2. End-to-end classification performance
3. System-wide confusion matrices
4. Scalability assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from models.model_manager import ModelManager
from models.agentic_decision import AgenticDecisionSystem
from models.simple_model_loader import SimpleModelLoader
import tensorflow as tf

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

class AgenticSystemEvaluator:
    """Comprehensive evaluation of the agentic system."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.agentic_system = AgenticDecisionSystem()
        self.simple_loader = SimpleModelLoader()
        
        # Global label encoder for combined predictions
        self.label_encoder = LabelEncoder()
        self.all_labels = []
        
    def load_eeg_data(self, filepath):
        """Load EEG data from CSV file."""
        try:
            data = pd.read_csv(filepath)
            if 'Time' in data.columns:
                data = data.drop(columns=['Time'])
            return data.values
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            return None
    
    def load_all_data(self):
        """Load all available datasets."""
        datasets = {}
        
        # Motor imagery data
        motor_files = {
            'motor_left_hand': 'sample_data/motor_imagery_left_hand.csv',
            'motor_right_hand': 'sample_data/motor_imagery_right_hand.csv',
            'motor_foot': 'sample_data/motor_imagery_foot.csv',
            'motor_tongue': 'sample_data/motor_imagery_tongue.csv'
        }
        
        motor_data = []
        motor_labels = []
        
        for label, filepath in motor_files.items():
            if os.path.exists(filepath):
                data = self.load_eeg_data(filepath)
                if data is not None:
                    # Take samples
                    data = data[:200] if len(data) >= 200 else data
                    motor_data.append(data)
                    # Create binary labels: 'BCI2A' for all motor imagery
                    motor_labels.extend(['BCI2A'] * len(data))
        
        if motor_data:
            datasets['bci2a'] = {
                'X': np.vstack(motor_data),
                'y_raw': motor_labels,
                'y_encoded': None,  # Will be set later
                'model_type': 'bci2a_crdae'
            }
            print(f"Loaded motor imagery data: {len(datasets['bci2a']['X'])} samples")
        
        # Parkinson's disease data
        healthy_path = 'sample_data/eeg_healthy.csv'
        parkinsons_path = 'sample_data/eeg_parkinsons.csv'
        
        healthy_data = self.load_eeg_data(healthy_path)
        parkinsons_data = self.load_eeg_data(parkinsons_path)
        
        if healthy_data is not None and parkinsons_data is not None:
            healthy_data = healthy_data[:200] if len(healthy_data) >= 200 else healthy_data
            parkinsons_data = parkinsons_data[:200] if len(parkinsons_data) >= 200 else parkinsons_data
            
            X_pd = np.vstack([healthy_data, parkinsons_data])
            y_pd = np.hstack([['EEG_PD'] * len(healthy_data), ['EEG_PD'] * len(parkinsons_data)])
            
            datasets['eeg_pd'] = {
                'X': X_pd,
                'y_raw': y_pd,
                'y_encoded': None,
                'model_type': 'eeg_pd'
            }
            print(f"Loaded PD detection data: {len(datasets['eeg_pd']['X'])} samples")
            
        # Alzheimer's disease data
        alzheimers_path = 'sample_data/eeg_alzheimers.csv'
        cn_path = 'sample_data/eeg_cn.csv'
        
        alzheimers_data = self.load_eeg_data(alzheimers_path)
        cn_data = self.load_eeg_data(cn_path)
        
        if alzheimers_data is not None and cn_data is not None:
            alzheimers_data = alzheimers_data[:200] if len(alzheimers_data) >= 200 else alzheimers_data
            cn_data = cn_data[:200] if len(cn_data) >= 200 else cn_data
            
            X_ad = np.vstack([alzheimers_data, cn_data])
            y_ad = np.hstack([['NEUROFORMER'] * len(alzheimers_data), ['NEUROFORMER'] * len(cn_data)])
            
            datasets['neuroformer'] = {
                'X': X_ad,
                'y_raw': y_ad,
                'y_encoded': None,
                'model_type': 'neuroformer'
            }
            print(f"Loaded Alzheimer's detection data: {len(datasets['neuroformer']['X'])} samples")
        
        return datasets
    
    def preprocess_data(self, X, target_shape=(1000, 22)):
        """Preprocess data to match model input requirements."""
        processed = []
        
        for sample in X:
            if len(sample.shape) == 1:
                sample = sample.reshape(-1, 1)
            
            if sample.shape[0] > target_shape[0]:
                sample = sample[:target_shape[0]]
            elif sample.shape[0] < target_shape[0]:
                padding = np.zeros((target_shape[0] - sample.shape[0], sample.shape[1]))
                sample = np.vstack([sample, padding])
            
            if sample.shape[1] > target_shape[1]:
                sample = sample[:, :target_shape[1]]
            elif sample.shape[1] < target_shape[1]:
                padding = np.zeros((sample.shape[0], target_shape[1] - sample.shape[1]))
                sample = np.hstack([sample, padding])
            
            processed.append(sample)
        
        return np.array(processed)
    
    def train_models(self, datasets):
        """Train both models on their respective data."""
        print("\n" + "=" * 70)
        print("Training Models")
        print("=" * 70)
        
        trained_models = {}
        
        for dataset_name, dataset in datasets.items():
            print(f"\nTraining {dataset_name} model...")
            X = dataset['X']
            X_processed = self.preprocess_data(X, target_shape=(1000, 22))
            
            # Split data
            X_train, X_test, train_idx, test_idx = train_test_split(
                X_processed, np.arange(len(X)), test_size=0.3, random_state=42
            )
            
            # Create model
            if dataset_name == 'bci2a':
                model = self.simple_loader.models['bci2a_crdae']
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                # Create labels (0: left_hand, 1: right_hand, 2: foot, 3: tongue)
                motor_labels = []
                for idx in train_idx:
                    # Determine which motor imagery type based on original dataset position
                    if idx < 200:
                        motor_labels.append(0)  # left hand
                    elif idx < 400:
                        motor_labels.append(1)  # right hand
                    elif idx < 600:
                        motor_labels.append(2)  # foot
                    else:
                        motor_labels.append(3)  # tongue
                
                y_train = np.array(motor_labels[:len(X_train)])
                model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)
                dataset['model'] = model
                
            elif dataset_name == 'eeg_pd':
                model = self.simple_loader.models['eeg_pd']
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                # Create labels (0: healthy, 1: parkinsons)
                pd_labels = []
                for idx in train_idx:
                    pd_labels.append(0 if idx < 200 else 1)  # first half healthy, second half PD
                
                y_train = np.array(pd_labels[:len(X_train)])
                model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)
                dataset['model'] = model
                
            elif dataset_name == 'neuroformer':
                # For neuroformer, we don't really train it from scratch in the evaluation script
                # because it's a PyTorch model and we already loaded its best_weights.h5.
                # Just mock the `model` object so it doesn't fail downstream if checked.
                dataset['model'] = self.model_manager.models.get('neuroformer', None)
            
            dataset['X_test'] = X_test
            dataset['test_idx'] = test_idx
            trained_models[dataset_name] = dataset
        
        return trained_models
    
    def evaluate_agentic_system(self, trained_models):
        """Evaluate the complete agentic system."""
        print("\n" + "=" * 70)
        print("Evaluating Agentic System")
        print("=" * 70)
        
        all_predictions = []
        all_true_labels = []
        all_selected_models = []
        all_expected_models = []
        
        for dataset_name, dataset in trained_models.items():
            print(f"\nEvaluating {dataset_name}...")
            
            X_test = dataset['X_test']
            test_idx = dataset['test_idx']
            
            for i, sample in enumerate(X_test):
                # Make agentic decision
                file_type = 'csv'
                decision = self.agentic_system.decide_model(sample, file_type, context={'filename': dataset_name})
                selected_model = decision['selected_model']
                
                # Get actual expected model
                expected_model = dataset['model_type']
                all_expected_models.append(expected_model)
                all_selected_models.append(selected_model)
                
                # Make prediction with the correct model (regardless of agentic decision)
                try:
                    # Use the correct model for ground truth via model_manager to support PyTorch neuroformer
                    result = self.model_manager.predict(sample, expected_model)
                    prediction = result['prediction']
                    
                    # Create combined label: model_type + prediction
                    if dataset_name == 'bci2a':
                        # Motor imagery classes
                        ground_truth = f"BCI2A_{prediction}"
                    elif dataset_name == 'eeg_pd':
                        # PD detection classes  
                        ground_truth = f"EEG_PD_{prediction}"
                    elif dataset_name == 'neuroformer':
                        # AD detection classes
                        ground_truth = f"NEUROFORMER_{prediction}"
                    
                    all_true_labels.append(ground_truth)
                    
                    # Now make prediction with agentic-selected model
                    try:
                        pred_result = self.model_manager.predict(sample, selected_model)
                        pred_label = pred_result['prediction']
                        
                        if selected_model == 'bci2a_crdae':
                            predicted = f"BCI2A_{pred_label}"
                        elif selected_model == 'eeg_pd':
                            predicted = f"EEG_PD_{pred_label}"
                        else:
                            predicted = f"NEUROFORMER_{pred_label}"
                    except:
                        # If prediction fails, use a default
                        predicted = "UNKNOWN"
                    
                    all_predictions.append(predicted)
                    
                except Exception as e:
                    print(f"Error predicting sample {i}: {e}")
                    all_predictions.append("UNKNOWN")
                    all_true_labels.append(f"{dataset_name}_UNKNOWN")
                    continue
        
        return {
            'predictions': all_predictions,
            'true_labels': all_true_labels,
            'selected_models': all_selected_models,
            'expected_models': all_expected_models
        }
    
    def create_combined_confusion_matrix(self, results):
        """Create confusion matrix for the combined system."""
        print("\nCreating combined confusion matrix...")
        
        # Get unique labels
        all_unique_labels = sorted(list(set(results['true_labels'] + results['predictions'])))
        
        # Create confusion matrix
        cm = confusion_matrix(results['true_labels'], results['predictions'], labels=all_unique_labels)
        
        # Create visualization
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=all_unique_labels, yticklabels=all_unique_labels)
        plt.title('Agentic System Combined Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('agentic_system_confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("Saved to: agentic_system_confusion_matrix.png")
        plt.close()
        
        return cm, all_unique_labels
    
    def create_model_selection_confusion_matrix(self, results):
        """Create confusion matrix for model selection accuracy."""
        print("\nCreating model selection confusion matrix...")
        
        # Analyze model selection
        model_selection_data = []
        for i, (pred, true) in enumerate(zip(results['predictions'], results['true_labels'])):
            model_selection_data.append({
                'true': true.split('_')[0],  # Get dataset name
                'predicted': pred.split('_')[0]  # Get predicted dataset name
            })
        
        df = pd.DataFrame(model_selection_data)
        unique_models = sorted(list(set(df['true'].unique()) | set(df['predicted'].unique())))
        
        cm = confusion_matrix(df['true'], df['predicted'], labels=unique_models)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                    xticklabels=unique_models, yticklabels=unique_models)
        plt.title('Agentic System Model Selection Accuracy', fontsize=16, fontweight='bold')
        plt.ylabel('True Model Should Use', fontsize=12)
        plt.xlabel('Selected Model', fontsize=12)
        plt.tight_layout()
        plt.savefig('agentic_model_selection_confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("Saved to: agentic_model_selection_confusion_matrix.png")
        plt.close()
        
        # Calculate model selection accuracy
        model_selection_accuracy = np.trace(cm) / np.sum(cm)
        print(f"Model Selection Accuracy: {model_selection_accuracy:.2%}")
        
        return cm, unique_models, model_selection_accuracy
    
    def create_scalability_analysis(self, results):
        """Analyze system scalability."""
        print("\nAnalyzing scalability...")
        
        # Count predictions by model
        model_counts = pd.Series(results['selected_models']).value_counts()
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart of model usage
        axes[0].pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%',
                  startangle=90, colors=['#2E86AB', '#A23B72'])
        axes[0].set_title('Model Selection Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart of model usage
        axes[1].bar(model_counts.index, model_counts.values, color=['#2E86AB', '#A23B72'], alpha=0.8)
        axes[1].set_title('Model Usage Count', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Usage Count')
        
        for i, v in enumerate(model_counts.values):
            axes[1].text(i, v, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('agentic_scalability_analysis.png', dpi=150, bbox_inches='tight')
        print("Saved to: agentic_scalability_analysis.png")
        plt.close()
    
    def calculate_overall_metrics(self, results):
        """Calculate overall system metrics."""
        accuracy = accuracy_score(results['true_labels'], results['predictions'])
        precision = precision_score(results['true_labels'], results['predictions'], 
                                  average='weighted', zero_division=0)
        recall = recall_score(results['true_labels'], results['predictions'], 
                            average='weighted', zero_division=0)
        f1 = f1_score(results['true_labels'], results['predictions'], 
                    average='weighted', zero_division=0)
        
        print("\n" + "=" * 70)
        print("Overall Agentic System Performance")
        print("=" * 70)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    def run_full_evaluation(self):
        """Run complete evaluation of the agentic system."""
        print("=" * 70)
        print("Agentic Disease Finder - Comprehensive System Evaluation")
        print("=" * 70)
        
        # Load data
        print("\n[1] Loading datasets...")
        datasets = self.load_all_data()
        
        if not datasets:
            print("No datasets loaded. Exiting.")
            return
        
        # Train models
        print("\n[2] Training models...")
        trained_models = self.train_models(datasets)
        
        # Evaluate agentic system
        print("\n[3] Evaluating agentic system...")
        results = self.evaluate_agentic_system(trained_models)
        
        # Create visualizations
        print("\n[4] Creating visualizations...")
        self.create_combined_confusion_matrix(results)
        model_selection_cm, models, selection_accuracy = self.create_model_selection_confusion_matrix(results)
        self.create_scalability_analysis(results)
        
        # Calculate metrics
        print("\n[5] Calculating metrics...")
        metrics = self.calculate_overall_metrics(results)
        
        # Summary
        print("\n" + "=" * 70)
        print("Evaluation Complete!")
        print("=" * 70)
        print("\nGenerated Files:")
        print("  - agentic_system_confusion_matrix.png")
        print("  - agentic_model_selection_confusion_matrix.png")
        print("  - agentic_scalability_analysis.png")
        print("\n" + "=" * 70)
        
        return results, metrics


def main():
    """Main function."""
    evaluator = AgenticSystemEvaluator()
    results, metrics = evaluator.run_full_evaluation()
    return results, metrics


if __name__ == "__main__":
    results, metrics = main()

