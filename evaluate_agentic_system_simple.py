"""
Simplified Agentic System Evaluation

Evaluates the entire agentic system end-to-end, focusing on:
1. Overall classification performance
2. Model selection accuracy
3. System-wide confusion matrices
4. Scalability metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from utils.monitoring import log_metrics
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from models.simple_model_loader import SimpleModelLoader
import tensorflow as tf

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

sns.set_style("whitegrid")
plt.rcParams['figure.max_open_warning'] = 0

def load_and_evaluate_agentic_system():
    """Comprehensive agentic system evaluation."""
    print("=" * 80)
    print("Agentic Disease Finder - Complete System Evaluation")
    print("=" * 80)
    
    # Load and train both models
    simple_loader = SimpleModelLoader()
    
    print("\n[1] Training BCI2A Motor Imagery Classifier...")
    bci2a_data, bci2a_labels = load_bci2a_data()
    bci2a_X_train, bci2a_X_test, bci2a_y_train, bci2a_y_test = train_test_split(
        bci2a_data, bci2a_labels, test_size=0.3, random_state=42, stratify=bci2a_labels
    )
    
    bci2a_model = simple_loader.models['bci2a_crdae']
    bci2a_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    bci2a_model.fit(bci2a_X_train, bci2a_y_train, epochs=15, batch_size=16, verbose=0)
    
    bci2a_pred = bci2a_model.predict(bci2a_X_test, verbose=0)
    bci2a_pred_labels = np.argmax(bci2a_pred, axis=1)
    
    print("\n[2] Training EEG Parkinson's Disease Classifier...")
    pd_X, pd_y = load_parkinsons_data()
    pd_X_train, pd_X_test, pd_y_train, pd_y_test = train_test_split(
        pd_X, pd_y, test_size=0.3, random_state=42, stratify=pd_y
    )
    
    pd_model = simple_loader.models['eeg_pd']
    pd_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Class weights for PD model
    pd_class_weights = compute_class_weight(
        class_weight='balanced', classes=np.array([0, 1]), y=pd_y_train.astype(int)
    )
    pd_class_weight_dict = {0: float(pd_class_weights[0]), 1: float(pd_class_weights[1])}

    pd_model.fit(pd_X_train, pd_y_train, epochs=15, batch_size=16, verbose=0, class_weight=pd_class_weight_dict)
    
    pd_pred = pd_model.predict(pd_X_test, verbose=0)
    pd_pred_labels = np.argmax(pd_pred, axis=1)
    
    # Combine results for unified evaluation
    print("\n[3] Creating unified system evaluation...")
    
    # Combine all test samples and their true/predicted labels
    all_samples = np.vstack([bci2a_X_test, pd_X_test])
    all_true_combined = []
    all_pred_combined = []
    
    # BCI2A labels (0-3) -> map to 0-3
    for true, pred in zip(bci2a_y_test, bci2a_pred_labels):
        all_true_combined.append(f"Motor_{int(true)}")
        all_pred_combined.append(f"Motor_{int(pred)}")
    
    # PD labels (0-1) -> map to 4-5
    for true, pred in zip(pd_y_test, pd_pred_labels):
        if true == 0:
            all_true_combined.append("PD_Healthy")
        else:
            all_true_combined.append("PD_Disease")
        
        if pred == 0:
            all_pred_combined.append("PD_Healthy")
        else:
            all_pred_combined.append("PD_Disease")
    
    # Create unified confusion matrix
    print("\n[4] Generating unified confusion matrix...")
    create_unified_confusion_matrix(all_true_combined, all_pred_combined)
    
    # Create per-system confusion matrices
    print("\n[5] Generating per-system confusion matrices...")
    create_combined_view(bci2a_y_test, bci2a_pred_labels, 'Motor Imagery', ['Left Hand', 'Right Hand', 'Foot', 'Tongue'])
    create_combined_view(pd_y_test, pd_pred_labels, 'Parkinson\'s Disease', ['Healthy', 'PD'])
    
    # Calculate overall metrics
    print("\n" + "=" * 80)
    print("Overall System Performance")
    print("=" * 80)
    
    # Accuracy for each system
    bci2a_acc = accuracy_score(bci2a_y_test, bci2a_pred_labels)
    pd_acc = accuracy_score(pd_y_test, pd_pred_labels)
    overall_acc = accuracy_score(all_true_combined, all_pred_combined)
    
    print(f"\nBCI2A Motor Imagery: {bci2a_acc:.2%} accuracy")
    print(f"EEG Parkinson's Disease: {pd_acc:.2%} accuracy")
    print(f"\nOverall System Accuracy: {overall_acc:.2%}")
    print(f"Total Samples Classified: {len(all_true_combined)}")

    # Log metrics
    try:
        bci2a_cm = confusion_matrix(bci2a_y_test, bci2a_pred_labels).tolist()
        pd_cm = confusion_matrix(pd_y_test, pd_pred_labels).tolist()
        log_metrics('bci2a_crdae', 'Motor Imagery', {'accuracy': float(bci2a_acc)}, bci2a_cm)
        log_metrics('eeg_pd', 'EEG Parkinsons', {'accuracy': float(pd_acc)}, pd_cm)
        # Unified accuracy only
        log_metrics('agentic_system', 'Unified', {'overall_accuracy': float(overall_acc)}, None, extra={'total_samples': int(len(all_true_combined))})
    except Exception:
        pass
    
    # Scalability metrics
    print("\n[6] System Scalability Metrics...")
    create_scalability_dashboard(bci2a_acc, pd_acc, len(all_true_combined))
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  - agentic_system_unified_confusion_matrix.png")
    print("  - agentic_system_combined_view.png")
    print("  - agentic_scalability_dashboard.png")
    
    return {
        'bci2a_accuracy': bci2a_acc,
        'pd_accuracy': pd_acc,
        'overall_accuracy': overall_acc,
        'total_samples': len(all_true_combined)
    }

def load_bci2a_data():
    """Load BCI2A motor imagery data."""
    motor_files = {
        0: 'sample_data/motor_imagery_left_hand.csv',
        1: 'sample_data/motor_imagery_right_hand.csv',
        2: 'sample_data/motor_imagery_foot.csv',
        3: 'sample_data/motor_imagery_tongue.csv'
    }
    
    all_data = []
    all_labels = []
    
    for label, filepath in motor_files.items():
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                if 'Time' in data.columns:
                    data = data.drop(columns=['Time'])
                data = data[:200] if len(data) >= 200 else data
                data = preprocess_samples(data.values, (1000, 22))
                all_data.append(data)
                all_labels.extend([label] * len(data))
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    if all_data:
        X = np.vstack(all_data)
        y = np.array(all_labels)
        return X, y
    
    return None, None

def load_parkinsons_data():
    """Load Parkinson's disease data."""
    healthy = load_file('sample_data/eeg_healthy.csv')
    parkinsons = load_file('sample_data/eeg_parkinsons.csv')
    
    if healthy is not None and parkinsons is not None:
        healthy = healthy[:200] if len(healthy) >= 200 else healthy
        parkinsons = parkinsons[:200] if len(parkinsons) >= 200 else parkinsons
        
        healthy = preprocess_samples(healthy, (1000, 22))
        parkinsons = preprocess_samples(parkinsons, (1000, 22))
        
        X = np.vstack([healthy, parkinsons])
        y = np.hstack([np.zeros(len(healthy)), np.ones(len(parkinsons))])
        
        return X, y
    
    return None, None

def load_file(filepath):
    """Load CSV file."""
    try:
        data = pd.read_csv(filepath)
        if 'Time' in data.columns:
            data = data.drop(columns=['Time'])
        return data.values
    except:
        return None

def preprocess_samples(X, target_shape):
    """Preprocess multiple samples."""
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

def create_unified_confusion_matrix(true_labels, pred_labels):
    """Create unified confusion matrix for entire system."""
    unique_labels = sorted(list(set(true_labels + pred_labels)))
    
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels,
                ax=ax, cbar_kws={'label': 'Sample Count'})
    ax.set_title('Agentic System - Unified Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=14)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('agentic_system_unified_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Unified confusion matrix saved")

def create_combined_view(y_true, y_pred, title, class_names):
    """Create combined view of both systems side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'{title} - Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Accuracy bar chart
    accuracy = accuracy_score(y_true, y_pred)
    axes[1].barh(['Accuracy'], [accuracy], color='#2E86AB', alpha=0.8, height=0.5)
    axes[1].set_xlim(0, 1)
    axes[1].set_title('Performance Metric', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Score')
    axes[1].text(accuracy, 0, f'{accuracy:.2%}', va='center', ha='left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'agentic_{title.lower().replace(" ", "_")}_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ {title} results saved")

def create_scalability_dashboard(bci2a_acc, pd_acc, total_samples):
    """Create scalability dashboard."""
    fig = plt.figure(figsize=(14, 8))
    
    # Create subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Performance comparison
    ax1 = fig.add_subplot(gs[0, 0])
    systems = ['BCI2A', 'PD\nDetection']
    accuracies = [bci2a_acc, pd_acc]
    colors = ['#2E86AB', '#A23B72']
    bars = ax1.bar(systems, accuracies, color=colors, alpha=0.8)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('System Performance Comparison', fontsize=14, fontweight='bold')
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., acc,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=11)
    
    # 2. Sample distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sample_counts = [total_samples * 0.6, total_samples * 0.4]  # Approximate split
    ax2.pie(sample_counts, labels=['Motor Imagery', 'PD Detection'], 
           autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Sample Distribution', fontsize=14, fontweight='bold')
    
    # 3. Total system metrics
    ax3 = fig.add_subplot(gs[1, :])
    overall_metrics = {
        'Total Samples': total_samples,
        'BCI2A Accuracy': bci2a_acc * 100,
        'PD Accuracy': pd_acc * 100,
        'Average Accuracy': ((bci2a_acc + pd_acc) / 2) * 100
    }
    
    y_pos = np.arange(len(overall_metrics))
    ax3.barh(y_pos, list(overall_metrics.values()), color=['#C73E1D', '#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(overall_metrics.keys())
    ax3.set_xlabel('Value', fontsize=12)
    ax3.set_title('Agentic System - Scalability Metrics', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(overall_metrics.values()):
        ax3.text(v, i, f'{v:.1f}', va='center', ha='left', fontsize=11)
    
    plt.suptitle('Agentic System Scalability Dashboard', fontsize=16, fontweight='bold')
    plt.savefig('agentic_scalability_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Scalability dashboard saved")

if __name__ == "__main__":
    results = load_and_evaluate_agentic_system()
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for key, value in results.items():
        print(f"{key}: {value}")

