"""
Generate PDF Report from Non-Quantum System Evaluation Results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from utils.monitoring import log_metrics
import os
from models.simple_model_loader import SimpleModelLoader
import tensorflow as tf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

def load_eeg_data(filepath):
    """Load EEG data from CSV file."""
    try:
        data = pd.read_csv(filepath)
        if 'Time' in data.columns:
            data = data.drop(columns=['Time'])
        return data.values
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

def load_motor_imagery_data():
    """Load all motor imagery data files."""
    base_path = 'sample_data/'
    files = {
        'left_hand': 'motor_imagery_left_hand.csv',
        'right_hand': 'motor_imagery_right_hand.csv',
        'foot': 'motor_imagery_foot.csv',
        'tongue': 'motor_imagery_tongue.csv'
    }
    
    all_data = []
    all_labels = []
    
    for label_idx, (label, filename) in enumerate(files.items()):
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            data = load_eeg_data(filepath)
            if data is not None:
                data = data[:250] if len(data) >= 250 else data
                all_data.append(data)
                all_labels.extend([label_idx] * len(data))
    
    if not all_data:
        return None, None
    
    X = np.vstack(all_data)
    y = np.array(all_labels)
    
    return X, y

def load_parkinsons_data():
    """Load Parkinson's disease detection data."""
    healthy_path = 'sample_data/eeg_healthy.csv'
    parkinsons_path = 'sample_data/eeg_parkinsons.csv'
    
    X_healthy = load_eeg_data(healthy_path)
    X_parkinsons = load_eeg_data(parkinsons_path)
    
    if X_healthy is None or X_parkinsons is None:
        return None, None
    
    X_healthy = X_healthy[:250]
    X_parkinsons = X_parkinsons[:250]
    
    X = np.vstack([X_healthy, X_parkinsons])
    y = np.hstack([np.zeros(len(X_healthy)), np.ones(len(X_parkinsons))])
    
    return X, y

def preprocess_data(X, target_shape=(1000, 22)):
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

def evaluate_bci2a():
    """Evaluate BCI2A motor imagery classifier."""
    X, y = load_motor_imagery_data()
    if X is None:
        return None
    
    X_processed = preprocess_data(X, target_shape=(1000, 22))
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = SimpleModelLoader().models['bci2a_crdae']
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=0)
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm,
        'class_names': ['Left Hand', 'Right Hand', 'Foot', 'Tongue']
    }

def evaluate_eeg_pd():
    """Evaluate EEG Parkinson's Disease classifier."""
    X, y = load_parkinsons_data()
    if X is None:
        return None
    
    X_processed = preprocess_data(X, target_shape=(1000, 22))
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = SimpleModelLoader().models['eeg_pd']
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Class weights for robustness
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.array([0, 1]), y=y_train.astype(int)
    )
    class_weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}

    model.fit(
        X_train, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=0,
        class_weight=class_weight_dict
    )
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm,
        'class_names': ['Healthy', 'Parkinson\'s Disease']
    }

    # Log metrics
    try:
        log_metrics(
            model_name='eeg_pd',
            dataset_name='EEG Parkinsons',
            metrics={
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
            },
            confusion_matrix=cm.tolist(),
        )
    except Exception:
        pass

    return results

def create_title_page(pdf):
    """Create title page for PDF."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.8, 'Non-Quantum System Evaluation Report', 
            ha='center', va='center', fontsize=28, fontweight='bold')
    
    # Subtitle
    ax.text(0.5, 0.7, 'Agentic Disease Finder', 
            ha='center', va='center', fontsize=18, style='italic')
    
    # Report info
    ax.text(0.5, 0.5, 'Performance Metrics & Confusion Matrices', 
            ha='center', va='center', fontsize=16)
    
    ax.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%B %d, %Y")}', 
            ha='center', va='center', fontsize=12)
    
    # Models evaluated
    ax.text(0.5, 0.2, 'Models Evaluated:', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.15, '• BCI2A Motor Imagery Classifier', 
            ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.1, '• EEG Parkinson\'s Disease Classifier', 
            ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_summary_page(pdf, bci2a_results, eeg_pd_results):
    """Create summary page with performance metrics."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Performance Summary', 
            ha='center', va='top', fontsize=20, fontweight='bold')
    
    y_pos = 0.85
    
    # BCI2A Results
    if bci2a_results:
        ax.text(0.1, y_pos, 'BCI2A Motor Imagery Classifier', 
                fontsize=14, fontweight='bold')
        y_pos -= 0.05
        ax.text(0.1, y_pos, f'Accuracy: {bci2a_results["accuracy"]:.2%}', fontsize=12)
        y_pos -= 0.03
        ax.text(0.1, y_pos, f'Precision: {bci2a_results["precision"]:.2%}', fontsize=12)
        y_pos -= 0.03
        ax.text(0.1, y_pos, f'Recall: {bci2a_results["recall"]:.2%}', fontsize=12)
        y_pos -= 0.03
        ax.text(0.1, y_pos, f'F1-Score: {bci2a_results["f1"]:.2%}', fontsize=12)
        y_pos -= 0.05
    
    # EEG PD Results
    if eeg_pd_results:
        ax.text(0.1, y_pos, 'EEG Parkinson\'s Disease Classifier', 
                fontsize=14, fontweight='bold')
        y_pos -= 0.05
        ax.text(0.1, y_pos, f'Accuracy: {eeg_pd_results["accuracy"]:.2%}', fontsize=12)
        y_pos -= 0.03
        ax.text(0.1, y_pos, f'Precision: {eeg_pd_results["precision"]:.2%}', fontsize=12)
        y_pos -= 0.03
        ax.text(0.1, y_pos, f'Recall: {eeg_pd_results["recall"]:.2%}', fontsize=12)
        y_pos -= 0.03
        ax.text(0.1, y_pos, f'F1-Score: {eeg_pd_results["f1"]:.2%}', fontsize=12)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_confusion_matrix_page(pdf, results, title):
    """Create confusion matrix visualization."""
    if results is None:
        return
    
    fig = plt.figure(figsize=(11, 8.5))
    
    # Confusion matrix
    ax1 = plt.subplot(211)
    sns.heatmap(results['cm'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=results['class_names'], yticklabels=results['class_names'],
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title(f'{title}\nAccuracy: {results["accuracy"]:.2%}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Performance metrics bar chart
    ax2 = plt.subplot(212)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['accuracy'], results['precision'], results['recall'], results['f1']]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.8)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_detailed_metrics_page(pdf, results, title):
    """Create detailed metrics page."""
    if results is None:
        return
    
    from sklearn.metrics import classification_report
    
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, f'{title} - Detailed Metrics', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    
    # Classification report
    report = classification_report(
        results['y_test'], results['y_pred'], 
        target_names=results['class_names'],
        output_dict=True
    )
    
    # Create table
    y_start = 0.85
    row_height = 0.08
    
    # Header
    headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    x_positions = [0.15, 0.35, 0.55, 0.75, 0.9]
    
    for i, header in enumerate(headers):
        ax.text(x_positions[i], y_start, header, fontsize=11, fontweight='bold')
    
    y_start -= row_height
    ax.axhline(y=y_start, xmin=0.05, xmax=0.95, color='black', linewidth=1)
    
    # Rows
    y_pos = y_start - 0.03
    for class_name in results['class_names']:
        ax.text(x_positions[0], y_pos, class_name, fontsize=10)
        metrics = report[class_name]
        ax.text(x_positions[1], y_pos, f'{metrics["precision"]:.3f}', fontsize=10)
        ax.text(x_positions[2], y_pos, f'{metrics["recall"]:.3f}', fontsize=10)
        ax.text(x_positions[3], y_pos, f'{metrics["f1-score"]:.3f}', fontsize=10)
        ax.text(x_positions[4], y_pos, f'{int(metrics["support"])}', fontsize=10)
        y_pos -= row_height
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def generate_pdf_report():
    """Generate comprehensive PDF report."""
    print("=" * 70)
    print("Generating PDF Report for Non-Quantum System Evaluation")
    print("=" * 70)
    
    # Evaluate models
    print("\nEvaluating BCI2A Motor Imagery Classifier...")
    bci2a_results = evaluate_bci2a()
    
    print("Evaluating EEG Parkinson's Disease Classifier...")
    eeg_pd_results = evaluate_eeg_pd()
    
    # Create PDF
    output_file = 'non_quantum_evaluation_report.pdf'
    print(f"\nCreating PDF: {output_file}")
    
    with PdfPages(output_file) as pdf:
        # Title page
        print("Creating title page...")
        create_title_page(pdf)
        
        # Summary page
        print("Creating summary page...")
        create_summary_page(pdf, bci2a_results, eeg_pd_results)
        
        # BCI2A pages
        if bci2a_results:
            print("Creating BCI2A confusion matrix page...")
            create_confusion_matrix_page(pdf, bci2a_results, 'BCI2A Motor Imagery Classifier')
            print("Creating BCI2A detailed metrics page...")
            create_detailed_metrics_page(pdf, bci2a_results, 'BCI2A Motor Imagery Classifier')
        
        # EEG PD pages
        if eeg_pd_results:
            print("Creating EEG PD confusion matrix page...")
            create_confusion_matrix_page(pdf, eeg_pd_results, 'EEG Parkinson\'s Disease Classifier')
            print("Creating EEG PD detailed metrics page...")
            create_detailed_metrics_page(pdf, eeg_pd_results, 'EEG Parkinson\'s Disease Classifier')
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'Non-Quantum System Evaluation Report'
        d['Author'] = 'Agentic Disease Finder'
        d['Subject'] = 'Performance Evaluation of EEG Classification Models'
        d['Keywords'] = 'Machine Learning, EEG, Parkinson\'s Disease, Motor Imagery'
        d['CreationDate'] = datetime.now()
    
    print(f"\n✓ PDF report saved to: {output_file}")
    print("=" * 70)
    print("PDF Generation Complete!")
    print("=" * 70)

if __name__ == "__main__":
    generate_pdf_report()

