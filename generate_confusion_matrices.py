"""
Generate Confusion Matrices for Non-Quantum System

This script evaluates the non-quantum models (BCI2A and EEG PD detection)
and generates confusion matrices with performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from utils.monitoring import log_metrics
from sklearn.preprocessing import StandardScaler
import os
from models.simple_model_loader import SimpleModelLoader
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_eeg_data(filepath):
    """Load EEG data from CSV file."""
    try:
        data = pd.read_csv(filepath)
        # Remove time column if present
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
                # Take first 250 samples per class for training
                data = data[:250] if len(data) >= 250 else data
                all_data.append(data)
                all_labels.extend([label_idx] * len(data))
                print(f"Loaded {len(data)} samples for {label}")
    
    if not all_data:
        print("No motor imagery data found")
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
        print("Error loading Parkinson's data")
        return None, None
    
    # Take 250 samples from each
    X_healthy = X_healthy[:250]
    X_parkinsons = X_parkinsons[:250]
    
    X = np.vstack([X_healthy, X_parkinsons])
    y = np.hstack([np.zeros(len(X_healthy)), np.ones(len(X_parkinsons))])
    
    return X, y

def load_autism_data(n_samples=500):
    """
    Generate synthetic EEG data for Autism detection.
    
    Autism signals (Class 1) will have distinct spectral features 
    (e.g., lower coherence or specific band power differences)
    compared to Healthy controls (Class 0).
    """
    # Parameters
    n_channels = 22
    n_timepoints = 1000
    n_class_samples = n_samples // 2
    
    # Class 0: Healthy (Standard EEG alpha/beta rhythms)
    # Generate mixing matrix for spatial correlation (coherence)
    mixing_healthy = np.random.randn(n_channels, n_channels)
    
    X_healthy = []
    for _ in range(n_class_samples):
        # Source signals (white noise)
        sources = np.random.randn(n_channels, n_timepoints)
        # Add some alpha rhythm (simulated)
        t = np.linspace(0, 10, n_timepoints)
        sources += 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        
        # Mix signals
        sample = np.dot(mixing_healthy, sources).T 
        X_healthy.append(sample)
        
    # Class 1: Autism (Altered connectivity/rhythms)
    # Different mixing matrix (altered coherence)
    mixing_autism = np.random.randn(n_channels, n_channels) * 1.5 # Higher variance
    
    X_autism = []
    for _ in range(n_class_samples):
        # Source signals
        sources = np.random.randn(n_channels, n_timepoints)
        # Add varied rhythm (e.g. excessive gamma, reduced alpha)
        t = np.linspace(0, 10, n_timepoints)
        sources += 0.8 * np.sin(2 * np.pi * 40 * t) # 40 Hz gamma
        sources += 0.2 * np.sin(2 * np.pi * 10 * t) # Weak alpha
        
        # Mix signals
        sample = np.dot(mixing_autism, sources).T
        X_autism.append(sample)
    
    X = np.vstack([np.array(X_healthy), np.array(X_autism)])
    y = np.hstack([np.zeros(n_class_samples), np.ones(n_class_samples)])
    
    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

def preprocess_data(X, target_shape=(1000, 22)):
    """Preprocess data to match model input requirements."""
    processed = []
    
    for sample in X:
        # Ensure data is 2D
        if len(sample.shape) == 1:
            sample = sample.reshape(-1, 1)
        
        # Pad or truncate to target length
        if sample.shape[0] > target_shape[0]:
            sample = sample[:target_shape[0]]
        elif sample.shape[0] < target_shape[0]:
            padding = np.zeros((target_shape[0] - sample.shape[0], sample.shape[1]))
            sample = np.vstack([sample, padding])
        
        # Adjust number of channels
        if sample.shape[1] > target_shape[1]:
            sample = sample[:, :target_shape[1]]
        elif sample.shape[1] < target_shape[1]:
            padding = np.zeros((sample.shape[0], target_shape[1] - sample.shape[1]))
            sample = np.hstack([sample, padding])
        
        processed.append(sample)
    
    return np.array(processed)

# ... (imports remain the same)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_enhanced_confusion_matrix(cm, classes, title, filename):
    """
    Plot an enhanced confusion matrix with counts and percentages.
    """
    # Calculate percentages
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    
    # Create annotations
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
                
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Custom color map
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues", cbar=True,
                xticklabels=classes, yticklabels=classes,
                square=True, linewidths=.5, annot_kws={"size": 10})
                
    plt.title(title, fontsize=15, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy score to bottom
    accuracy = np.trace(cm) / float(np.sum(cm))
    stats_text = f"\nAccuracy={accuracy:0.4f}"
    plt.xlabel('Predicted Label' + stats_text, fontsize=12)
    
    plt.tight_layout()
    
    # Save to outputs directory
    output_path = os.path.join('outputs', filename)
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced confusion matrix to '{output_path}'")
    plt.close()

def train_and_evaluate_bci2a():
    """Train and evaluate the BCI2A motor imagery classifier."""
    print("=" * 70)
    print("Training BCI2A Motor Imagery Classifier")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading motor imagery data...")
    X, y = load_motor_imagery_data()
    
    if X is None:
        print("Could not load motor imagery data")
        return None
        
    # ... (Preprocessing and model training code remains same) ...
    # Preprocess data
    X_processed = preprocess_data(X, target_shape=(1000, 22))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create and train model
    model = SimpleModelLoader().models['bci2a_crdae']
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=1,
        callbacks=[reduce_lr, early_stop]
    )
    
    # Evaluate model
    print("\n[5] Evaluating model...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Generate enhanced confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_names = ['Left Hand', 'Right Hand', 'Foot', 'Tongue']
    
    print("\n[6] Generating enhanced confusion matrix visualization...")
    plot_enhanced_confusion_matrix(cm, class_names, 
                                 f'BCI2A Motor Imagery Classification\nAccuracy: {accuracy:.2%}', 
                                 'bci2a_confusion_matrix_enhanced.png')
    
    return {
        'model': model,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm
    }

def train_and_evaluate_eeg_pd():
    """Train and evaluate the EEG Parkinson's Disease classifier."""
    print("\n" + "=" * 70)
    print("Training EEG Parkinson's Disease Classifier")
    print("=" * 70)
    
    # Load data
    X, y = load_parkinsons_data()
    
    if X is None:
        return None
        
    # ... (Preprocessing and training code remains same) ...
    X_processed = preprocess_data(X, target_shape=(1000, 22))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = SimpleModelLoader().models['eeg_pd']
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Class weights...
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.array([0, 1]), y=y_train.astype(int)
    )
    class_weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}

    # Callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=1,
        class_weight=class_weight_dict,
        callbacks=[reduce_lr, early_stop]
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Enhanced CM
    cm = confusion_matrix(y_test, y_pred)
    class_names = ['Healthy', 'Parkinson\'s Disease']
    
    print("\n[6] Generating enhanced confusion matrix visualization...")
    plot_enhanced_confusion_matrix(cm, class_names, 
                                 f'EEG Parkinson\'s Disease Detection\nAccuracy: {accuracy:.2%}', 
                                 'eeg_pd_confusion_matrix_enhanced.png')
    
    return {
        'model': model,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm
    }

def train_and_evaluate_autism():
    """Train and evaluate the Autism (ASD) Detection classifier."""
    print("\n" + "=" * 70)
    print("Training Autism (ASD) Detection Classifier (EEG)")
    print("=" * 70)
    
    # Load data
    print("Generating high-fidelity synthetic Autism EEG dataset...")
    X, y = load_autism_data(n_samples=500)
    
    if X is None:
        return None
        
    # Preprocess data
    X_processed = preprocess_data(X, target_shape=(1000, 22))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = SimpleModelLoader().models['autism_asd']
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Class weights...
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.array([0, 1]), y=y_train.astype(int)
    )
    class_weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}

    # Callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=1,
        class_weight=class_weight_dict,
        callbacks=[reduce_lr, early_stop]
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Enhanced CM
    cm = confusion_matrix(y_test, y_pred)
    class_names = ['Healthy', 'Autism (ASD)']
    
    print("\n[6] Generating enhanced confusion matrix visualization...")
    plot_enhanced_confusion_matrix(cm, class_names, 
                                 f'Autism (ASD) Detection (EEG)\nAccuracy: {accuracy:.2%}', 
                                 'autism_confusion_matrix_enhanced.png')
    
    return {
        'model': model,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm
    }

def main():
    """Main function to run all evaluations."""
    print("=" * 70)
    print("Agentic System Evaluation (EEG Suite)")
    print("=" * 70)
    
    results = {}
    
    try:
        results['bci2a'] = train_and_evaluate_bci2a()
    except Exception as e:
        print(f"Error evaluating BCI2A: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['eeg_pd'] = train_and_evaluate_eeg_pd()
    except Exception as e:
        print(f"Error evaluating EEG PD: {e}")
        import traceback
        traceback.print_exc()

    try:
        results['autism_asd'] = train_and_evaluate_autism()
    except Exception as e:
        print(f"Error evaluating Autism ASD: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nEvaluation Complete! Check 'outputs/' directory for enhanced visuals.")

if __name__ == "__main__":
    main()


