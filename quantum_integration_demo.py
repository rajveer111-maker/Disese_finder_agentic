"""
Quantum Computing Integration Demo for Medical Disease Finder

This script demonstrates how to integrate quantum machine learning
with the existing Agentic Disease Finder system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantum_classifier import QuantumDiseaseClassifier, create_quantum_bell_state, execute_bell_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns


def load_medical_data(filepath):
    """Load medical data (EEG, etc.) from CSV files."""
    try:
        data = pd.read_csv(filepath)
        print(f"Loaded data: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def prepare_quantum_classification_data(data, label_column='label', n_samples=None):
    """
    Prepare data for quantum classification.
    
    Parameters:
    -----------
    data : DataFrame
        Input data
    label_column : str
        Name of the label column
    n_samples : int, optional
        Number of samples to use (for testing)
    
    Returns:
    --------
    X, y : arrays
        Features and labels
    """
    if n_samples and len(data) > n_samples:
        data = data.sample(n=n_samples, random_state=42)
    
    y = data[label_column].values
    X = data.drop(columns=[label_column]).values
    
    return X, y


def run_quantum_medical_classification():
    """
    Run quantum classification on medical data.
    """
    print("=" * 70)
    print("Quantum Medical Disease Finder - Integration Demo")
    print("=" * 70)
    
    # 1. Demonstrate quantum computing basics
    print("\n[1] Quantum Computing Basics:")
    print("-" * 70)
    print("Creating a Bell State quantum circuit...")
    bell_circuit = create_quantum_bell_state()
    print("\nQuantum Circuit:")
    print(bell_circuit)
    
    print("\nExecuting Bell State Circuit...")
    bell_results = execute_bell_state()
    
    # 2. Try loading real medical data
    print("\n[2] Loading Medical Data:")
    print("-" * 70)
    
    # Try to load EEG data
    try:
        eeg_data = load_medical_data('sample_data/eeg_parkinsons.csv')
        if eeg_data is not None:
            print("✓ Successfully loaded Parkinson's EEG data")
            print(f"  Shape: {eeg_data.shape}")
            print(f"  Columns: {list(eeg_data.columns)}")
        else:
            print("✗ Could not load EEG data, generating synthetic data")
            eeg_data = None
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        eeg_data = None
    
    # 3. Generate synthetic medical data if needed
    if eeg_data is None:
        print("\n[3] Generating Synthetic Medical Data:")
        print("-" * 70)
        print("Generating synthetic EEG-like data for demonstration...")
        
        np.random.seed(42)
        n_samples = 150
        n_features = 10
        
        # Generate features (EEG-like signals)
        X = np.random.randn(n_samples, n_features)
        
        # Generate labels based on some pattern (simulating disease detection)
        # Labels: 0 = Healthy, 1 = Disease
        disease_pattern = (X[:, 0] ** 2 + X[:, 1] ** 2) > 1.5
        y = disease_pattern.astype(int)
        
        print(f"Generated {n_samples} samples with {n_features} features")
        print(f"Class distribution: Healthy={np.sum(y==0)}, Disease={np.sum(y==1)}")
    else:
        # Use real data
        print("\n[3] Preparing Real Medical Data:")
        print("-" * 70)
        
        # Assume last column is label, or create synthetic label
        if 'label' in eeg_data.columns:
            X, y = prepare_quantum_classification_data(eeg_data, label_column='label', n_samples=150)
        else:
            # Generate synthetic labels for demonstration
            X = eeg_data.values[:, :10]  # Use first 10 features
            y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1.5).astype(int)
    
    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # 5. Train quantum classifier
    print("\n[4] Training Quantum Variational Classifier:")
    print("-" * 70)
    
    # Create and train quantum classifier
    quantum_model = QuantumDiseaseClassifier(
        num_qubits=4,      # Use 4 qubits
        num_layers=2,      # 2 layers in variational form
        max_iter=50,       # 50 optimization iterations
        seed=42
    )
    
    quantum_model.train(X_train, y_train, X_test, y_test)
    
    # 6. Evaluate model
    print("\n[5] Evaluating Quantum Model:")
    print("-" * 70)
    
    # Make predictions
    y_pred = quantum_model.predict(X_test)
    y_proba = quantum_model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Disease']))
    
    # 7. Visualize results
    print("\n[6] Visualizing Results:")
    print("-" * 70)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 5))
    
    # Confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Disease'],
                yticklabels=['Healthy', 'Disease'])
    plt.title(f'Quantum Classifier Confusion Matrix\nAccuracy: {accuracy:.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Prediction probabilities
    plt.subplot(1, 2, 2)
    healthy_probs = y_proba[y_test == 0, 0]
    disease_probs = y_proba[y_test == 1, 1]
    
    plt.hist(healthy_probs, bins=20, alpha=0.7, label='Healthy', color='green')
    plt.hist(disease_probs, bins=20, alpha=0.7, label='Disease', color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantum_classification_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to 'quantum_classification_results.png'")
    
    # 8. Quantum circuit visualization (optional)
    print("\n[7] Quantum Circuit Architecture:")
    print("-" * 70)
    try:
        circuit = quantum_model.visualize_circuit()
        plt.figure(figsize=(10, 6))
        circuit
        plt.savefig('quantum_circuit_visualization.png', dpi=150, bbox_inches='tight')
        print("Circuit visualization saved to 'quantum_circuit_visualization.png'")
    except Exception as e:
        print(f"Could not visualize circuit: {e}")
    
    # 9. Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"✓ Quantum classifier successfully trained")
    print(f"✓ Test accuracy: {accuracy:.2%}")
    print(f"✓ Model uses {quantum_model.num_qubits} qubits with {quantum_model.num_layers} layers")
    print(f"✓ Results saved to 'quantum_classification_results.png'")
    print("\n" + "=" * 70)
    
    return quantum_model, X_test, y_test, y_pred


if __name__ == "__main__":
    # Run the demo
    model, X_test, y_test, y_pred = run_quantum_medical_classification()
