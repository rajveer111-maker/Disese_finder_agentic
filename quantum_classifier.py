"""
Quantum Classifier using Qiskit for Medical Disease Detection
This module implements quantum machine learning for EEG signal analysis
"""

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle


class QuantumDiseaseClassifier:
    """
    Quantum Variational Quantum Classifier for disease detection from EEG signals.
    
    Uses Variational Quantum Classifier (VQC) with:
    - ZZFeatureMap for encoding classical data into quantum states
    - RealAmplitudes for parameterized quantum circuit
    - SPSA optimizer for variational optimization
    """
    
    def __init__(self, num_qubits=4, num_layers=3, max_iter=100, seed=42):
        """
        Initialize the Quantum Disease Classifier.
        
        Parameters:
        -----------
        num_qubits : int
            Number of qubits for the quantum circuit
        num_layers : int
            Number of layers in the variational form
        max_iter : int
            Maximum iterations for optimization
        seed : int
            Random seed for reproducibility
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.max_iter = max_iter
        self.seed = seed
        self.model = None
        self.feature_map = None
        self.ansatz = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
        # Create feature map and ansatz
        self._build_quantum_circuit()
    
    def _build_quantum_circuit(self):
        """Build the quantum circuit with feature map and ansatz."""
        # Feature map for encoding classical data
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.num_qubits,
            reps=2,
            entanglement='linear'
        )
        
        # Ansatz (parameterized quantum circuit)
        self.ansatz = RealAmplitudes(
            num_qubits=self.num_qubits,
            reps=self.num_layers
        )
        
        print(f"Quantum Circuit: {self.num_qubits} qubits, {self.num_layers} layers")
        print(f"Feature map: {self.feature_map.num_parameters} parameters")
        print(f"Ansatz: {self.ansatz.num_parameters} parameters")
    
    def _preprocess_data(self, X, fit=False):
        """
        Preprocess the input data for quantum circuit.
        
        Parameters:
        -----------
        X : array-like
            Input features
        fit : bool
            Whether to fit the scaler
        
        Returns:
        --------
        X_scaled : array
            Scaled and reduced features
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Reduce dimensionality to match number of qubits
        if X_scaled.shape[1] > self.num_qubits:
            # Use PCA or simple sampling
            X_reduced = X_scaled[:, :self.num_qubits]
        elif X_scaled.shape[1] < self.num_qubits:
            # Pad with zeros
            padding = np.zeros((X_scaled.shape[0], self.num_qubits - X_scaled.shape[1]))
            X_reduced = np.hstack([X_scaled, padding])
        else:
            X_reduced = X_scaled
        
        return X_reduced
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the quantum classifier.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation labels
        """
        print("Preprocessing training data...")
        X_train_scaled = self._preprocess_data(X_train, fit=True)
        
        # Create Variational Quantum Classifier
        print("Creating Variational Quantum Classifier...")
        self.model = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=SPSA(maxiter=self.max_iter),
            sampler=Sampler(),
            callback=self._callback if X_val is not None and y_val is not None else None
        )
        
        # Train the model
        print(f"Training quantum classifier (max {self.max_iter} iterations)...")
        self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        print("Training complete!")
        
        # Evaluate on training data
        train_score = self.model.score(X_train_scaled, y_train)
        print(f"Training Accuracy: {train_score:.4f}")
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self._preprocess_data(X_val, fit=False)
            val_score = self.model.score(X_val_scaled, y_val)
            print(f"Validation Accuracy: {val_score:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained quantum classifier.
        
        Parameters:
        -----------
        X : array-like
            Input features
        
        Returns:
        --------
        predictions : array
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        X_scaled = self._preprocess_data(X, fit=False)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like
            Input features
        
        Returns:
        --------
        probabilities : array
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        X_scaled = self._preprocess_data(X, fit=False)
        
        # Get the underlying QNN
        # Note: VQC doesn't directly support predict_proba,
        # so we'll use a workaround
        probabilities = self.model.decision_function(X_scaled)
        
        # Convert to probabilities (sigmoid)
        from scipy.special import expit
        prob_positive = expit(probabilities)
        prob_negative = 1 - prob_positive
        
        return np.column_stack([prob_negative, prob_positive])
    
    def _callback(self, weights, obj_func_eval):
        """Callback function for optimization tracking."""
        print(f"Iteration: {len(self.model._fit_result) if hasattr(self.model, '_fit_result') else 0}, "
              f"Objective: {obj_func_eval:.4f}")
    
    def get_quantum_circuit(self):
        """
        Get the complete quantum circuit.
        
        Returns:
        --------
        circuit : QuantumCircuit
            The complete quantum circuit
        """
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(self.ansatz, inplace=True)
        circuit.measure_all()
        return circuit
    
    def visualize_circuit(self):
        """Visualize the quantum circuit."""
        circuit = self.get_quantum_circuit()
        print(circuit)
        return circuit.draw('mpl', output='mpl')
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'num_qubits': self.num_qubits,
                'num_layers': self.num_layers,
                'is_trained': self.is_trained
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.num_qubits = data['num_qubits']
            self.num_layers = data['num_layers']
            self.is_trained = data['is_trained']
        
        # Rebuild feature map and ansatz
        self._build_quantum_circuit()
        print(f"Model loaded from {filepath}")


def create_quantum_bell_state():
    """
    Create a simple quantum Bell state circuit for demonstration.
    
    Returns:
    --------
    qc : QuantumCircuit
        A 2-qubit Bell state circuit
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)  # Apply Hadamard gate
    qc.cx(0, 1)  # Apply CNOT gate
    qc.measure_all()
    return qc


def execute_bell_state():
    """
    Execute a Bell state circuit and return the results.
    
    Returns:
    --------
    counts : dict
        Measurement counts
    """
    qc = create_quantum_bell_state()
    
    # Use Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')
    
    # Execute the circuit on the simulator
    job = execute(qc, simulator, shots=1024)
    
    # Get the results
    result = job.result()
    counts = result.get_counts(qc)
    
    print("Bell State Results:")
    print(counts)
    
    return counts


if __name__ == "__main__":
    # Example usage
    print("Quantum Disease Classifier Demo")
    print("=" * 50)
    
    # 1. Demonstrate Bell state
    print("\n1. Creating Bell State:")
    print("-" * 30)
    bell_counts = execute_bell_state()
    
    # 2. Example with synthetic data
    print("\n2. Training Quantum Classifier with Synthetic Data:")
    print("-" * 30)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 8
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train quantum classifier
    quantum_model = QuantumDiseaseClassifier(num_qubits=4, num_layers=2, max_iter=10)
    
    # Train with reduced iterations for demo
    quantum_model.train(X_train, y_train, X_test, y_test)
    
    # Make predictions
    predictions = quantum_model.predict(X_test)
    print(f"\nPredictions on test set: {predictions[:10]}")
    
    # Test probabilities
    proba = quantum_model.predict_proba(X_test)
    print(f"\nPrediction probabilities (first 5 samples):")
    for i in range(min(5, len(proba))):
        print(f"Sample {i}: Class 0: {proba[i][0]:.3f}, Class 1: {proba[i][1]:.3f}")
    
    print("\n" + "=" * 50)
    print("Demo complete!")
