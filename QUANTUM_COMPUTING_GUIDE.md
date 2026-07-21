# Quantum Computing Guide for Agentic Disease Finder

This guide explains how to use the quantum machine learning features integrated into the Agentic Disease Finder system.

## Overview

The quantum computing integration uses **Qiskit** to implement Variational Quantum Classifiers (VQC) for medical disease detection. This adds quantum machine learning capabilities to analyze EEG signals and other medical data.

## What's Included

### Files Created:

1. **`quantum_classifier.py`** - Core quantum classifier implementation
2. **`quantum_integration_demo.py`** - Demo script showing full workflow
3. **`QUANTUM_COMPUTING_GUIDE.md`** - This guide

### Features:

- **Quantum Variational Classifier (VQC)** using Qiskit
- **ZZFeatureMap** for encoding classical data into quantum states
- **RealAmplitudes** parameterized quantum circuit
- **SPSA optimizer** for variational optimization
- **Quantum circuit visualization**
- **Model save/load functionality**
- **Integration with existing medical data**

## Installation

The quantum computing libraries have been added to `requirements.txt`:

```
qiskit>=0.45.0
qiskit-algorithms>=0.2.0
qiskit-machine-learning>=0.7.0
qiskit-aer>=0.12.0
qiskit-ibm-provider>=0.10.0
```

Install them with:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install qiskit qiskit-algorithms qiskit-machine-learning qiskit-aer qiskit-ibm-provider
```

## Quick Start

### 1. Run the Demo Script

```bash
python quantum_integration_demo.py
```

This will:
- Create a Bell state quantum circuit
- Train a quantum classifier on medical data
- Generate evaluation metrics
- Create visualizations

### 2. Use the Quantum Classifier

```python
from quantum_classifier import QuantumDiseaseClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Prepare your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train quantum classifier
quantum_model = QuantumDiseaseClassifier(
    num_qubits=4,      # Number of qubits (matches feature dimension)
    num_layers=2,      # Depth of variational form
    max_iter=100,      # Optimization iterations
    seed=42
)

# Train the model
quantum_model.train(X_train, y_train, X_test, y_test)

# Make predictions
predictions = quantum_model.predict(X_test)
probabilities = quantum_model.predict_proba(X_test)
```

### 3. Visualize Quantum Circuit

```python
# View the quantum circuit architecture
quantum_model.visualize_circuit()

# Get the circuit for further analysis
circuit = quantum_model.get_quantum_circuit()
print(circuit)
```

### 4. Save and Load Models

```python
# Save trained model
quantum_model.save_model('quantum_disease_model.pkl')

# Load previously trained model
new_model = QuantumDiseaseClassifier()
new_model.load_model('quantum_disease_model.pkl')
```

## Understanding Quantum Machine Learning

### Key Concepts:

#### 1. **Variational Quantum Classifier (VQC)**
A quantum machine learning algorithm that:
- Encodes classical data into quantum states using a **feature map**
- Applies a parameterized quantum circuit (**ansatz**)
- Optimizes parameters using classical optimization
- Uses quantum measurement to generate predictions

#### 2. **Feature Map (ZZFeatureMap)**
Encodes classical data into quantum Hilbert space:
- Uses entangling gates (CNOT, ZZ interactions)
- Creates quantum correlations between features
- Linear entanglement pattern

#### 3. **Ansatz (RealAmplitudes)**
Parameterized quantum circuit for optimization:
- Rotation gates (Ry) for encoding parameters
- CNOT gates for entanglement
- Depth controlled by `num_layers`

#### 4. **Optimizer (SPSA)**
Simultaneous Perturbation Stochastic Approximation:
- Efficient for noisy quantum systems
- Works with quantum simulators and real hardware

### Quantum Circuit Architecture:

```
Classical Data → [Feature Map] → [Ansatz] → Measurement → Prediction
                  (ZZFeatureMap)  (RealAmplitudes)
```

## Configuration Options

### QuantumClassifier Parameters:

```python
QuantumDiseaseClassifier(
    num_qubits=4,      # Number of qubits (adjust based on features)
    num_layers=3,      # Layers in variational form (2-5 typical)
    max_iter=100,      # Optimization iterations (50-200)
    seed=42           # Random seed for reproducibility
)
```

### Recommendations:

- **Small datasets (<100 samples)**: `num_qubits=2-3`, `num_layers=1-2`, `max_iter=50`
- **Medium datasets (100-1000 samples)**: `num_qubits=4-6`, `num_layers=2-3`, `max_iter=100`
- **Large datasets (>1000 samples)**: Consider hybrid quantum-classical approaches

## Integration with Medical Data

The quantum classifier works with your existing medical data formats:

### EEG Data:

```python
import pandas as pd

# Load EEG data
data = pd.read_csv('sample_data/eeg_parkinsons.csv')

# Extract features and labels
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Labels

# Train quantum classifier
quantum_model = QuantumDiseaseClassifier(num_qubits=4, num_layers=2)
quantum_model.train(X_train, y_train)
```

### Motor Imagery Data:

```python
# Load motor imagery data
data = pd.read_csv('sample_data/motor_imagery_right_hand.csv')
X = data.values[:, :10]  # Use first 10 features

# Train and evaluate
quantum_model.train(X_train, y_train, X_test, y_test)
```

## Performance Tips

### 1. **Data Preprocessing**
The quantum classifier includes automatic scaling and dimensionality reduction:
- Scales features to [0, 1] range
- Reduces or pads features to match number of qubits

### 2. **Optimization**
- Start with fewer iterations and increase if needed
- Monitor training/validation accuracy
- Use early stopping to prevent overfitting

### 3. **Quantum Advantage**
Quantum advantage is most apparent when:
- Working with high-dimensional data
- Need quantum entanglement for feature relationships
- Using quantum hardware with error correction

### 4. **Hybrid Approaches**
For best results, combine quantum and classical methods:
- Use quantum for feature extraction
- Use classical for final classification
- Ensemble multiple quantum circuits

## Advanced Usage

### Custom Feature Maps

```python
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap

# Try different feature maps
feature_map = PauliFeatureMap(
    feature_dimension=4,
    reps=2
)
```

### Custom Ansatz

```python
from qiskit.circuit.library import EfficientSU2

# Use EfficientSU2 ansatz
ansatz = EfficientSU2(num_qubits=4, reps=2)
```

### Running on Quantum Hardware

```python
from qiskit_ibm_provider import IBMProvider

# Load your IBM Quantum account
provider = IBMProvider()

# Get quantum computer
backend = provider.get_backend('ibm_osaka')

# Use in classifier (requires modifications)
# Note: Quantum hardware has long wait times
```

## Limitations and Considerations

1. **Simulation Speed**: Quantum simulators are slower than classical ML
2. **Scalability**: Number of qubits limited (current: 4-127 qubits)
3. **Noise**: Real quantum hardware has noise/errors
4. **Training Time**: Quantum optimization can be slower
5. **Data Size**: Best for small-to-medium datasets

## Example Outputs

Running `quantum_integration_demo.py` creates:

1. **`quantum_classification_results.png`** - Confusion matrix and confidence distributions
2. **`quantum_circuit_visualization.png`** - Circuit architecture diagram

## Troubleshooting

### Import Errors

```bash
# Reinstall Qiskit
pip install --upgrade qiskit qiskit-algorithms qiskit-machine-learning qiskit-aer
```

### Memory Issues

```python
# Reduce qubit count
quantum_model = QuantumDiseaseClassifier(num_qubits=2, ...)
```

### Slow Training

```python
# Reduce max iterations
quantum_model = QuantumDiseaseClassifier(max_iter=50, ...)
```

## Additional Resources

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Quantum Machine Learning Tutorial](https://qiskit.org/ecosystem/machine-learning/tutorials/)
- [Qiskit Textbook - ML](https://qiskit.org/learn/machine-learning/)

## Citation

If you use this quantum classifier in research:

```bibtex
@misc{quantum_disease_finder,
  title={Quantum Machine Learning for Medical Disease Detection},
  author={Agentic Disease Finder},
  year={2024},
  howpublished={\url{https://github.com/yourrepo/AgenticDiseaseFinder}}
}
```

## License

Same as Agentic Disease Finder project.

---

**Note**: This quantum implementation is experimental and for research purposes. For production medical applications, ensure compliance with medical device regulations.

