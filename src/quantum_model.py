"""
quantum_model.py

This file defines the quantum neural network model using amplitude embedding.
- Amplitude embedding: encodes 16 classical features into 4 qubits
- Variational quantum circuit: learns to classify watermarked vs clean images
"""

import pennylane as qml
import numpy as np

class QuantumWatermarkModel:
    """
    Quantum Neural Network for watermark detection using amplitude embedding
    """
    
    def __init__(self, n_qubits=4, n_layers=3):
        """
        Initialize the quantum model
        
        Args:
            n_qubits: Number of qubits (4 qubits can handle 16 features via amplitude embedding)
            n_layers: Number of variational layers in the circuit
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Define the quantum device (simulator)
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        
        # Initialize random parameters for the variational circuit
        # Shape: (n_layers, n_qubits, 3) - 3 parameters per qubit per layer (RX, RY, RZ)
        self.weights = np.random.uniform(0, 2 * np.pi, size=(self.n_layers, self.n_qubits, 3))
        
        print(f"Quantum model initialized with {self.n_qubits} qubits and {self.n_layers} layers")
        print(f"Total parameters: {self.weights.size}")
    
    def create_quantum_circuit(self):
        """
        Create the quantum circuit as a QNode
        """
        @qml.qnode(self.dev, interface='numpy')
        def quantum_circuit(inputs, weights):
            """
            Define the quantum circuit
            
            Args:
                inputs: Normalized input vector (16 elements for amplitude embedding)
                weights: Variational parameters for the circuit
            
            Returns:
                Expectation value of PauliZ on the first qubit
            """
            # Reset the device state
            qml.AmplitudeEmbedding(
                features=inputs,  # 16-element input vector
                wires=range(self.n_qubits),  # Apply to all qubits
                pad_with=0.0,  # Pad with zeros if input < 2^n_qubits
                normalize=True  # Automatically normalize the input vector
            )
            
            # Apply variational layers
            for layer in range(self.n_layers):
                # Apply rotations to each qubit
                for qubit in range(self.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                
                # Add entangling gates between adjacent qubits
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            
            # Return expectation value of PauliZ on the first qubit
            # This will be between -1 and 1, which we can convert to probability
            return qml.expval(qml.PauliZ(0))
        
        return quantum_circuit
    
    def predict_probability(self, inputs, weights, qnode):
        """
        Predict probability that input contains watermark
        
        Args:
            inputs: Normalized input vector
            weights: Current circuit parameters
            qnode: The quantum node function
        
        Returns:
            Probability between 0 and 1 (where 1 = watermarked, 0 = clean)
        """
        # Get expectation value from quantum circuit
        exp_val = qnode(inputs, weights)
        
        # Convert expectation value (-1 to 1) to probability (0 to 1)
        # Formula: (exp_val + 1) / 2
        probability = (float(exp_val) + 1) / 2  # Convert to float to handle Pennylane types
        
        return probability
    
    def predict_class(self, inputs, weights, qnode, threshold=0.5):
        """
        Predict class (0 for clean, 1 for watermarked) based on probability
        
        Args:
            inputs: Normalized input vector
            weights: Current circuit parameters
            qnode: The quantum node function
            threshold: Decision threshold (default 0.5)
        
        Returns:
            0 or 1 (predicted class)
        """
        prob = self.predict_probability(inputs, weights, qnode)
        return 1 if prob >= threshold else 0

def loss_function(weights, qnode, X_batch, y_batch):
    """
    Binary cross-entropy loss function for quantum classification
    
    Args:
        weights: Current circuit parameters
        qnode: Quantum node function
        X_batch: Batch of input features
        y_batch: Batch of true labels
    
    Returns:
        Average loss across the batch
    """
    total_loss = 0
    
    for i in range(len(X_batch)):
        inputs = X_batch[i]
        true_label = y_batch[i]
        
        # Get prediction from quantum circuit
        exp_val = qnode(inputs, weights)
        prob = (float(exp_val) + 1) / 2  # Convert to probability and handle Pennylane types
        prob = np.clip(prob, 1e-7, 1 - 1e-7)  # Clip to avoid log(0)
        
        # Binary cross-entropy loss
        if true_label == 1:
            loss = -np.log(prob)
        else:
            loss = -np.log(1 - prob)
        
        total_loss += loss
    
    return total_loss / len(X_batch)

def accuracy_metric(weights, qnode, X, y, threshold=0.5):
    """
    Calculate accuracy of the model
    
    Args:
        weights: Current circuit parameters
        qnode: Quantum node function
        X: Input features
        y: True labels
        threshold: Decision threshold
    
    Returns:
        Accuracy as a percentage
    """
    correct = 0
    total = len(X)
    
    for i in range(total):
        inputs = X[i]
        true_label = y[i]
        
        # Get prediction
        exp_val = qnode(inputs, weights)
        prob = (float(exp_val) + 1) / 2
        predicted = 1 if prob >= threshold else 0
        
        if predicted == true_label:
            correct += 1
    
    return correct / total

if __name__ == "__main__":
    # Test the quantum model
    print("Testing quantum model...")
    model = QuantumWatermarkModel(n_qubits=4, n_layers=2)
    qnode = model.create_quantum_circuit()
    
    # Create a sample input (16 features, normalized)
    sample_input = np.random.rand(16)
    sample_input = sample_input / np.linalg.norm(sample_input)  # Normalize
    
    # Test prediction
    prob = model.predict_probability(sample_input, model.weights, qnode)
    pred_class = model.predict_class(sample_input, model.weights, qnode)
    
    print(f"Sample input prediction: probability = {prob:.3f}, class = {pred_class}")
    print("Quantum model test successful!")