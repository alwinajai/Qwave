"""
quantum_predictor.py

Simplified quantum model for prediction only (no training).
Used in the GUI application.
"""

import pennylane as qml
import numpy as np

class QuantumWatermarkPredictor:
    """
    Simplified quantum model for prediction only
    """
    
    def __init__(self, n_qubits=4, n_layers=3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
    
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
            return qml.expval(qml.PauliZ(0))
        
        return quantum_circuit
    
    def predict_probability(self, inputs, weights, qnode):
        """
        Predict probability that input contains watermark
        """
        exp_val = qnode(inputs, weights)
        probability = (float(exp_val) + 1) / 2
        return probability
    
    def predict_class(self, inputs, weights, qnode, threshold=0.5):
        """
        Predict class (0 for clean, 1 for watermarked) based on probability
        """
        prob = self.predict_probability(inputs, weights, qnode)
        return 1 if prob >= threshold else 0