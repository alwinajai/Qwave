"""
trainer.py

This file contains the training logic for the quantum model.
- Uses gradient descent to optimize quantum circuit parameters
- Tracks loss and accuracy during training
"""

import numpy as np
import matplotlib.pyplot as plt
import os

class QuantumTrainer:
    """
    Handles training of the quantum watermark detection model
    """
    
    def __init__(self, model, learning_rate=0.1):
        """
        Initialize the trainer
        
        Args:
            model: QuantumWatermarkModel instance
            learning_rate: Learning rate for gradient descent
        """
        self.model = model
        self.learning_rate = learning_rate
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
        print(f"Trainer initialized with learning rate: {learning_rate}")
    
    def train_epoch(self, qnode, X_train, y_train, weights):
        """
        Perform one training epoch
        
        Args:
            qnode: Quantum node function
            X_train: Training features
            y_train: Training labels
            weights: Current parameters
        
        Returns:
            Updated weights after one epoch
        """
        # Compute gradient using finite differences
        # This is a simple numerical gradient - Pennylane can do automatic differentiation too
        grad = np.zeros_like(weights)
        
        # Compute original loss
        original_loss = self.loss_function(weights, qnode, X_train, y_train)
        
        # Compute gradients for each parameter
        eps = 1e-5  # Small perturbation for numerical gradient
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                for k in range(weights.shape[2]):
                    # Perturb parameter
                    weights_plus = weights.copy()
                    weights_minus = weights.copy()
                    weights_plus[i, j, k] += eps
                    weights_minus[i, j, k] -= eps
                    
                    # Compute loss with perturbed parameters
                    loss_plus = self.loss_function(weights_plus, qnode, X_train, y_train)
                    loss_minus = self.loss_function(weights_minus, qnode, X_train, y_train)
                    
                    # Compute gradient
                    grad[i, j, k] = (loss_plus - loss_minus) / (2 * eps)
        
        # Update weights using gradient descent
        new_weights = weights - self.learning_rate * grad
        
        return new_weights
    
    def loss_function(self, weights, qnode, X_batch, y_batch):
        """
        Binary cross-entropy loss function for quantum classification
        """
        from src.quantum_model import loss_function as model_loss_function
        return model_loss_function(weights, qnode, X_batch, y_batch)
    
    def accuracy_metric(self, weights, qnode, X, y, threshold=0.5):
        """
        Calculate accuracy of the model
        """
        from src.quantum_model import accuracy_metric
        return accuracy_metric(weights, qnode, X, y, threshold)
    
    def train(self, qnode, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
        """
        Train the quantum model
        
        Args:
            qnode: Quantum node function
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Size of training batches (not used in this simple version)
        
        Returns:
            Final trained weights
        """
        print(f"Starting training for {epochs} epochs...")
        weights = self.model.weights
        
        for epoch in range(epochs):
            # Train for one epoch
            weights = self.train_epoch(qnode, X_train, y_train, weights)
            
            # Calculate metrics
            train_loss = self.loss_function(weights, qnode, X_train, y_train)
            train_acc = self.accuracy_metric(weights, qnode, X_train, y_train)
            val_acc = self.accuracy_metric(weights, qnode, X_val, y_val)
            
            # Store metrics
            self.loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            
            # Print progress every 10 epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
        
        print("Training completed!")
        return weights
    
    def plot_training_history(self):
        """
        Plot training history (loss and accuracy over time)
        """
        os.makedirs('data/results', exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.loss_history, label='Training Loss', color='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True)
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.train_acc_history, label='Training Accuracy', color='green')
        ax2.plot(self.val_acc_history, label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Over Time')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('data/results/training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # This file is meant to be imported, not run directly
    print("Trainer module loaded successfully!")