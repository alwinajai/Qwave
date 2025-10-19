"""
main.py

Main script that runs the complete QML Quantum Watermark Detection pipeline:
1. Load CIFAR-100 dataset with watermarks
2. Initialize quantum model
3. Train the model
4. Evaluate results
5. Create visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.quantum_model import QuantumWatermarkModel
from src.trainer import QuantumTrainer
from src.evaluator import evaluate_model, visualize_results, test_single_sample
from src.cifar_watermark_processor import create_cifar100_watermark_dataset, split_dataset

def main():
    """
    Main function that runs the complete pipeline
    """
    print("=" * 60)
    print("QML Quantum Digital Watermarking Detector")
    print("Using CIFAR-100 with Algorithmic Watermarks")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("data/results", exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. GENERATE DATASET (CIFAR-100 with watermarks)
    print("\n1. Generating Dataset from CIFAR-100 with Watermarks...")
    print("-" * 30)
    
    # Create dataset using CIFAR-100 (reduced for faster execution)
    X, y = create_cifar100_watermark_dataset(n_samples_per_class=200, use_cache=True)  # 200 clean + 200 watermarked
    
    # Split into train/test
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)
    
    # Further split training data for validation
    X_train_final, X_val, y_train_final, y_val = split_dataset(
        X_train, y_train, test_size=0.25, random_state=42
    )  # This gives us ~320 train, ~80 val, ~80 test
    
    # 2. INITIALIZE QUANTUM MODEL
    print("\n2. Initializing Quantum Model...")
    print("-" * 30)
    model = QuantumWatermarkModel(n_qubits=4, n_layers=3)
    qnode = model.create_quantum_circuit()  # Create the QNode properly
    
    # 3. TRAIN THE MODEL
    print("\n3. Training Model...")
    print("-" * 30)
    trainer = QuantumTrainer(model, learning_rate=0.05)
    
    final_weights = trainer.train(
        qnode=qnode,
        X_train=X_train_final,
        y_train=y_train_final,
        X_val=X_val,
        y_val=y_val,
        epochs=35,  # OPTIMIZED: Reduced from 80 to 35 epochs
        batch_size=16
    )
    
    # Plot training history
    print("\nPlotting training history...")
    trainer.plot_training_history()
    
    # Save training history
    training_data = {
        'loss_history': trainer.loss_history,
        'train_acc_history': trainer.train_acc_history,
        'val_acc_history': trainer.val_acc_history,
        'final_weights': final_weights
    }
    with open('data/results/training_data.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    print("Training data saved to data/results/training_data.pkl")
    
    # 4. EVALUATE THE MODEL
    print("\n4. Evaluating Model...")
    print("-" * 30)
    results = evaluate_model(model, qnode, X_test, y_test, final_weights)
    
    # Save evaluation results
    with open('data/results/evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Evaluation results saved to data/results/evaluation_results.pkl")
    
    # Print summary of results
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {results['accuracy']:.3f}")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall: {results['recall']:.3f}")
    print(f"  F1-Score: {results['f1_score']:.3f}")
    
    # Visualize results
    print("\nCreating evaluation visualizations...")
    visualize_results(results)
    
    # 5. TEST ON SINGLE SAMPLES
    print("\n5. Testing Individual Samples...")
    print("-" * 30)
    
    # Test on a few samples from the test set
    for i in range(min(5, len(X_test))):
        print(f"\nSample {i + 1}:")
        test_single_sample(model, qnode, X_test[i], final_weights, y_test[i])
    
    # 6. SUMMARY
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model Summary:")
    print(f"  - Dataset: CIFAR-100 with algorithmic watermarks")
    print(f"  - Total samples: {len(X)} (50% clean, 50% watermarked)")
    print(f"  - Quantum Circuit: 4 qubits, 3 variational layers")
    print(f"  - Embedding: Amplitude embedding (16 features → 4 qubits)")
    print(f"  - Final Accuracy: {results['accuracy']:.3f}")
    print(f"  - Training completed in {len(trainer.loss_history)} epochs")
    print(f"  - Features extracted from 32x32x3 CIFAR-100 images")
    print(f"  - Optimized for efficiency: 35 epochs (not overfitted)")
    print("\nAll results saved to:")
    print("  - data/results/training_history.png")
    print("  - data/results/evaluation_results.png")
    print("  - data/results/training_data.pkl")
    print("  - data/results/evaluation_results.pkl")
    print("=" * 60)

if __name__ == "__main__":
    main()