"""
evaluator.py

This file contains functions for evaluating the trained quantum model.
- Tests on unseen data
- Generates performance metrics
- Creates visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate_model(model, qnode, X_test, y_test, final_weights):
    """
    Comprehensive evaluation of the trained model
    
    Args:
        model: QuantumWatermarkModel instance
        qnode: Quantum node function
        X_test: Test features
        y_test: Test labels
        final_weights: Trained weights
    
    Returns:
        Dictionary with evaluation results
    """
    print("Evaluating model on test set...")
    
    # Get predictions for all test samples
    predictions = []
    probabilities = []
    
    for i in range(len(X_test)):
        inputs = X_test[i]
        
        # Get probability and class prediction - NOW PASS QNODE AS PARAMETER
        prob = model.predict_probability(inputs, final_weights, qnode)
        pred_class = model.predict_class(inputs, final_weights, qnode)
        
        probabilities.append(prob)
        predictions.append(pred_class)
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy:.3f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, 
                              target_names=['Clean (0)', 'Watermarked (1)']))
    
    # Calculate additional metrics
    true_positives = np.sum((y_test == 1) & (predictions == 1))
    false_positives = np.sum((y_test == 0) & (predictions == 1))
    true_negatives = np.sum((y_test == 0) & (predictions == 0))
    false_negatives = np.sum((y_test == 1) & (predictions == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': cm,
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': y_test
    }
    
    return results

def visualize_results(results):
    """
    Create visualizations for the evaluation results
    
    Args:
        results: Dictionary returned by evaluate_model
    """
    os.makedirs('data/results', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Confusion Matrix
    sns.heatmap(results['confusion_matrix'], 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Clean', 'Watermarked'],
                yticklabels=['Clean', 'Watermarked'],
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. Probability Distribution
    clean_probs = results['probabilities'][results['true_labels'] == 0]
    watermarked_probs = results['probabilities'][results['true_labels'] == 1]
    
    axes[0, 1].hist(clean_probs, bins=20, alpha=0.7, label='Clean', color='blue', density=True)
    axes[0, 1].hist(watermarked_probs, bins=20, alpha=0.7, label='Watermarked', color='red', density=True)
    axes[0, 1].set_xlabel('Prediction Probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution of Prediction Probabilities')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Prediction vs True Labels Scatter
    axes[1, 0].scatter(range(len(results['probabilities'])), 
                      results['probabilities'], 
                      c=results['true_labels'], 
                      cmap='viridis', 
                      alpha=0.7)
    axes[1, 0].axhline(y=0.5, color='red', linestyle='--', label='Decision Threshold')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Prediction Probability')
    axes[1, 0].set_title('Predictions vs True Labels')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Metrics Summary
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['accuracy'], results['precision'], results['recall'], results['f1_score']]
    
    bars = axes[1, 1].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('data/results/evaluation_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_single_sample(model, qnode, sample_input, weights, true_label=None):
    """
    Test the model on a single sample
    
    Args:
        model: QuantumWatermarkModel instance
        qnode: Quantum node function
        sample_input: Single input sample (16-element normalized vector)
        weights: Trained weights
        true_label: True label for the sample (optional)
    """
    # Get prediction - NOW PASS QNODE AS PARAMETER
    probability = model.predict_probability(sample_input, weights, qnode)
    predicted_class = model.predict_class(sample_input, weights, qnode)
    
    print(f"Single Sample Test:")
    print(f"  Input shape: {sample_input.shape}")
    print(f"  Prediction probability: {probability:.4f}")
    print(f"  Predicted class: {'Watermarked (1)' if predicted_class == 1 else 'Clean (0)'}")
    
    if true_label is not None:
        actual_class = 'Watermarked (1)' if true_label == 1 else 'Clean (0)'
        correctness = '✓ CORRECT' if predicted_class == true_label else '✗ INCORRECT'
        print(f"  Actual class: {actual_class}")
        print(f"  Result: {correctness}")
    
    return predicted_class, probability

if __name__ == "__main__":
    # This file is meant to be imported, not run directly
    print("Evaluator module loaded successfully!")