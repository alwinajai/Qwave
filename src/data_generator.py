"""
data_generator.py

This file generates datasets for training:
- Option 1: Synthetic 4x4 binary images (original method)
- Option 2: Real images processed to 16 features (new method)
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def generate_synthetic_data(n_samples_per_class=100):
    """
    Original method: Generate synthetic 4x4 binary images
    """
    def generate_clean_image():
        image = np.random.choice([0, 1], size=(4, 4))
        return image.flatten()

    def generate_watermarked_image():
        image = np.random.choice([0, 1], size=(4, 4))
        watermark_pattern = [1, 0, 1, 0]
        image[0, 0] = watermark_pattern[0]
        image[0, 1] = watermark_pattern[1]
        image[1, 0] = watermark_pattern[2]
        image[1, 1] = watermark_pattern[3]
        return image.flatten()

    def normalize_for_amplitude_embedding(data):
        norm = np.linalg.norm(data)
        if norm == 0:
            normalized = np.zeros_like(data)
            normalized[0] = 1.0
            return normalized
        return data / norm

    clean_images = []
    watermarked_images = []

    for _ in range(n_samples_per_class):
        clean_img = generate_clean_image()
        clean_images.append(clean_img)

    for _ in range(n_samples_per_class):
        wm_img = generate_watermarked_image()
        watermarked_images.append(wm_img)

    X_clean = np.array(clean_images)
    X_watermarked = np.array(watermarked_images)

    X_clean_normalized = np.array([normalize_for_amplitude_embedding(img) for img in X_clean])
    X_watermarked_normalized = np.array([normalize_for_amplitude_embedding(img) for img in X_watermarked])

    X = np.vstack([X_clean_normalized, X_watermarked_normalized])
    y = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])

    print(f"Synthetic dataset created: {X.shape[0]} samples, {X.shape[1]} features each")
    return X, y

def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test the synthetic data generation
    print("Testing synthetic data generation...")
    X_synthetic, y_synthetic = generate_synthetic_data(n_samples_per_class=50)
    print(f"Synthetic: {X_synthetic.shape}")
    print("Synthetic data generation test successful!")