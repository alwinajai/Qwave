"""
cifar_watermark_processor.py

This file handles:
- Loading CIFAR-100 dataset
- Adding watermarks to images
- Extracting features for quantum processing
- Managing data storage and caching
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import os
import pickle

class CIFARWatermarkProcessor:
    """
    Process CIFAR-100 images with watermarks for quantum ML
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.watermark_pattern = self.create_watermark_pattern()
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(os.path.join(self.data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "results"), exist_ok=True)
    
    def create_watermark_pattern(self):
        """
        Create a simple watermark pattern (4x4 binary pattern)
        """
        # 4x4 watermark pattern
        pattern = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ], dtype=np.float32)
        return pattern
    
    def add_watermark_to_image(self, image, alpha=0.1):
        """
        Add watermark to a single CIFAR-100 image
        
        Args:
            image: 32x32x3 RGB image
            alpha: Watermark strength (0.0 to 1.0)
        
        Returns:
            Watermarked image (same shape)
        """
        # Convert to float for processing
        img_float = image.astype(np.float32)
        
        # Resize watermark to fit in top-left corner of 32x32 image
        watermark_resized = cv2.resize(self.watermark_pattern, (8, 8), interpolation=cv2.INTER_NEAREST)
        
        # Normalize watermark to image intensity range
        watermark_normalized = watermark_resized * 255.0
        
        # Add watermark to top-left corner
        img_float[:8, :8, 0] += alpha * watermark_normalized[:8, :8]  # Red channel
        img_float[:8, :8, 1] += alpha * watermark_normalized[:8, :8]  # Green channel
        img_float[:8, :8, 2] += alpha * watermark_normalized[:8, :8]  # Blue channel
        
        # Clip values to valid range
        img_watermarked = np.clip(img_float, 0, 255).astype(np.uint8)
        
        return img_watermarked
    
    def extract_features_from_cifar_image(self, image, method='statistical'):
        """
        Extract 16 features from a CIFAR-100 image
        
        Args:
            image: 32x32x3 RGB image
            method: Feature extraction method
        
        Returns:
            16-element normalized feature vector
        """
        if method == 'statistical':
            # Convert to grayscale for statistical analysis
            gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
            
            features = []
            
            # Basic statistical features
            features.append(np.mean(gray))
            features.append(np.std(gray))
            features.append(np.var(gray))
            features.append(np.min(gray))
            features.append(np.max(gray))
            features.append(np.median(gray))
            
            # Block-based features (divide 32x32 into 8x8 blocks)
            block_means = []
            for i in range(0, 32, 8):  # 4 blocks in each dimension
                for j in range(0, 32, 8):
                    block = gray[i:i+8, j:j+8]
                    block_means.append(np.mean(block))
            
            # Take first 10 features to make 16 total
            features.extend(block_means[:10])
        
        elif method == 'frequency':
            # Convert to grayscale
            gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
            
            # Apply FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Extract frequency domain features
            features = []
            
            # Low frequency features (center)
            center_region = magnitude_spectrum[12:20, 12:20].flatten()
            features.extend(center_region[:8])
            
            # High frequency features (corners)
            corners = np.concatenate([
                magnitude_spectrum[:4, :4].flatten(),
                magnitude_spectrum[:4, 28:].flatten(),
                magnitude_spectrum[28:, :4].flatten(),
                magnitude_spectrum[28:, 28:].flatten()
            ])
            features.extend(corners[:8])
        
        # Ensure we have exactly 16 features
        features = np.array(features[:16])
        if len(features) < 16:
            features = np.pad(features, (0, 16 - len(features)), 'constant')
        
        # Normalize for amplitude embedding
        norm = np.linalg.norm(features)
        if norm == 0:
            features[0] = 1.0
        else:
            features = features / norm
        
        return features
    
    def save_processed_data(self, X, y, filename):
        """
        Save processed data to file
        
        Args:
            X: Feature matrix
            y: Labels
            filename: Name of file to save to
        """
        filepath = os.path.join(self.data_dir, "processed", filename)
        data = {
            'X': X,
            'y': y
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved processed data to {filepath}")
    
    def load_processed_data(self, filename):
        """
        Load processed data from file
        
        Args:
            filename: Name of file to load from
        
        Returns:
            X: Feature matrix
            y: Labels
        """
        filepath = os.path.join(self.data_dir, "processed", filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded processed data from {filepath}")
            return data['X'], data['y']
        else:
            return None, None
    
    def load_and_process_cifar100(self, n_samples_per_class=200, method='statistical', use_cache=True):
        """
        Load CIFAR-100, add watermarks, and extract features
        
        Args:
            n_samples_per_class: Number of samples per class to use
            method: Feature extraction method
            use_cache: Whether to use cached processed data if available
        
        Returns:
            X: Feature matrix (n_samples, 16)
            y: Labels (0 for clean, 1 for watermarked)
        """
        # Check if cached data exists
        cache_filename = f"cifar100_{n_samples_per_class}_{method}_features.pkl"
        if use_cache:
            X, y = self.load_processed_data(cache_filename)
            if X is not None and y is not None:
                print(f"Using cached data: {X.shape[0]} samples")
                return X, y
        
        print("Loading CIFAR-100 dataset...")
        
        # Load CIFAR-100
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        
        # Combine train and test for more data
        x_all = np.concatenate([x_train, x_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        
        # Use only first n_samples_per_class*2 samples (100 per clean/watermarked)
        total_samples = n_samples_per_class * 2
        if len(x_all) > total_samples:
            x_all = x_all[:total_samples]
            y_all = y_all[:total_samples]
        
        print(f"Selected {len(x_all)} total samples")
        
        # Split into clean and watermarked
        n_total = len(x_all)
        n_clean = n_total // 2
        n_watermarked = n_total - n_clean
        
        clean_indices = np.arange(0, n_clean)
        watermarked_indices = np.arange(n_clean, n_total)
        
        print(f"Processing {n_clean} clean and {n_watermarked} watermarked samples")
        
        # Process clean images
        print("Processing clean images...")
        clean_features = []
        for idx in clean_indices:
            features = self.extract_features_from_cifar_image(x_all[idx], method)
            clean_features.append(features)
        
        # Process watermarked images
        print("Processing watermarked images...")
        watermarked_features = []
        for idx in watermarked_indices:
            # Add watermark to image
            watermarked_img = self.add_watermark_to_image(x_all[idx])
            # Extract features from watermarked image
            features = self.extract_features_from_cifar_image(watermarked_img, method)
            watermarked_features.append(features)
        
        # Combine everything
        X_clean = np.array(clean_features)
        X_watermarked = np.array(watermarked_features)
        
        X = np.vstack([X_clean, X_watermarked])
        y = np.hstack([np.zeros(len(X_clean)), np.ones(len(X_watermarked))])
        
        print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} features each")
        print(f"Clean samples: {np.sum(y == 0)}, Watermarked samples: {np.sum(y == 1)}")
        
        # Save processed data for future use
        if use_cache:
            self.save_processed_data(X, y, cache_filename)
        
        return X, y

def create_cifar100_watermark_dataset(n_samples_per_class=200, method='statistical', use_cache=True):
    """
    Create CIFAR-100 based watermark detection dataset
    
    Args:
        n_samples_per_class: Number of samples per class (total will be 2 * n_samples_per_class)
        method: Feature extraction method
        use_cache: Whether to use cached data
    
    Returns:
        X: Feature matrix
        y: Labels
    """
    processor = CIFARWatermarkProcessor(data_dir="data")
    return processor.load_and_process_cifar100(n_samples_per_class, method, use_cache)

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
    print("Testing CIFAR-100 watermark processor...")
    
    # Test with small dataset first
    X, y = create_cifar100_watermark_dataset(n_samples_per_class=50, use_cache=False)  # Small test
    print(f"Test dataset shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y.astype(int))}")
    
    print("CIFAR-100 watermark processing test successful!")