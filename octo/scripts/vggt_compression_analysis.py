#!/usr/bin/env python3
"""
VGGT Token Compression Analysis and Implementation using PCA.
"""
import numpy as np
import pickle
from sklearn.decomposition import PCA
from typing import Tuple, Dict

class VGGTCompressor:
    """Intelligent VGGT token compressor using Principal Component Analysis (PCA)."""
    
    def __init__(self, target_size=(32, 48)):
        self.method = 'pca'
        self.target_size = target_size
        self.target_dims = target_size[0] * target_size[1]
        self.compressor = None
        self.compression_stats = {}
        
    def fit_compressor(self, vggt_tokens_sample: np.ndarray):
        """Fit the PCA compressor and analyze explained variance."""
        print("🧮 Fitting PCA compressor...")
        original_shape = vggt_tokens_sample.shape
        flattened = vggt_tokens_sample.reshape(original_shape[0], -1)
        
        if flattened.shape[0] < self.target_dims:
            raise ValueError(f"Number of samples ({flattened.shape[0]}) must be >= target dimensions ({self.target_dims}) for PCA.")

        pca = PCA(n_components=self.target_dims)
        pca.fit(flattened)
        
        variance_preserved = np.sum(pca.explained_variance_ratio_)
        
        self.compressor = pca
        self.compression_stats = {
            'variance_preserved': variance_preserved,
            'compression_ratio': flattened.shape[1] / self.target_dims
        }
        
        print(f"✅ PCA fitted with {self.target_dims} components.")
        print(f"📈 Variance preserved: {variance_preserved:.4f}")

    def compress(self, vggt_tokens: np.ndarray) -> np.ndarray:
        """Compress VGGT tokens using the fitted PCA compressor."""
        if self.compressor is None:
            raise ValueError("Compressor has not been fitted. Call fit_compressor() first.")
        
        original_shape = vggt_tokens.shape
        flattened = vggt_tokens.reshape(original_shape[0], -1)
        compressed_flat = self.compressor.transform(flattened)
        
        target_h, target_w = self.target_size
        compressed = compressed_flat.reshape(original_shape[0], target_h, target_w)
        
        return compressed.astype(np.float16)

    def save_compressor(self, filepath: str):
        """Save fitted compressor to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Compressor saved to {filepath}")
    
    @classmethod
    def load_compressor(cls, filepath: str) -> 'VGGTCompressor':
        """Load fitted compressor from disk."""
        with open(filepath, 'rb') as f:
            compressor = pickle.load(f)
        print(f"📂 Compressor loaded from {filepath}")
        return compressor