#!/usr/bin/env python3
"""
VGGT Token Compression Analysis and Implementation using PCA.
"""
import numpy as np
import pickle
from sklearn.decomposition import PCA
from typing import Tuple, Dict

class VGGTCompressor:
    """VGGT token compressor using PCA on the feature dimension.

    Expects per-sample tokens shaped (num_tokens, feature_dim), e.g., (64, 2048),
    and compresses to (num_tokens, target_feature_dim), e.g., (64, 512).
    """

    def __init__(self, target_size=(64, 512)):
        self.method = 'pca'
        self.target_size = target_size  # (num_tokens, target_feature_dim)
        self.num_tokens = target_size[0]
        self.target_feature_dim = target_size[1]
        self.pca: PCA | None = None
        self.compression_stats: Dict = {}
        # Mirror attribute name used by callers for logging, populated after fit
        self.explained_variance_ratio_ = None

    def fit_compressor(self, vggt_tokens_sample: np.ndarray):
        """Fit PCA along feature dimension using stacked tokens across samples.

        vggt_tokens_sample: [num_samples, num_tokens, feature_dim], e.g., [N, 64, 2048]
        We reshape to [N * num_tokens, feature_dim] and fit PCA(n_components=target_feature_dim).
        """
        print("🧮 Fitting PCA compressor (token-wise, feature-dim reduction)...")

        if vggt_tokens_sample.ndim != 3 or vggt_tokens_sample.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected samples of shape [*, {self.num_tokens}, *], got {vggt_tokens_sample.shape}")

        num_samples, num_tokens, feature_dim = vggt_tokens_sample.shape
        if self.target_feature_dim > feature_dim:
            raise ValueError(
                f"target_feature_dim ({self.target_feature_dim}) must be <= feature_dim ({feature_dim})")

        # Stack tokens across samples for robust PCA fit
        X = vggt_tokens_sample.reshape(num_samples * num_tokens, feature_dim)

        # PCA on feature dimension to target_feature_dim
        pca = PCA(n_components=self.target_feature_dim)
        pca.fit(X)

        variance_preserved = float(np.sum(pca.explained_variance_ratio_))

        self.pca = pca
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.compression_stats = {
            'variance_preserved': variance_preserved,
            'compression_ratio': feature_dim / float(self.target_feature_dim),
            'num_fit_samples': int(X.shape[0]),
        }

        print(f"✅ PCA fitted with {self.target_feature_dim} components.")
        print(f"📈 Variance preserved: {variance_preserved:.4f}")

    def compress(self, vggt_tokens: np.ndarray) -> np.ndarray:
        """Compress per-sample tokens of shape [T, num_tokens, feature_dim] to [T, num_tokens, target_feature_dim]."""
        if self.pca is None:
            raise ValueError("Compressor has not been fitted. Call fit_compressor() first.")

        if vggt_tokens.ndim != 3 or vggt_tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected tokens of shape [T, {self.num_tokens}, *], got {vggt_tokens.shape}")

        T, num_tokens, feature_dim = vggt_tokens.shape
        X = vggt_tokens.reshape(T * num_tokens, feature_dim)
        Z = self.pca.transform(X)  # [T * num_tokens, target_feature_dim]
        compressed = Z.reshape(T, num_tokens, self.target_feature_dim)
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