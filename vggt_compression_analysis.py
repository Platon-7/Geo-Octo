#!/usr/bin/env python3
"""
VGGT Token Compression Analysis and Implementation
Analyzes VGGT embeddings to find optimal compression strategy using PCA, SVD, and other techniques.
"""

import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
import seaborn as sns
from typing import Tuple, Dict, Any
import pickle
import os
from tqdm import tqdm

class VGGTCompressor:
    """Intelligent VGGT token compressor using multiple techniques."""
    
    def __init__(self, method='pca', target_size=(64, 256)):
        """
        Args:
            method: 'pca', 'svd', 'random_projection', or 'hybrid'
            target_size: (height, width) of compressed tokens
        """
        self.method = method
        self.target_size = target_size
        self.target_dims = target_size[0] * target_size[1]  # Total compressed dimensions
        self.compressor = None
        self.compression_stats = {}
        
    def analyze_vggt_statistics(self, vggt_tokens_sample: np.ndarray) -> Dict[str, Any]:
        """
        Analyze VGGT token statistics to understand information distribution.
        
        Args:
            vggt_tokens_sample: Sample of VGGT tokens, shape (n_samples, 261, 2048)
        """
        print("🔍 Analyzing VGGT token statistics...")
        
        # Flatten spatial dimensions for analysis
        original_shape = vggt_tokens_sample.shape
        flattened = vggt_tokens_sample.reshape(original_shape[0], -1)  # (n_samples, 261*2048)
        
        stats = {
            'original_shape': original_shape,
            'flattened_shape': flattened.shape,
            'total_original_dims': original_shape[1] * original_shape[2],
            'target_dims': self.target_dims,
            'compression_ratio': (original_shape[1] * original_shape[2]) / self.target_dims,
        }
        
        # Statistical analysis
        stats['mean'] = np.mean(flattened, axis=0)
        stats['std'] = np.std(flattened, axis=0)
        stats['variance'] = np.var(flattened, axis=0)
        
        # Sparsity analysis
        threshold = 0.01  # Consider values below this as "sparse"
        stats['sparsity_ratio'] = np.mean(np.abs(flattened) < threshold)
        
        print(f"📊 Original dimensions: {stats['total_original_dims']}")
        print(f"📊 Target dimensions: {stats['target_dims']}")
        print(f"📊 Compression ratio: {stats['compression_ratio']:.1f}x")
        print(f"📊 Sparsity ratio: {stats['sparsity_ratio']:.3f}")
        
        return stats
    
    def fit_pca_compressor(self, vggt_tokens_sample: np.ndarray) -> Tuple[PCA, Dict]:
        """Fit PCA compressor and analyze explained variance."""
        print("🧮 Fitting PCA compressor...")
        
        # Flatten spatial dimensions
        original_shape = vggt_tokens_sample.shape
        flattened = vggt_tokens_sample.reshape(original_shape[0], -1)
        
        # Fit PCA with target dimensions
        pca = PCA(n_components=self.target_dims)
        pca.fit(flattened)
        
        # Analysis
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        variance_preserved = cumulative_variance[-1]
        
        # Find number of components for different variance thresholds
        variance_thresholds = [0.90, 0.95, 0.99]
        components_needed = {}
        for threshold in variance_thresholds:
            idx = np.argmax(cumulative_variance >= threshold)
            components_needed[f'{threshold:.0%}'] = idx + 1
        
        analysis = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumulative_variance,
            'variance_preserved': variance_preserved,
            'components_needed': components_needed,
            'compression_ratio': flattened.shape[1] / self.target_dims,
        }
        
        print(f"✅ PCA fitted with {self.target_dims} components")
        print(f"📈 Variance preserved: {variance_preserved:.3f}")
        print(f"📈 Components needed for 95% variance: {components_needed['95%']}")
        
        return pca, analysis
    
    def fit_svd_compressor(self, vggt_tokens_sample: np.ndarray) -> Tuple[TruncatedSVD, Dict]:
        """Fit SVD compressor for sparse data."""
        print("🧮 Fitting SVD compressor...")
        
        original_shape = vggt_tokens_sample.shape
        flattened = vggt_tokens_sample.reshape(original_shape[0], -1)
        
        svd = TruncatedSVD(n_components=self.target_dims, random_state=42)
        svd.fit(flattened)
        
        cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
        
        analysis = {
            'explained_variance_ratio': svd.explained_variance_ratio_,
            'cumulative_variance': cumulative_variance,
            'variance_preserved': cumulative_variance[-1],
        }
        
        print(f"✅ SVD fitted with {self.target_dims} components")
        print(f"📈 Variance preserved: {analysis['variance_preserved']:.3f}")
        
        return svd, analysis
    
    def fit_hybrid_compressor(self, vggt_tokens_sample: np.ndarray) -> Tuple[Dict, Dict]:
        """Fit hybrid compressor: spatial + channel compression."""
        print("🧮 Fitting hybrid compressor...")
        
        original_shape = vggt_tokens_sample.shape  # (n_samples, 261, 2048)
        
        # Strategy: Compress spatial (261) and channel (2048) dimensions separately
        target_spatial, target_channels = self.target_size
        
        # 1. Spatial compression: average pool or PCA across spatial dimension
        spatial_pooled = np.mean(vggt_tokens_sample, axis=1)  # (n_samples, 2048)
        
        # 2. Channel compression: PCA across channel dimension
        channel_pca = PCA(n_components=target_channels)
        compressed_channels = channel_pca.fit_transform(spatial_pooled)  # (n_samples, target_channels)
        
        # 3. Spatial reconstruction: Use top spatial locations
        spatial_variance = np.var(vggt_tokens_sample, axis=(0, 2))  # Variance across samples and channels
        top_spatial_indices = np.argsort(spatial_variance)[-target_spatial:]  # Top spatial locations
        
        compressor = {
            'channel_pca': channel_pca,
            'top_spatial_indices': top_spatial_indices,
            'method': 'hybrid'
        }
        
        # Test compression
        test_compressed = self.compress_hybrid(vggt_tokens_sample[:10], compressor)
        
        analysis = {
            'channel_variance_preserved': np.sum(channel_pca.explained_variance_ratio_),
            'spatial_compression_ratio': 261 / target_spatial,
            'channel_compression_ratio': 2048 / target_channels,
            'total_compression_ratio': (261 * 2048) / (target_spatial * target_channels),
            'test_output_shape': test_compressed.shape,
        }
        
        print(f"✅ Hybrid compressor fitted")
        print(f"📈 Channel variance preserved: {analysis['channel_variance_preserved']:.3f}")
        print(f"📈 Total compression: {analysis['total_compression_ratio']:.1f}x")
        
        return compressor, analysis
    
    def compress_hybrid(self, vggt_tokens: np.ndarray, compressor: Dict) -> np.ndarray:
        """Apply hybrid compression."""
        channel_pca = compressor['channel_pca']
        top_spatial_indices = compressor['top_spatial_indices']
        
        # 1. Select top spatial locations
        spatial_selected = vggt_tokens[:, top_spatial_indices, :]  # (n_samples, target_spatial, 2048)
        
        # 2. Flatten and apply channel PCA
        n_samples, spatial_dim, channel_dim = spatial_selected.shape
        flattened = spatial_selected.reshape(n_samples * spatial_dim, channel_dim)
        compressed_channels = channel_pca.transform(flattened)
        
        # 3. Reshape back
        target_spatial, target_channels = self.target_size
        compressed = compressed_channels.reshape(n_samples, target_spatial, target_channels)
        
        return compressed.astype(np.float16)
    
    def fit_compressor(self, vggt_tokens_sample: np.ndarray) -> None:
        """Fit the specified compressor type."""
        self.compression_stats = self.analyze_vggt_statistics(vggt_tokens_sample)
        
        if self.method == 'pca':
            self.compressor, analysis = self.fit_pca_compressor(vggt_tokens_sample)
        elif self.method == 'svd':
            self.compressor, analysis = self.fit_svd_compressor(vggt_tokens_sample)
        elif self.method == 'hybrid':
            self.compressor, analysis = self.fit_hybrid_compressor(vggt_tokens_sample)
        else:
            raise ValueError(f"Unknown compression method: {self.method}")
        
        self.compression_stats.update(analysis)
    
    def compress(self, vggt_tokens: np.ndarray) -> np.ndarray:
        """Compress VGGT tokens using fitted compressor."""
        if self.compressor is None:
            raise ValueError("Compressor not fitted. Call fit_compressor() first.")
        
        if self.method == 'hybrid':
            return self.compress_hybrid(vggt_tokens, self.compressor)
        else:
            # PCA or SVD
            original_shape = vggt_tokens.shape
            flattened = vggt_tokens.reshape(original_shape[0], -1)
            compressed_flat = self.compressor.transform(flattened)
            
            # Reshape to target size
            target_h, target_w = self.target_size
            compressed = compressed_flat.reshape(original_shape[0], target_h, target_w)
            
            return compressed.astype(np.float16)
    
    def save_compressor(self, filepath: str) -> None:
        """Save fitted compressor to disk."""
        data = {
            'compressor': self.compressor,
            'method': self.method,
            'target_size': self.target_size,
            'compression_stats': self.compression_stats,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Compressor saved to {filepath}")
    
    @classmethod
    def load_compressor(cls, filepath: str) -> 'VGGTCompressor':
        """Load fitted compressor from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        compressor = cls(data['method'], data['target_size'])
        compressor.compressor = data['compressor']
        compressor.compression_stats = data['compression_stats']
        
        print(f"📂 Compressor loaded from {filepath}")
        return compressor

def compare_compression_methods(vggt_sample: np.ndarray, target_size=(64, 256)) -> Dict:
    """Compare different compression methods on the same data."""
    print("🔬 Comparing compression methods...")
    
    methods = ['pca', 'svd', 'hybrid']
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} ---")
        compressor = VGGTCompressor(method=method, target_size=target_size)
        compressor.fit_compressor(vggt_sample)
        
        # Test compression
        test_compressed = compressor.compress(vggt_sample[:100])
        
        results[method] = {
            'compressor': compressor,
            'compressed_shape': test_compressed.shape,
            'compression_ratio': compressor.compression_stats.get('compression_ratio', 0),
            'variance_preserved': compressor.compression_stats.get('variance_preserved', 0),
            'memory_reduction': (261 * 2048 * 2) / (target_size[0] * target_size[1] * 2),  # bytes
        }
        
        print(f"✅ {method}: {results[method]['variance_preserved']:.3f} variance preserved")
    
    return results

def create_vggt_sample_from_dataset(dataset_path: str, num_samples: int = 1000) -> np.ndarray:
    """Extract VGGT token samples from existing dataset for analysis."""
    print(f"📦 Extracting {num_samples} VGGT samples from {dataset_path}")
    
    try:
        import tensorflow_datasets as tfds
        
        # Load dataset
        builder = tfds.builder('libero_object_vggt', data_dir=dataset_path)
        ds = builder.as_dataset(split='train').take(100)  # Take first 100 episodes
        
        vggt_samples = []
        for episode in tqdm(ds, desc="Extracting VGGT tokens"):
            for step in episode['steps']:
                vggt_token = step['observation']['vggt_tokens'].numpy()
                vggt_samples.append(vggt_token)
                
                if len(vggt_samples) >= num_samples:
                    break
            if len(vggt_samples) >= num_samples:
                break
        
        vggt_array = np.stack(vggt_samples[:num_samples])
        print(f"✅ Extracted {vggt_array.shape[0]} samples, shape: {vggt_array.shape}")
        return vggt_array
        
    except Exception as e:
        print(f"❌ Could not extract from dataset: {e}")
        print("🎲 Generating synthetic VGGT samples for testing...")
        return np.random.randn(num_samples, 261, 2048).astype(np.float16)

if __name__ == "__main__":
    print("🚀 VGGT Compression Analysis")
    
    # Configuration
    dataset_path = "/scratch-shared/tmp.cwkV8vOvfY/libero_vggt_datasets2"
    target_size = (64, 256)  # Much smaller than (261, 2048)
    
    # Extract sample data
    vggt_sample = create_vggt_sample_from_dataset(dataset_path, num_samples=1000)
    
    # Compare compression methods
    results = compare_compression_methods(vggt_sample, target_size)
    
    # Show results
    print("\n" + "="*60)
    print("📊 COMPRESSION COMPARISON RESULTS")
    print("="*60)
    
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Compression ratio: {result['compression_ratio']:.1f}x")
        print(f"  Variance preserved: {result['variance_preserved']:.3f}")
        print(f"  Memory reduction: {result['memory_reduction']:.1f}x")
        print(f"  Output shape: {result['compressed_shape']}")
    
    # Recommend best method
    best_method = max(results.keys(), key=lambda k: results[k]['variance_preserved'])
    print(f"\n🏆 RECOMMENDED: {best_method.upper()}")
    print(f"   Preserves {results[best_method]['variance_preserved']:.3f} of original information")
    print(f"   Reduces memory by {results[best_method]['memory_reduction']:.1f}x")
    
    # Save best compressor
    best_compressor = results[best_method]['compressor']
    best_compressor.save_compressor(f"vggt_compressor_{best_method}.pkl")
    
    print(f"\n💡 Use this compressor in your dataset creation script!")
    print(f"   Original: (261, 2048) = {261*2048:,} dimensions")
    print(f"   Compressed: {target_size} = {target_size[0]*target_size[1]:,} dimensions")
    print(f"   Memory per timestep: {261*2048*2/1024:.1f}KB → {target_size[0]*target_size[1]*2/1024:.1f}KB")