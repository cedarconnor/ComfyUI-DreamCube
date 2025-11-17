"""
ComfyUI-DreamCube - Validation and Quality Metrics

This module provides quality metrics for evaluating cubemap projections
and depth consistency.

Author: Cedar
License: Apache 2.0
"""

import numpy as np
from typing import Tuple, Dict
import sys
sys.path.append('..')
from core.cubemap import CubemapData


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.

    Args:
        img1: First image
        img2: Second image

    Returns:
        PSNR value in dB
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0  # Perfect match

    max_pixel = 1.0 if img1.max() <= 1.0 else 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index between two images.

    Simplified SSIM implementation.

    Args:
        img1: First image [H, W, C]
        img2: Second image [H, W, C]

    Returns:
        SSIM value in [0, 1]
    """
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Convert to grayscale if needed
    if img1.ndim == 3 and img1.shape[2] == 3:
        img1_gray = np.mean(img1, axis=2)
    else:
        img1_gray = img1.squeeze()

    if img2.ndim == 3 and img2.shape[2] == 3:
        img2_gray = np.mean(img2, axis=2)
    else:
        img2_gray = img2.squeeze()

    # Scale to [0, 255] if needed
    if img1_gray.max() <= 1.0:
        img1_gray *= 255
    if img2_gray.max() <= 1.0:
        img2_gray *= 255

    # Compute means
    mu1 = img1_gray.mean()
    mu2 = img2_gray.mean()

    # Compute variances and covariance
    sigma1_sq = np.var(img1_gray)
    sigma2_sq = np.var(img2_gray)
    sigma12 = np.mean((img1_gray - mu1) * (img2_gray - mu2))

    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim = numerator / denominator
    return float(np.clip(ssim, 0.0, 1.0))


def validate_projection_accuracy(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> Dict[str, float]:
    """
    Validate projection accuracy with multiple metrics.

    Args:
        original: Original equirectangular image
        reconstructed: Reconstructed image after round-trip conversion

    Returns:
        Dictionary with quality metrics
    """
    psnr = calculate_psnr(original, reconstructed)
    ssim = calculate_ssim(original, reconstructed)

    # Mean Absolute Error
    mae = np.mean(np.abs(original - reconstructed))

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2))

    # Max error
    max_error = np.max(np.abs(original - reconstructed))

    return {
        'psnr': psnr,
        'ssim': ssim,
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'is_high_quality': psnr > 45.0 and ssim > 0.99
    }


def validate_cubemap_integrity(cubemap: CubemapData) -> Dict[str, any]:
    """
    Validate cubemap data integrity.

    Args:
        cubemap: CubemapData object

    Returns:
        Dictionary with validation results
    """
    valid, msg = cubemap.validate()

    metrics = {
        'is_valid': valid,
        'message': msg,
        'resolution': cubemap.resolution,
        'has_depth': cubemap.has_depth,
        'all_faces_set': cubemap.all_faces_set(),
        'all_depth_faces_set': cubemap.all_depth_faces_set()
    }

    if valid:
        # Additional quality checks
        face_means = []
        face_stds = []

        for face_name in cubemap.get_face_names():
            face = cubemap.get_face(face_name)
            if face is not None:
                face_means.append(np.mean(face))
                face_stds.append(np.std(face))

        if face_means:
            metrics['mean_brightness'] = np.mean(face_means)
            metrics['std_brightness'] = np.std(face_means)
            metrics['brightness_uniformity'] = 1.0 - (np.std(face_means) / (np.mean(face_means) + 1e-6))

    return metrics


def calculate_seam_quality(
    cubemap: CubemapData,
    threshold: float = 0.05
) -> Dict[str, any]:
    """
    Calculate seam quality metrics for depth continuity.

    Args:
        cubemap: CubemapData with depth information
        threshold: Acceptable error threshold

    Returns:
        Dictionary with seam quality metrics
    """
    if not cubemap.has_depth:
        return {'error': 'No depth information'}

    adjacency_map = cubemap.get_adjacency_map()
    seam_errors = []
    detailed_errors = {}

    for face_name in cubemap.get_face_names():
        depth_face = cubemap.get_depth_face(face_name)
        if depth_face is None:
            continue

        adjacent_faces = adjacency_map[face_name]

        for adj_face, edge in adjacent_faces.items():
            adj_depth = cubemap.get_depth_face(adj_face)
            if adj_depth is None:
                continue

            # Compute boundary error
            error = _compute_edge_error(depth_face, adj_depth, edge)
            seam_errors.append(error)
            detailed_errors[f"{face_name}_{edge}_{adj_face}"] = error

    if not seam_errors:
        return {'error': 'No seams to validate'}

    return {
        'max_error': np.max(seam_errors),
        'mean_error': np.mean(seam_errors),
        'median_error': np.median(seam_errors),
        'std_error': np.std(seam_errors),
        'num_seams': len(seam_errors),
        'is_valid': np.max(seam_errors) < threshold,
        'threshold': threshold,
        'detailed_errors': detailed_errors
    }


def _compute_edge_error(
    depth: np.ndarray,
    adj_depth: np.ndarray,
    edge: str
) -> float:
    """
    Compute depth error at a specific edge.

    Args:
        depth: Current face depth
        adj_depth: Adjacent face depth
        edge: Edge identifier

    Returns:
        Maximum absolute difference
    """
    # Ensure 2D
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if adj_depth.ndim == 3:
        adj_depth = adj_depth[:, :, 0]

    if edge == 'left':
        diff = np.abs(depth[:, 0] - adj_depth[:, -1])
    elif edge == 'right':
        diff = np.abs(depth[:, -1] - adj_depth[:, 0])
    elif edge == 'top':
        diff = np.abs(depth[0, :] - adj_depth[-1, :])
    elif edge == 'bottom':
        diff = np.abs(depth[-1, :] - adj_depth[0, :])
    else:
        return 0.0

    return float(np.max(diff))


def benchmark_performance(
    operation: callable,
    *args,
    num_runs: int = 5,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark performance of an operation.

    Args:
        operation: Function to benchmark
        *args: Arguments to pass to operation
        num_runs: Number of runs for averaging
        **kwargs: Keyword arguments to pass to operation

    Returns:
        Dictionary with timing statistics
    """
    import time

    times = []

    for _ in range(num_runs):
        start_time = time.time()
        operation(*args, **kwargs)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms

    return {
        'mean_time_ms': np.mean(times),
        'median_time_ms': np.median(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'std_time_ms': np.std(times),
        'num_runs': num_runs
    }
