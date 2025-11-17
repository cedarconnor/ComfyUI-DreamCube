"""
ComfyUI-DreamCube - Depth Consistency Enforcement

This module implements algorithms for ensuring depth continuity
across cubemap face boundaries.

Author: Cedar
License: Apache 2.0
"""

import numpy as np
from typing import Tuple, Dict
from scipy.ndimage import gaussian_filter

from .cubemap import CubemapData


class DepthConsistencyEnforcer:
    """
    Ensures depth continuity across cubemap face boundaries.

    This class implements various blending and smoothing techniques
    to eliminate visible seams at face edges.
    """

    def __init__(self, boundary_width: int = 16, method: str = 'blend'):
        """
        Initialize consistency enforcer.

        Args:
            boundary_width: Width of boundary region to blend (in pixels)
            method: Blending method - 'blend', 'gradient', or 'poisson'
        """
        self.boundary_width = boundary_width
        self.method = method

    def enforce_consistency(
        self,
        cubemap: CubemapData,
        iterations: int = 1
    ) -> CubemapData:
        """
        Enforce depth consistency across all face boundaries.

        Args:
            cubemap: CubemapData with depth_faces populated
            iterations: Number of smoothing iterations

        Returns:
            CubemapData with smoothed depth at boundaries
        """
        if not cubemap.has_depth:
            raise ValueError("Cubemap has no depth information")

        adjacency_map = cubemap.get_adjacency_map()

        # Iterative smoothing
        for _ in range(iterations):
            # Create a copy to avoid in-place modifications affecting other faces
            new_depth_faces = {name: depth.copy() for name, depth in cubemap.depth_faces.items()}

            for face_name in cubemap.get_face_names():
                depth_face = new_depth_faces[face_name]
                adjacent_faces = adjacency_map[face_name]

                for adj_face_name, edge in adjacent_faces.items():
                    adj_depth = cubemap.depth_faces[adj_face_name]

                    # Blend depths at boundary
                    blended_depth = self._blend_boundary(
                        depth_face, adj_depth, edge, adj_face_name
                    )

                    new_depth_faces[face_name] = blended_depth

            # Update cubemap with blended depths
            for face_name, depth in new_depth_faces.items():
                cubemap.depth_faces[face_name] = depth

        return cubemap

    def _blend_boundary(
        self,
        depth: np.ndarray,
        adj_depth: np.ndarray,
        edge: str,
        adj_face_name: str
    ) -> np.ndarray:
        """
        Blend depth values at a face boundary.

        Uses weighted average based on distance from boundary.

        Args:
            depth: Current face depth [H, W] or [H, W, 1]
            adj_depth: Adjacent face depth
            edge: Edge identifier ('left', 'right', 'top', 'bottom')
            adj_face_name: Name of adjacent face

        Returns:
            Blended depth array
        """
        # Ensure 2D arrays
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if adj_depth.ndim == 3:
            adj_depth = adj_depth[:, :, 0]

        H, W = depth.shape
        w = min(self.boundary_width, H // 4, W // 4)  # Limit boundary width

        # Create result copy
        result = depth.copy()

        # Create blend weights (1.0 at center, 0.5 at boundary)
        weights = np.linspace(0.5, 1.0, w)

        if edge == 'left':
            # Blend left edge with adjacent face's corresponding edge
            for i in range(w):
                alpha = weights[i]
                result[:, i] = alpha * depth[:, i] + (1 - alpha) * adj_depth[:, -(w-i)]

        elif edge == 'right':
            # Blend right edge
            for i in range(w):
                alpha = weights[w - i - 1]
                result[:, -(i+1)] = alpha * depth[:, -(i+1)] + (1 - alpha) * adj_depth[:, i]

        elif edge == 'top':
            # Blend top edge
            for i in range(w):
                alpha = weights[i]
                result[i, :] = alpha * depth[i, :] + (1 - alpha) * adj_depth[-(w-i), :]

        elif edge == 'bottom':
            # Blend bottom edge
            for i in range(w):
                alpha = weights[w - i - 1]
                result[-(i+1), :] = alpha * depth[-(i+1), :] + (1 - alpha) * adj_depth[i, :]

        # Restore channel dimension if needed
        if len(result.shape) == 2:
            result = result[:, :, np.newaxis]

        return result

    def validate_seams(
        self,
        cubemap: CubemapData,
        threshold: float = 0.05
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Validate depth continuity at seams.

        Args:
            cubemap: CubemapData with depth information
            threshold: Maximum acceptable depth difference

        Returns:
            Tuple of (is_valid, max_error, error_dict)
            - is_valid: True if all seams are below threshold
            - max_error: Maximum depth difference across all seams
            - error_dict: Dictionary of errors per face pair
        """
        if not cubemap.has_depth:
            raise ValueError("Cubemap has no depth information")

        adjacency_map = cubemap.get_adjacency_map()
        max_error = 0.0
        error_dict = {}

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
                error = self._compute_boundary_error(depth_face, adj_depth, edge)
                max_error = max(max_error, error)

                edge_key = f"{face_name}_{edge}_{adj_face}"
                error_dict[edge_key] = error

        is_valid = max_error < threshold
        return is_valid, max_error, error_dict

    def _compute_boundary_error(
        self,
        depth: np.ndarray,
        adj_depth: np.ndarray,
        edge: str
    ) -> float:
        """
        Compute maximum depth difference at a boundary.

        Args:
            depth: Current face depth
            adj_depth: Adjacent face depth
            edge: Edge identifier

        Returns:
            Maximum absolute difference at boundary
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

    def smooth_globally(
        self,
        cubemap: CubemapData,
        sigma: float = 1.0
    ) -> CubemapData:
        """
        Apply global Gaussian smoothing to all depth faces.

        This can help reduce high-frequency noise while preserving
        overall depth structure.

        Args:
            cubemap: CubemapData with depth information
            sigma: Gaussian kernel standard deviation

        Returns:
            Cubemap with smoothed depth
        """
        if not cubemap.has_depth:
            raise ValueError("Cubemap has no depth information")

        for face_name in cubemap.get_face_names():
            depth_face = cubemap.get_depth_face(face_name)
            if depth_face is None:
                continue

            # Apply Gaussian filter
            if depth_face.ndim == 3:
                smoothed = gaussian_filter(depth_face[:, :, 0], sigma=sigma)
                smoothed = smoothed[:, :, np.newaxis]
            else:
                smoothed = gaussian_filter(depth_face, sigma=sigma)

            cubemap.depth_faces[face_name] = smoothed

        return cubemap

    def align_depth_scales(
        self,
        cubemap: CubemapData,
        reference_face: str = 'front'
    ) -> CubemapData:
        """
        Align depth scales across all faces to match a reference face.

        This helps when different faces have different depth ranges.

        Args:
            cubemap: CubemapData with depth information
            reference_face: Face to use as reference

        Returns:
            Cubemap with aligned depth scales
        """
        if not cubemap.has_depth:
            raise ValueError("Cubemap has no depth information")

        ref_depth = cubemap.get_depth_face(reference_face)
        if ref_depth is None:
            raise ValueError(f"Reference face {reference_face} has no depth")

        # Get reference statistics
        if ref_depth.ndim == 3:
            ref_depth = ref_depth[:, :, 0]

        ref_mean = np.mean(ref_depth)
        ref_std = np.std(ref_depth)

        # Align other faces
        for face_name in cubemap.get_face_names():
            if face_name == reference_face:
                continue

            depth_face = cubemap.get_depth_face(face_name)
            if depth_face is None:
                continue

            if depth_face.ndim == 3:
                depth_2d = depth_face[:, :, 0]
            else:
                depth_2d = depth_face

            # Standardize and rescale
            face_mean = np.mean(depth_2d)
            face_std = np.std(depth_2d)

            if face_std > 1e-6:
                aligned = (depth_2d - face_mean) / face_std
                aligned = aligned * ref_std + ref_mean
            else:
                aligned = depth_2d

            # Update with channel dimension
            if depth_face.ndim == 3:
                aligned = aligned[:, :, np.newaxis]

            cubemap.depth_faces[face_name] = aligned

        return cubemap

    def get_seam_statistics(
        self,
        cubemap: CubemapData
    ) -> Dict[str, float]:
        """
        Compute statistics about seam quality.

        Args:
            cubemap: CubemapData with depth information

        Returns:
            Dictionary with seam quality metrics
        """
        if not cubemap.has_depth:
            raise ValueError("Cubemap has no depth information")

        is_valid, max_error, error_dict = self.validate_seams(cubemap, threshold=1.0)

        errors = list(error_dict.values())

        return {
            'max_error': max_error,
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'num_seams': len(errors),
            'is_valid_0.05': max_error < 0.05,
            'is_valid_0.03': max_error < 0.03,
            'is_valid_0.01': max_error < 0.01
        }
