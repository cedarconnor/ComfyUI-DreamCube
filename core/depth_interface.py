"""
ComfyUI-DreamCube - Generic Depth Model Interface

This module provides a generic interface for applying depth estimation
to cubemap faces, compatible with any ComfyUI depth node output.

Author: Cedar
License: Apache 2.0
"""

import numpy as np
import torch
from typing import Optional, Callable, List, Dict

from .cubemap import CubemapData


class DepthModelInterface:
    """
    Generic interface for applying any depth model to cubemap faces.

    This class doesn't contain depth models itself - it provides utilities
    for working with depth outputs from ComfyUI depth nodes.
    """

    def __init__(self, normalization: str = 'global'):
        """
        Initialize depth interface.

        Args:
            normalization: 'global', 'per_face', or 'adaptive'
                - global: Normalize all faces to same depth range
                - per_face: Normalize each face independently
                - adaptive: Use model's native output range
        """
        self.normalization = normalization
        self.supported_formats = ['comfyui_depth_anything', 'midas', 'custom']

    def apply_depth_to_face(
        self,
        cubemap: CubemapData,
        face_name: str,
        depth_map: np.ndarray
    ) -> CubemapData:
        """
        Apply depth estimation to a single cubemap face.

        Args:
            cubemap: CubemapData object
            face_name: Which face to process ('front', 'back', etc.)
            depth_map: Depth map from any ComfyUI depth node [H, W] or [H, W, 1]

        Returns:
            Updated cubemap with depth assigned to specified face

        Raises:
            ValueError: If depth_map size doesn't match face resolution
        """
        # Handle different depth map formats
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()

        # Ensure 2D or 3D array
        if depth_map.ndim == 4:  # [B, H, W, C]
            depth_map = depth_map[0]  # Take first batch

        if depth_map.ndim == 3 and depth_map.shape[2] > 1:
            # If multi-channel, take first channel (grayscale depth)
            depth_map = depth_map[:, :, 0]

        # Normalize depth
        depth_map = self._normalize_depth(depth_map)

        # Set depth face
        cubemap.set_depth_face(face_name, depth_map)

        return cubemap

    def apply_depth_to_all_faces(
        self,
        cubemap: CubemapData,
        depth_maps: Dict[str, np.ndarray]
    ) -> CubemapData:
        """
        Apply depth maps to all 6 cubemap faces.

        Args:
            cubemap: CubemapData object
            depth_maps: Dictionary mapping face names to depth arrays

        Returns:
            Updated cubemap with all depth faces set
        """
        all_depths = []

        # First pass: collect all depths for global normalization
        for face_name in cubemap.get_face_names():
            if face_name not in depth_maps:
                raise ValueError(f"Missing depth for face: {face_name}")

            depth = depth_maps[face_name]
            if isinstance(depth, torch.Tensor):
                depth = depth.cpu().numpy()

            # Handle shape variations
            if depth.ndim == 4:
                depth = depth[0]
            if depth.ndim == 3 and depth.shape[2] > 1:
                depth = depth[:, :, 0]
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth[:, :, 0]

            all_depths.append(depth)

        # Global normalization if needed
        if self.normalization == 'global':
            global_min = min(d.min() for d in all_depths)
            global_max = max(d.max() for d in all_depths)

            # Normalize all depths to [0, 1]
            for i, depth in enumerate(all_depths):
                if global_max - global_min > 1e-6:
                    all_depths[i] = (depth - global_min) / (global_max - global_min)

        # Second pass: assign normalized depths
        for face_name, depth in zip(cubemap.get_face_names(), all_depths):
            if self.normalization == 'per_face':
                depth = self._normalize_depth(depth)

            cubemap.set_depth_face(face_name, depth)

        return cubemap

    def batch_process_faces(
        self,
        cubemap: CubemapData,
        depth_estimator: Callable[[np.ndarray], np.ndarray]
    ) -> CubemapData:
        """
        Process all 6 faces through a depth estimation callback.

        This is useful when you want to programmatically call a depth
        estimator on each face.

        Args:
            cubemap: Input cubemap with RGB faces
            depth_estimator: Function that takes RGB image and returns depth map

        Returns:
            CubemapData with depth_faces populated
        """
        depth_maps = {}

        for face_name in cubemap.get_face_names():
            face_rgb = cubemap.get_face(face_name)
            if face_rgb is None:
                raise ValueError(f"Face {face_name} is not set")

            # Call depth estimator
            depth_map = depth_estimator(face_rgb)

            depth_maps[face_name] = depth_map

        # Apply all depths
        return self.apply_depth_to_all_faces(cubemap, depth_maps)

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Normalize depth to [0, 1] range.

        Args:
            depth: Raw depth map

        Returns:
            Normalized depth map
        """
        d_min, d_max = depth.min(), depth.max()

        # Avoid division by zero
        if d_max - d_min < 1e-6:
            return np.zeros_like(depth)

        return (depth - d_min) / (d_max - d_min)

    def convert_comfyui_depth(self, depth_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert ComfyUI depth node output to numpy array.

        ComfyUI typically outputs depth as [B, H, W, C] tensors.

        Args:
            depth_tensor: Depth tensor from ComfyUI node

        Returns:
            Numpy depth array [H, W]
        """
        # Convert to numpy
        if isinstance(depth_tensor, torch.Tensor):
            depth_np = depth_tensor.cpu().numpy()
        else:
            depth_np = depth_tensor

        # Handle batch dimension
        if depth_np.ndim == 4:
            depth_np = depth_np[0]  # Take first batch item

        # Handle channel dimension
        if depth_np.ndim == 3:
            if depth_np.shape[2] == 1:
                depth_np = depth_np[:, :, 0]
            else:
                # Take first channel or average
                depth_np = depth_np[:, :, 0]

        return depth_np

    def merge_rgb_depth(
        self,
        cubemap_rgb: CubemapData,
        cubemap_depth: CubemapData
    ) -> CubemapData:
        """
        Merge RGB and depth cubemaps into a single RGBD cubemap.

        Args:
            cubemap_rgb: CubemapData with RGB faces
            cubemap_depth: CubemapData with depth faces

        Returns:
            New CubemapData with combined RGBD faces
        """
        if cubemap_rgb.resolution != cubemap_depth.resolution:
            raise ValueError("RGB and depth cubemaps must have same resolution")

        # Create new cubemap
        merged = CubemapData(resolution=cubemap_rgb.resolution)

        # Merge faces
        for face_name in cubemap_rgb.get_face_names():
            rgb_face = cubemap_rgb.get_face(face_name)
            depth_face = cubemap_depth.get_depth_face(face_name)

            if rgb_face is None or depth_face is None:
                raise ValueError(f"Missing data for face: {face_name}")

            # Copy RGB
            merged.set_face(face_name, rgb_face.copy())

            # Copy depth
            merged.set_depth_face(face_name, depth_face.copy())

        return merged

    def extract_depth_channel(self, cubemap: CubemapData) -> CubemapData:
        """
        Extract depth channel as separate cubemap for visualization.

        Args:
            cubemap: CubemapData with depth information

        Returns:
            New CubemapData with depth as RGB (for preview)
        """
        if not cubemap.has_depth:
            raise ValueError("Cubemap has no depth information")

        # Create new cubemap for depth visualization
        depth_cubemap = CubemapData(resolution=cubemap.resolution)

        for face_name in cubemap.get_face_names():
            depth_face = cubemap.get_depth_face(face_name)
            if depth_face is None:
                continue

            # Convert depth to RGB (grayscale visualization)
            if depth_face.ndim == 2:
                depth_rgb = np.stack([depth_face] * 3, axis=-1)
            else:
                # Already has channel dimension
                d = depth_face[:, :, 0]
                depth_rgb = np.stack([d] * 3, axis=-1)

            depth_cubemap.set_face(face_name, depth_rgb)

        return depth_cubemap
