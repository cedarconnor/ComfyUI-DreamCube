"""
ComfyUI-DreamCube - Visualization Utilities

This module provides tools for visualizing cubemap data in various layouts.

Author: Cedar
License: Apache 2.0
"""

import numpy as np
from typing import Literal, Tuple
import sys
sys.path.append('..')
from core.cubemap import CubemapData


LayoutType = Literal['horizontal', 'cross', 'vertical', 'grid']


def create_cubemap_preview(
    cubemap: CubemapData,
    layout: LayoutType = 'cross',
    use_depth: bool = False,
    add_labels: bool = False
) -> np.ndarray:
    """
    Create a preview image showing all 6 cubemap faces.

    Args:
        cubemap: CubemapData object
        layout: Layout type - 'horizontal', 'cross', 'vertical', or 'grid'
        use_depth: If True, visualize depth instead of RGB
        add_labels: If True, add face name labels (not implemented yet)

    Returns:
        Preview image as numpy array
    """
    if layout == 'horizontal':
        return _create_horizontal_layout(cubemap, use_depth)
    elif layout == 'cross':
        return _create_cross_layout(cubemap, use_depth)
    elif layout == 'vertical':
        return _create_vertical_layout(cubemap, use_depth)
    elif layout == 'grid':
        return _create_grid_layout(cubemap, use_depth)
    else:
        raise ValueError(f"Unknown layout: {layout}")


def _create_horizontal_layout(cubemap: CubemapData, use_depth: bool) -> np.ndarray:
    """
    Create horizontal layout: [left, front, right, back, top, bottom]

    Args:
        cubemap: CubemapData object
        use_depth: Use depth instead of RGB

    Returns:
        Preview image
    """
    face_dict = cubemap.depth_faces if use_depth else cubemap.faces
    resolution = cubemap.resolution

    # Get faces
    faces_order = ['left', 'front', 'right', 'back', 'top', 'bottom']
    faces = []

    for face_name in faces_order:
        face = face_dict[face_name]
        if face is None:
            # Create blank face
            face = np.zeros((resolution, resolution, 3))
        elif use_depth and face.ndim == 3 and face.shape[2] == 1:
            # Convert depth to RGB for visualization
            face = np.repeat(face, 3, axis=2)
        faces.append(face)

    # Concatenate horizontally
    preview = np.concatenate(faces, axis=1)
    return preview


def _create_cross_layout(cubemap: CubemapData, use_depth: bool) -> np.ndarray:
    """
    Create cross layout:
             [top]
        [left][front][right][back]
            [bottom]

    Args:
        cubemap: CubemapData object
        use_depth: Use depth instead of RGB

    Returns:
        Preview image
    """
    face_dict = cubemap.depth_faces if use_depth else cubemap.faces
    resolution = cubemap.resolution

    def get_face(name):
        face = face_dict[name]
        if face is None:
            return np.zeros((resolution, resolution, 3))
        elif use_depth and face.ndim == 3 and face.shape[2] == 1:
            return np.repeat(face, 3, axis=2)
        return face

    # Create blank canvas (3 rows x 4 cols)
    canvas = np.zeros((resolution * 3, resolution * 4, 3))

    # Fill in faces
    # Top row: top face at column 1
    canvas[0:resolution, resolution:2*resolution] = get_face('top')

    # Middle row: left, front, right, back
    canvas[resolution:2*resolution, 0:resolution] = get_face('left')
    canvas[resolution:2*resolution, resolution:2*resolution] = get_face('front')
    canvas[resolution:2*resolution, 2*resolution:3*resolution] = get_face('right')
    canvas[resolution:2*resolution, 3*resolution:4*resolution] = get_face('back')

    # Bottom row: bottom face at column 1
    canvas[2*resolution:3*resolution, resolution:2*resolution] = get_face('bottom')

    return canvas


def _create_vertical_layout(cubemap: CubemapData, use_depth: bool) -> np.ndarray:
    """
    Create vertical layout: [top, front, bottom, back, left, right] stacked vertically.

    Args:
        cubemap: CubemapData object
        use_depth: Use depth instead of RGB

    Returns:
        Preview image
    """
    face_dict = cubemap.depth_faces if use_depth else cubemap.faces
    resolution = cubemap.resolution

    faces_order = ['top', 'front', 'bottom', 'back', 'left', 'right']
    faces = []

    for face_name in faces_order:
        face = face_dict[face_name]
        if face is None:
            face = np.zeros((resolution, resolution, 3))
        elif use_depth and face.ndim == 3 and face.shape[2] == 1:
            face = np.repeat(face, 3, axis=2)
        faces.append(face)

    # Concatenate vertically
    preview = np.concatenate(faces, axis=0)
    return preview


def _create_grid_layout(cubemap: CubemapData, use_depth: bool) -> np.ndarray:
    """
    Create 2x3 grid layout:
        [front][back]
        [left][right]
        [top][bottom]

    Args:
        cubemap: CubemapData object
        use_depth: Use depth instead of RGB

    Returns:
        Preview image
    """
    face_dict = cubemap.depth_faces if use_depth else cubemap.faces
    resolution = cubemap.resolution

    def get_face(name):
        face = face_dict[name]
        if face is None:
            return np.zeros((resolution, resolution, 3))
        elif use_depth and face.ndim == 3 and face.shape[2] == 1:
            return np.repeat(face, 3, axis=2)
        return face

    # Create rows
    row1 = np.concatenate([get_face('front'), get_face('back')], axis=1)
    row2 = np.concatenate([get_face('left'), get_face('right')], axis=1)
    row3 = np.concatenate([get_face('top'), get_face('bottom')], axis=1)

    # Stack rows
    preview = np.concatenate([row1, row2, row3], axis=0)
    return preview


def visualize_seams(
    cubemap: CubemapData,
    threshold: float = 0.05,
    highlight_color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Visualize cubemap seams with error highlighting.

    Args:
        cubemap: CubemapData with depth information
        threshold: Error threshold for highlighting
        highlight_color: RGB color for highlighting errors

    Returns:
        Visualization image with highlighted seams
    """
    if not cubemap.has_depth:
        raise ValueError("Cubemap has no depth information")

    # Create base visualization
    preview = create_cubemap_preview(cubemap, layout='cross', use_depth=True)

    # TODO: Add seam error highlighting
    # This would involve:
    # 1. Computing seam errors
    # 2. Drawing lines at face boundaries
    # 3. Coloring based on error magnitude

    return preview


def create_depth_colormap(
    depth: np.ndarray,
    colormap: str = 'viridis'
) -> np.ndarray:
    """
    Convert depth map to colorized visualization.

    Args:
        depth: Depth array [H, W] or [H, W, 1]
        colormap: Colormap name ('viridis', 'plasma', 'magma', 'inferno')

    Returns:
        RGB image [H, W, 3]
    """
    # Ensure 2D
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    # Normalize to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth_norm = (depth - d_min) / (d_max - d_min)
    else:
        depth_norm = depth

    # Apply simple colormap (simplified version - in practice would use matplotlib)
    if colormap == 'viridis':
        # Blue to green to yellow
        rgb = np.zeros((*depth_norm.shape, 3))
        rgb[:, :, 0] = depth_norm  # Red increases
        rgb[:, :, 1] = np.clip(depth_norm * 1.5, 0, 1)  # Green increases faster
        rgb[:, :, 2] = 1.0 - depth_norm  # Blue decreases
    else:
        # Grayscale fallback
        rgb = np.stack([depth_norm] * 3, axis=-1)

    return rgb
