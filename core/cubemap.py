"""
ComfyUI-DreamCube - Cubemap Data Structure

This module defines the core CubemapData class for storing and manipulating
6-face cubemap representations of 360° panoramic images.

Author: Cedar
License: Apache 2.0
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class CubemapData:
    """
    Represents a 6-face cubemap with optional depth channel.

    Attributes:
        resolution: Resolution of each square face (e.g., 1024 for 1024x1024)
        faces: Dictionary mapping face names to RGB image data [H, W, C]
        depth_faces: Dictionary mapping face names to depth maps [H, W, 1]
        has_depth: Flag indicating if depth information is present
    """

    resolution: int
    faces: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    depth_faces: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    has_depth: bool = False

    def __post_init__(self):
        """Initialize face dictionaries with None values."""
        face_names = ['front', 'back', 'left', 'right', 'top', 'bottom']

        if not self.faces:
            self.faces = {name: None for name in face_names}

        if not self.depth_faces:
            self.depth_faces = {name: None for name in face_names}

    def get_face_names(self) -> List[str]:
        """Return list of all face names in standard order."""
        return ['front', 'back', 'left', 'right', 'top', 'bottom']

    def get_adjacency_map(self) -> Dict[str, Dict[str, str]]:
        """
        Get adjacency mapping between cube faces.

        Returns a dictionary where each face maps to its adjacent faces
        and their corresponding edge orientations.

        Example:
            'front': {
                'right': 'left',   # Front's right edge touches Right's left edge
                'left': 'right',   # Front's left edge touches Left's right edge
                'top': 'bottom',   # Front's top edge touches Top's bottom edge
                'bottom': 'top'    # Front's bottom edge touches Bottom's top edge
            }

        Returns:
            Dictionary mapping face names to their adjacency information
        """
        return {
            'front': {
                'right': 'left',
                'left': 'right',
                'top': 'bottom',
                'bottom': 'top'
            },
            'back': {
                'right': 'right',
                'left': 'left',
                'top': 'top',
                'bottom': 'bottom'
            },
            'left': {
                'front': 'right',
                'back': 'right',
                'top': 'left',
                'bottom': 'left'
            },
            'right': {
                'front': 'left',
                'back': 'left',
                'top': 'right',
                'bottom': 'right'
            },
            'top': {
                'front': 'top',
                'back': 'bottom',
                'left': 'top',
                'right': 'top'
            },
            'bottom': {
                'front': 'bottom',
                'back': 'top',
                'left': 'bottom',
                'right': 'bottom'
            }
        }

    def set_face(self, face_name: str, image: np.ndarray):
        """
        Set RGB image data for a specific face.

        Args:
            face_name: Name of the face ('front', 'back', etc.)
            image: Numpy array of shape [H, W, C]

        Raises:
            ValueError: If face_name is invalid or image shape is incorrect
        """
        if face_name not in self.faces:
            raise ValueError(f"Invalid face name: {face_name}")

        if image.shape[0] != self.resolution or image.shape[1] != self.resolution:
            raise ValueError(
                f"Image size {image.shape[:2]} doesn't match "
                f"cubemap resolution {self.resolution}"
            )

        self.faces[face_name] = image

    def set_depth_face(self, face_name: str, depth: np.ndarray):
        """
        Set depth data for a specific face.

        Args:
            face_name: Name of the face ('front', 'back', etc.)
            depth: Numpy array of shape [H, W] or [H, W, 1]

        Raises:
            ValueError: If face_name is invalid or depth shape is incorrect
        """
        if face_name not in self.depth_faces:
            raise ValueError(f"Invalid face name: {face_name}")

        # Handle both [H, W] and [H, W, 1] shapes
        if depth.ndim == 2:
            depth = depth[..., np.newaxis]

        if depth.shape[0] != self.resolution or depth.shape[1] != self.resolution:
            raise ValueError(
                f"Depth size {depth.shape[:2]} doesn't match "
                f"cubemap resolution {self.resolution}"
            )

        self.depth_faces[face_name] = depth
        self.has_depth = True

    def get_face(self, face_name: str) -> Optional[np.ndarray]:
        """Get RGB image data for a specific face."""
        return self.faces.get(face_name)

    def get_depth_face(self, face_name: str) -> Optional[np.ndarray]:
        """Get depth data for a specific face."""
        return self.depth_faces.get(face_name)

    def all_faces_set(self) -> bool:
        """Check if all RGB faces have been set."""
        return all(face is not None for face in self.faces.values())

    def all_depth_faces_set(self) -> bool:
        """Check if all depth faces have been set."""
        return all(face is not None for face in self.depth_faces.values())

    def validate(self) -> Tuple[bool, str]:
        """
        Validate cubemap data integrity.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if all faces are set
        if not self.all_faces_set():
            missing = [name for name, face in self.faces.items() if face is None]
            return False, f"Missing RGB faces: {', '.join(missing)}"

        # Check face resolutions
        for name, face in self.faces.items():
            if face.shape[0] != self.resolution or face.shape[1] != self.resolution:
                return False, f"Face {name} has incorrect resolution: {face.shape[:2]}"

        # Check depth faces if present
        if self.has_depth:
            if not self.all_depth_faces_set():
                missing = [name for name, face in self.depth_faces.items() if face is None]
                return False, f"Missing depth faces: {', '.join(missing)}"

            for name, depth in self.depth_faces.items():
                if depth.shape[0] != self.resolution or depth.shape[1] != self.resolution:
                    return False, f"Depth face {name} has incorrect resolution: {depth.shape[:2]}"

        return True, "Cubemap is valid"

    def copy(self) -> 'CubemapData':
        """Create a deep copy of this cubemap."""
        new_cubemap = CubemapData(resolution=self.resolution)

        # Copy RGB faces
        for name, face in self.faces.items():
            if face is not None:
                new_cubemap.faces[name] = face.copy()

        # Copy depth faces
        for name, depth in self.depth_faces.items():
            if depth is not None:
                new_cubemap.depth_faces[name] = depth.copy()

        new_cubemap.has_depth = self.has_depth

        return new_cubemap

    def to_dict(self) -> Dict:
        """
        Serialize cubemap to dictionary (for debugging/logging).

        Returns:
            Dictionary with cubemap metadata (not including image data)
        """
        return {
            'resolution': self.resolution,
            'has_depth': self.has_depth,
            'faces_set': [name for name, face in self.faces.items() if face is not None],
            'depth_faces_set': [name for name, face in self.depth_faces.items() if face is not None]
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        valid, msg = self.validate()
        status = "valid" if valid else "invalid"
        return (
            f"CubemapData(resolution={self.resolution}, "
            f"has_depth={self.has_depth}, "
            f"status={status})"
        )


def create_empty_cubemap(resolution: int) -> CubemapData:
    """
    Create an empty cubemap with specified resolution.

    Args:
        resolution: Resolution of each square face

    Returns:
        Empty CubemapData object
    """
    return CubemapData(resolution=resolution)


def create_cubemap_from_faces(
    faces: Dict[str, np.ndarray],
    depth_faces: Optional[Dict[str, np.ndarray]] = None
) -> CubemapData:
    """
    Create a cubemap from existing face images.

    Args:
        faces: Dictionary mapping face names to RGB images
        depth_faces: Optional dictionary mapping face names to depth maps

    Returns:
        CubemapData object with faces populated

    Raises:
        ValueError: If faces have inconsistent resolutions
    """
    # Determine resolution from first face
    first_face = next(iter(faces.values()))
    resolution = first_face.shape[0]

    # Verify all faces have same resolution
    for name, face in faces.items():
        if face.shape[0] != resolution or face.shape[1] != resolution:
            raise ValueError(f"Face {name} has inconsistent resolution")

    # Create cubemap
    cubemap = CubemapData(resolution=resolution)
    cubemap.faces = faces.copy()

    # Add depth if provided
    if depth_faces is not None:
        for name, depth in depth_faces.items():
            if depth.shape[0] != resolution or depth.shape[1] != resolution:
                raise ValueError(f"Depth face {name} has inconsistent resolution")
        cubemap.depth_faces = depth_faces.copy()
        cubemap.has_depth = True

    return cubemap
