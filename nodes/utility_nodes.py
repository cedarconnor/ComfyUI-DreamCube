"""
ComfyUI-DreamCube - Utility Nodes

Utility nodes for visualization, validation, and helper functions.

Author: Cedar
License: Apache 2.0
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.visualization import create_cubemap_preview
from utils.validation import calculate_seam_quality, validate_cubemap_integrity
from core.consistency import DepthConsistencyEnforcer


class CubemapPreview:
    """
    Create a preview visualization of the cubemap.

    Shows all 6 faces in various layouts for inspection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap": ("CUBEMAP",),
                "layout": (["horizontal", "cross", "vertical", "grid"], {"default": "cross"}),
                "show_depth": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview"
    CATEGORY = "DreamCube/Utility"

    def preview(self, cubemap, layout, show_depth):
        """
        Create preview visualization.

        Args:
            cubemap: CubemapData object
            layout: Layout style
            show_depth: Show depth instead of RGB

        Returns:
            Preview image as tensor
        """
        # Create preview
        preview_np = create_cubemap_preview(
            cubemap,
            layout=layout,
            use_depth=show_depth
        )

        # Convert to tensor
        preview_tensor = torch.from_numpy(preview_np).unsqueeze(0).float()

        return (preview_tensor,)


class CubemapSeamValidator:
    """
    Validate seam quality and depth consistency across cubemap faces.

    Outputs validation metrics and a boolean indicating if seams are acceptable.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap": ("CUBEMAP",),
                "threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "STRING")
    RETURN_NAMES = ("is_valid", "max_error", "report")
    FUNCTION = "validate"
    CATEGORY = "DreamCube/Utility"

    def validate(self, cubemap, threshold):
        """
        Validate seam quality.

        Args:
            cubemap: CubemapData with depth
            threshold: Maximum acceptable error

        Returns:
            Tuple of (is_valid, max_error, report_string)
        """
        if not cubemap.has_depth:
            return (False, 1.0, "ERROR: Cubemap has no depth information")

        # Calculate seam quality
        metrics = calculate_seam_quality(cubemap, threshold)

        is_valid = metrics['is_valid']
        max_error = metrics['max_error']

        # Create report string
        report = f"""Seam Validation Report:
Max Error: {max_error:.4f}
Mean Error: {metrics['mean_error']:.4f}
Median Error: {metrics['median_error']:.4f}
Threshold: {threshold:.4f}
Status: {'PASS' if is_valid else 'FAIL'}
Number of Seams: {metrics['num_seams']}
"""

        return (is_valid, max_error, report)


class CubemapInfo:
    """
    Display information about a cubemap.

    Useful for debugging and understanding cubemap properties.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap": ("CUBEMAP",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_info"
    CATEGORY = "DreamCube/Utility"

    def get_info(self, cubemap):
        """
        Get cubemap information.

        Args:
            cubemap: CubemapData object

        Returns:
            Information string
        """
        # Validate cubemap
        integrity = validate_cubemap_integrity(cubemap)

        info = f"""Cubemap Information:
Resolution: {cubemap.resolution}x{cubemap.resolution} per face
Has Depth: {cubemap.has_depth}
All Faces Set: {cubemap.all_faces_set()}
All Depth Faces Set: {cubemap.all_depth_faces_set()}
Valid: {integrity['is_valid']}
"""

        if 'mean_brightness' in integrity:
            info += f"""Mean Brightness: {integrity['mean_brightness']:.3f}
Brightness Uniformity: {integrity['brightness_uniformity']:.3f}
"""

        return (info,)


class EnforceDepthConsistency:
    """
    Manually enforce depth consistency across cubemap boundaries.

    Applies boundary blending to reduce seam artifacts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap": ("CUBEMAP",),
                "boundary_width": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "step": 4
                }),
                "iterations": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10
                }),
            }
        }

    RETURN_TYPES = ("CUBEMAP",)
    FUNCTION = "enforce"
    CATEGORY = "DreamCube/Utility"

    def enforce(self, cubemap, boundary_width, iterations):
        """
        Enforce depth consistency.

        Args:
            cubemap: CubemapData with depth
            boundary_width: Width of blending region
            iterations: Number of smoothing iterations

        Returns:
            Cubemap with smoothed boundaries
        """
        if not cubemap.has_depth:
            raise ValueError("Cubemap has no depth information")

        # Create enforcer
        enforcer = DepthConsistencyEnforcer(boundary_width=boundary_width)

        # Copy and enforce
        new_cubemap = cubemap.copy()
        new_cubemap = enforcer.enforce_consistency(new_cubemap, iterations=iterations)

        return (new_cubemap,)


class SmoothCubemapDepth:
    """
    Apply global smoothing to cubemap depth.

    Useful for reducing noise while preserving structure.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap": ("CUBEMAP",),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("CUBEMAP",)
    FUNCTION = "smooth"
    CATEGORY = "DreamCube/Utility"

    def smooth(self, cubemap, sigma):
        """
        Apply Gaussian smoothing.

        Args:
            cubemap: CubemapData with depth
            sigma: Gaussian kernel sigma

        Returns:
            Smoothed cubemap
        """
        if not cubemap.has_depth:
            raise ValueError("Cubemap has no depth information")

        # Create enforcer and smooth
        enforcer = DepthConsistencyEnforcer()
        new_cubemap = cubemap.copy()
        new_cubemap = enforcer.smooth_globally(new_cubemap, sigma=sigma)

        return (new_cubemap,)


class CreateEmptyCubemap:
    """
    Create an empty cubemap with specified resolution.

    Useful for advanced workflows where you want to manually populate faces.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 256
                }),
            }
        }

    RETURN_TYPES = ("CUBEMAP",)
    FUNCTION = "create"
    CATEGORY = "DreamCube/Utility"

    def create(self, resolution):
        """
        Create empty cubemap.

        Args:
            resolution: Face resolution

        Returns:
            Empty cubemap
        """
        from core.cubemap import create_empty_cubemap
        cubemap = create_empty_cubemap(resolution)

        # Initialize faces with black images
        black_face = np.zeros((resolution, resolution, 3), dtype=np.float32)
        for face_name in cubemap.get_face_names():
            cubemap.set_face(face_name, black_face.copy())

        return (cubemap,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "CubemapPreview": CubemapPreview,
    "CubemapSeamValidator": CubemapSeamValidator,
    "CubemapInfo": CubemapInfo,
    "EnforceDepthConsistency": EnforceDepthConsistency,
    "SmoothCubemapDepth": SmoothCubemapDepth,
    "CreateEmptyCubemap": CreateEmptyCubemap,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "CubemapPreview": "Cubemap Preview",
    "CubemapSeamValidator": "Validate Cubemap Seams",
    "CubemapInfo": "Cubemap Info",
    "EnforceDepthConsistency": "Enforce Depth Consistency",
    "SmoothCubemapDepth": "Smooth Cubemap Depth",
    "CreateEmptyCubemap": "Create Empty Cubemap",
}
