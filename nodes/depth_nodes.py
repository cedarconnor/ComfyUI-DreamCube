"""
ComfyUI-DreamCube - Depth Processing Nodes

Nodes for applying depth estimation to cubemap faces.

Author: Cedar
License: Apache 2.0
"""

import torch
import numpy as np

from ..core.depth_interface import DepthModelInterface
from ..core.consistency import DepthConsistencyEnforcer
from ..core.cubemap import CubemapData


class ApplyDepthToCubemapFace:
    """
    Apply depth map to a single cubemap face.

    Use this node to assign depth from any ComfyUI depth node to
    a specific cubemap face.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap": ("CUBEMAP",),
                "depth_map": ("IMAGE",),  # Depth from any depth node
                "face": (["front", "back", "left", "right", "top", "bottom"], {
                    "tooltip": "Target face for this depth map"
                }),
            }
        }

    RETURN_TYPES = ("CUBEMAP",)
    FUNCTION = "apply_depth"
    CATEGORY = "DreamCube/Depth"

    def apply_depth(self, cubemap, depth_map, face):
        """
        Apply depth to a single face.

        Args:
            cubemap: CubemapData object
            depth_map: Depth map from ComfyUI depth node
            face: Face name to apply depth to

        Returns:
            Updated cubemap with depth
        """
        # Convert depth to numpy
        if isinstance(depth_map, torch.Tensor):
            depth_np = depth_map.cpu().numpy()[0]
        else:
            depth_np = depth_map[0]

        # Create depth interface
        depth_interface = DepthModelInterface(normalization='per_face')

        # Create a copy
        new_cubemap = cubemap.copy()

        # Apply depth to face
        new_cubemap = depth_interface.apply_depth_to_face(new_cubemap, face, depth_np)

        return (new_cubemap,)


class BatchCubemapDepth:
    """
    Batch process all 6 cubemap faces with depth maps.

    This node takes depth maps for all 6 faces (from external depth nodes)
    and applies them to the cubemap with optional consistency enforcement.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap_rgb": ("CUBEMAP",),
                "depth_front": ("IMAGE",),
                "depth_back": ("IMAGE",),
                "depth_left": ("IMAGE",),
                "depth_right": ("IMAGE",),
                "depth_top": ("IMAGE",),
                "depth_bottom": ("IMAGE",),
                "enforce_consistency": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Blend seams after applying depths to reduce edge artifacts"
                }),
                "normalization": (["global", "per_face", "adaptive"], {
                    "default": "global",
                    "tooltip": "Depth scaling: global = shared min/max; per_face = independent; adaptive = raw"
                }),
            }
        }

    RETURN_TYPES = ("CUBEMAP",)
    FUNCTION = "apply_depth"
    CATEGORY = "DreamCube/Depth"

    def apply_depth(
        self,
        cubemap_rgb,
        depth_front,
        depth_back,
        depth_left,
        depth_right,
        depth_top,
        depth_bottom,
        enforce_consistency,
        normalization
    ):
        """
        Apply all depth maps to cubemap.

        Args:
            cubemap_rgb: CubemapData with RGB faces
            depth_*: Depth maps for each face
            enforce_consistency: Whether to blend boundaries
            normalization: Normalization method

        Returns:
            CubemapData with depth
        """
        # Collect depth maps
        depth_maps = {
            'front': depth_front,
            'back': depth_back,
            'left': depth_left,
            'right': depth_right,
            'top': depth_top,
            'bottom': depth_bottom
        }

        # Convert to numpy
        depth_maps_np = {}
        for face_name, depth_tensor in depth_maps.items():
            if isinstance(depth_tensor, torch.Tensor):
                depth_maps_np[face_name] = depth_tensor.cpu().numpy()[0]
            else:
                depth_maps_np[face_name] = depth_tensor[0]

        # Create depth interface
        depth_interface = DepthModelInterface(normalization=normalization)

        # Copy cubemap
        new_cubemap = cubemap_rgb.copy()

        # Apply all depths
        new_cubemap = depth_interface.apply_depth_to_all_faces(new_cubemap, depth_maps_np)

        # Enforce consistency if requested
        if enforce_consistency:
            enforcer = DepthConsistencyEnforcer(boundary_width=16)
            new_cubemap = enforcer.enforce_consistency(new_cubemap, iterations=2)

        return (new_cubemap,)


class MergeCubemapDepth:
    """
    Merge RGB and depth cubemaps into a single RGBD cubemap.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap_rgb": ("CUBEMAP",),
                "cubemap_depth": ("CUBEMAP",),
            }
        }

    RETURN_TYPES = ("CUBEMAP",)
    FUNCTION = "merge"
    CATEGORY = "DreamCube/Depth"

    def merge(self, cubemap_rgb, cubemap_depth):
        """
        Merge RGB and depth cubemaps.

        Args:
            cubemap_rgb: Cubemap with RGB data
            cubemap_depth: Cubemap with depth data

        Returns:
            Merged cubemap
        """
        depth_interface = DepthModelInterface()
        merged = depth_interface.merge_rgb_depth(cubemap_rgb, cubemap_depth)

        return (merged,)


class ExtractDepthChannel:
    """
    Extract depth channel from cubemap as a separate cubemap for visualization.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap": ("CUBEMAP",),
            }
        }

    RETURN_TYPES = ("CUBEMAP",)
    FUNCTION = "extract"
    CATEGORY = "DreamCube/Depth"

    def extract(self, cubemap):
        """
        Extract depth as separate cubemap.

        Args:
            cubemap: Cubemap with depth information

        Returns:
            New cubemap with depth as RGB
        """
        depth_interface = DepthModelInterface()
        depth_cubemap = depth_interface.extract_depth_channel(cubemap)

        return (depth_cubemap,)


class NormalizeCubemapDepth:
    """
    Normalize depth values across all cubemap faces.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap": ("CUBEMAP",),
                "method": (["global", "per_face", "align_scales"], {
                    "default": "global",
                    "tooltip": "global = shared min/max; per_face = independent scaling; align_scales = match reference face"
                }),
            }
        }

    RETURN_TYPES = ("CUBEMAP",)
    FUNCTION = "normalize"
    CATEGORY = "DreamCube/Depth"

    def normalize(self, cubemap, method):
        """
        Normalize depth values.

        Args:
            cubemap: Cubemap with depth
            method: Normalization method

        Returns:
            Cubemap with normalized depth
        """
        if not cubemap.has_depth:
            raise ValueError("Cubemap has no depth information")

        new_cubemap = cubemap.copy()

        if method == "align_scales":
            # Align to front face
            from core.consistency import DepthConsistencyEnforcer
            enforcer = DepthConsistencyEnforcer()
            new_cubemap = enforcer.align_depth_scales(new_cubemap, reference_face='front')
        else:
            # Re-apply normalization
            depth_maps = {name: new_cubemap.get_depth_face(name) for name in new_cubemap.get_face_names()}
            interface = DepthModelInterface(normalization=method)
            new_cubemap = interface.apply_depth_to_all_faces(new_cubemap, depth_maps)

        return (new_cubemap,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ApplyDepthToCubemapFace": ApplyDepthToCubemapFace,
    "BatchCubemapDepth": BatchCubemapDepth,
    "MergeCubemapDepth": MergeCubemapDepth,
    "ExtractDepthChannel": ExtractDepthChannel,
    "NormalizeCubemapDepth": NormalizeCubemapDepth,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyDepthToCubemapFace": "Apply Depth to Cubemap Face",
    "BatchCubemapDepth": "Batch Cubemap Depth",
    "MergeCubemapDepth": "Merge Cubemap Depth",
    "ExtractDepthChannel": "Extract Depth Channel",
    "NormalizeCubemapDepth": "Normalize Cubemap Depth",
}
