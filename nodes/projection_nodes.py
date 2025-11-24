"""
ComfyUI-DreamCube - Projection Nodes

Nodes for converting between equirectangular and cubemap projections.

Author: Cedar
License: Apache 2.0
"""

import torch
import numpy as np

from ..core.projection import (
    equirect_to_cubemap_fast,
    cubemap_to_equirect_fast
)
from ..core.cubemap import CubemapData


class EquirectToCubemap:
    """
    Convert equirectangular (360° panoramic) image to cubemap.

    This node takes a 2:1 aspect ratio equirectangular image and
    converts it to a 6-face cubemap representation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI IMAGE format [B, H, W, C]
                "cube_resolution": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 256,
                    "display": "number",
                    "tooltip": "Resolution (pixels) for each cube face; higher = sharper faces but more VRAM"
                }),
            }
        }

    RETURN_TYPES = ("CUBEMAP",)
    FUNCTION = "convert"
    CATEGORY = "DreamCube/Projection"

    def convert(self, image, cube_resolution):
        """
        Convert equirectangular to cubemap.

        Args:
            image: ComfyUI IMAGE tensor [B, H, W, C]
            cube_resolution: Resolution for each cubemap face

        Returns:
            CubemapData object
        """
        # Convert from ComfyUI tensor format to numpy
        if isinstance(image, torch.Tensor):
            equirect_np = image.cpu().numpy()[0]  # Take first batch
        else:
            equirect_np = image[0]

        # Convert to cubemap
        cubemap = equirect_to_cubemap_fast(equirect_np, cube_resolution)

        return (cubemap,)


class CubemapToEquirect:
    """
    Convert cubemap back to equirectangular (360° panoramic) image.

    This node takes a cubemap and reconstructs an equirectangular image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap": ("CUBEMAP",),
                "output_width": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 8192,
                    "step": 256,
                    "tooltip": "Output panorama width (typically 2x height)"
                }),
                "output_height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 256,
                    "tooltip": "Output panorama height"
                }),
                "output_type": (["rgb", "depth", "rgbd"], {
                    "tooltip": "Choose to render RGB, depth-only, or combined RGBD"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "DreamCube/Projection"

    def convert(self, cubemap, output_width, output_height, output_type):
        """
        Convert cubemap to equirectangular.

        Args:
            cubemap: CubemapData object
            output_width: Width of output equirect image
            output_height: Height of output equirect image
            output_type: 'rgb', 'depth', or 'rgbd'

        Returns:
            ComfyUI IMAGE tensor [1, H, W, C]
        """
        if output_type == "rgb":
            equirect = cubemap_to_equirect_fast(cubemap, output_width, output_height, use_depth=False)
        elif output_type == "depth":
            equirect = cubemap_to_equirect_fast(cubemap, output_width, output_height, use_depth=True)
            # Convert single channel depth to RGB for visualization
            if equirect.shape[2] == 1:
                equirect = np.repeat(equirect, 3, axis=2)
        elif output_type == "rgbd":
            # Combine RGB and depth
            rgb_equirect = cubemap_to_equirect_fast(cubemap, output_width, output_height, use_depth=False)
            depth_equirect = cubemap_to_equirect_fast(cubemap, output_width, output_height, use_depth=True)
            equirect = np.concatenate([rgb_equirect, depth_equirect], axis=-1)

        # Convert to ComfyUI tensor format [B, H, W, C]
        tensor = torch.from_numpy(equirect).unsqueeze(0).float()

        return (tensor,)


class ExtractCubemapFace:
    """
    Extract a single face from a cubemap as a standard image.

    Useful for processing individual faces through other ComfyUI nodes
    (e.g., depth estimation).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap": ("CUBEMAP",),
                "face": (["front", "back", "left", "right", "top", "bottom"], {
                    "tooltip": "Which cube face to extract"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract"
    CATEGORY = "DreamCube/Projection"

    def extract(self, cubemap, face):
        """
        Extract a single cubemap face.

        Args:
            cubemap: CubemapData object
            face: Face name to extract

        Returns:
            ComfyUI IMAGE tensor [1, H, W, C]
        """
        face_img = cubemap.get_face(face)

        if face_img is None:
            raise ValueError(f"Face {face} is not set in cubemap")

        # Convert to tensor [B, H, W, C]
        tensor = torch.from_numpy(face_img).unsqueeze(0).float()

        return (tensor,)


class InsertCubemapFace:
    """
    Insert/update a single face in a cubemap.

    Useful for putting processed faces (e.g., with depth) back into the cubemap.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap": ("CUBEMAP",),
                "image": ("IMAGE",),
                "face": (["front", "back", "left", "right", "top", "bottom"], {
                    "tooltip": "Which face to overwrite/insert"
                }),
            }
        }

    RETURN_TYPES = ("CUBEMAP",)
    FUNCTION = "insert"
    CATEGORY = "DreamCube/Projection"

    def insert(self, cubemap, image, face):
        """
        Insert image into cubemap face.

        Args:
            cubemap: CubemapData object
            image: ComfyUI IMAGE tensor [B, H, W, C]
            face: Face name to update

        Returns:
            Updated cubemap
        """
        # Convert from ComfyUI tensor to numpy
        if isinstance(image, torch.Tensor):
            face_np = image.cpu().numpy()[0]
        else:
            face_np = image[0]

        # Create a copy to avoid modifying original
        new_cubemap = cubemap.copy()

        # Set the face
        new_cubemap.set_face(face, face_np)

        return (new_cubemap,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "EquirectToCubemap": EquirectToCubemap,
    "CubemapToEquirect": CubemapToEquirect,
    "ExtractCubemapFace": ExtractCubemapFace,
    "InsertCubemapFace": InsertCubemapFace,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectToCubemap": "Equirect to Cubemap",
    "CubemapToEquirect": "Cubemap to Equirect",
    "ExtractCubemapFace": "Extract Cubemap Face",
    "InsertCubemapFace": "Insert Cubemap Face",
}
