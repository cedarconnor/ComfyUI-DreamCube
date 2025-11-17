"""
ComfyUI-DreamCube - 360° Panoramic Depth Estimation with Multi-plane Synchronization

A ComfyUI custom node pack implementing DreamCube's multi-plane synchronization
framework for consistent depth estimation on 360° panoramic images.

Author: Cedar
License: Apache 2.0
Based on: DreamCube (ICCV 2025) by Yukun Huang et al.
"""

from .nodes import projection_nodes, depth_nodes, utility_nodes

__version__ = "1.0.0"
__author__ = "Cedar"
__license__ = "Apache 2.0"

# Collect all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register projection nodes
NODE_CLASS_MAPPINGS.update(projection_nodes.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(projection_nodes.NODE_DISPLAY_NAME_MAPPINGS)

# Register depth nodes
NODE_CLASS_MAPPINGS.update(depth_nodes.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(depth_nodes.NODE_DISPLAY_NAME_MAPPINGS)

# Register utility nodes
NODE_CLASS_MAPPINGS.update(utility_nodes.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(utility_nodes.NODE_DISPLAY_NAME_MAPPINGS)

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"✅ ComfyUI-DreamCube v{__version__} loaded successfully!")
print(f"   {len(NODE_CLASS_MAPPINGS)} nodes registered")
