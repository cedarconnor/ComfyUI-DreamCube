"""
ComfyUI-DreamCube Core Package

Core algorithms for cubemap projection and depth processing.
"""

from .cubemap import CubemapData, create_empty_cubemap, create_cubemap_from_faces
from .projection import (
    equirect_to_cubemap,
    equirect_to_cubemap_fast,
    cubemap_to_equirect,
    cubemap_to_equirect_fast,
    face_coords_to_vector,
    vector_to_lonlat,
    lonlat_to_vector,
    vector_to_face_coords
)
from .depth_interface import DepthModelInterface
from .consistency import DepthConsistencyEnforcer
from .synchronization import (
    SyncedSelfAttention,
    SyncedConv2d,
    SyncedGroupNorm,
    MultiplaneSyncProcessor
)

__all__ = [
    "CubemapData",
    "create_empty_cubemap",
    "create_cubemap_from_faces",
    "equirect_to_cubemap",
    "equirect_to_cubemap_fast",
    "cubemap_to_equirect",
    "cubemap_to_equirect_fast",
    "face_coords_to_vector",
    "vector_to_lonlat",
    "lonlat_to_vector",
    "vector_to_face_coords",
    "DepthModelInterface",
    "DepthConsistencyEnforcer",
    "SyncedSelfAttention",
    "SyncedConv2d",
    "SyncedGroupNorm",
    "MultiplaneSyncProcessor"
]
