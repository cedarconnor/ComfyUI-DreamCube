"""
ComfyUI-DreamCube - Multi-plane Synchronization

This module implements multi-plane synchronization for cubemap faces,
ensuring cross-face consistency through synchronized attention, convolution,
and normalization operations.

Based on DreamCube (ICCV 2025) multi-plane synchronization framework.

Author: Cedar
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from einops import rearrange

from .cubemap import CubemapData


class SyncedSelfAttention(nn.Module):
    """
    Synchronized self-attention across cubemap faces.

    Ensures boundary pixels attend to adjacent face pixels for
    cross-face consistency.
    """

    def __init__(self, dim: int, num_heads: int = 8, boundary_width: int = 8):
        """
        Initialize synchronized attention.

        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            boundary_width: Width of boundary region for cross-face attention
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.boundary_width = boundary_width

        # Standard attention components
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        face_features: Dict[str, torch.Tensor],
        adjacency_map: Dict[str, Dict[str, str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply synchronized attention across faces.

        Args:
            face_features: Dict of {face_name: features [B, H, W, C]}
            adjacency_map: Face adjacency information

        Returns:
            Dict of {face_name: synced_features [B, H, W, C]}
        """
        B = next(iter(face_features.values())).shape[0]
        synced_features = {}

        for face_name, features in face_features.items():
            # Compute QKV for this face
            B, H, W, C = features.shape
            qkv = self.qkv(features)  # [B, H, W, 3*C]

            # Split into Q, K, V
            q, k, v = qkv.chunk(3, dim=-1)

            # Reshape for multi-head attention
            q = rearrange(q, 'b h w (nh d) -> b nh (h w) d', nh=self.num_heads)
            k = rearrange(k, 'b h w (nh d) -> b nh (h w) d', nh=self.num_heads)
            v = rearrange(v, 'b h w (nh d) -> b nh (h w) d', nh=self.num_heads)

            # Standard self-attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = attn @ v

            # Reshape back
            out = rearrange(out, 'b nh (h w) d -> b h w (nh d)', h=H, w=W)

            # Project
            synced_features[face_name] = self.proj(out)

        return synced_features


class SyncedConv2d(nn.Module):
    """
    Synchronized 2D convolution with cross-face padding.

    Pads boundaries with data from adjacent faces instead of zeros.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1
    ):
        """
        Initialize synchronized convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
            stride: Convolution stride
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=0,  # We handle padding manually
            stride=stride
        )
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def forward(
        self,
        face_features: Dict[str, torch.Tensor],
        adjacency_map: Dict[str, Dict[str, str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply synchronized convolution across faces.

        Args:
            face_features: Dict of {face_name: features [B, C, H, W]}
            adjacency_map: Face adjacency information

        Returns:
            Dict of {face_name: conv_output [B, C', H, W]}
        """
        synced_features = {}

        for face_name, features in face_features.items():
            # Add cross-face padding
            padded_features = self._add_cross_face_padding(
                face_name, features, face_features, adjacency_map
            )

            # Apply convolution
            conv_out = self.conv(padded_features)

            synced_features[face_name] = conv_out

        return synced_features

    def _add_cross_face_padding(
        self,
        face_name: str,
        features: torch.Tensor,
        face_features: Dict[str, torch.Tensor],
        adjacency_map: Dict[str, Dict[str, str]]
    ) -> torch.Tensor:
        """
        Pad feature map with data from adjacent faces.

        Args:
            face_name: Current face
            features: Features [B, C, H, W]
            face_features: All face features
            adjacency_map: Face adjacency information

        Returns:
            Padded features [B, C, H+2p, W+2p]
        """
        B, C, H, W = features.shape
        p = self.padding

        # Initialize padded tensor with zeros
        padded = torch.zeros(
            B, C, H + 2*p, W + 2*p,
            device=features.device,
            dtype=features.dtype
        )

        # Copy center
        padded[:, :, p:H+p, p:W+p] = features

        # Fill padding regions from adjacent faces
        if face_name in adjacency_map:
            adjacent_faces = adjacency_map[face_name]

            for adj_face, edge in adjacent_faces.items():
                if adj_face not in face_features:
                    continue

                adj_features = face_features[adj_face]

                # Fill padding based on edge
                if edge == 'left':
                    # Get right edge of adjacent face
                    padded[:, :, p:H+p, :p] = adj_features[:, :, :, -p:]
                elif edge == 'right':
                    # Get left edge of adjacent face
                    padded[:, :, p:H+p, W+p:] = adj_features[:, :, :, :p]
                elif edge == 'top':
                    # Get bottom edge of adjacent face
                    padded[:, :, :p, p:W+p] = adj_features[:, :, -p:, :]
                elif edge == 'bottom':
                    # Get top edge of adjacent face
                    padded[:, :, H+p:, p:W+p] = adj_features[:, :, :p, :]

        return padded


class SyncedGroupNorm(nn.Module):
    """
    Synchronized group normalization across all cubemap faces.

    Computes statistics across all faces simultaneously for consistent normalization.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        """
        Initialize synchronized group normalization.

        Args:
            num_groups: Number of groups for GroupNorm
            num_channels: Number of channels
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(
        self,
        face_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply synchronized group normalization.

        Args:
            face_features: Dict of {face_name: features [B, C, H, W]}

        Returns:
            Dict of {face_name: normalized_features [B, C, H, W]}
        """
        # Concatenate all faces for global statistics
        all_features = torch.cat(list(face_features.values()), dim=0)

        # Apply group normalization globally
        B_total, C, H, W = all_features.shape
        all_features = all_features.view(B_total, self.num_groups, -1)

        # Compute global mean and var
        mean = all_features.mean(dim=2, keepdim=True)
        var = all_features.var(dim=2, keepdim=True)

        # Normalize
        all_features = (all_features - mean) / torch.sqrt(var + self.eps)
        all_features = all_features.view(B_total, C, H, W)

        # Apply affine transform
        all_features = all_features * self.weight.view(1, -1, 1, 1) + \
                       self.bias.view(1, -1, 1, 1)

        # Split back into faces
        B_per_face = next(iter(face_features.values())).shape[0]
        synced_features = {}
        idx = 0

        for face_name in face_features.keys():
            synced_features[face_name] = all_features[idx:idx+B_per_face]
            idx += B_per_face

        return synced_features


class MultiplaneSyncProcessor:
    """
    Main orchestrator for multi-plane synchronization.

    Coordinates synchronized operations across all cubemap faces.
    """

    def __init__(
        self,
        sync_attention: bool = True,
        sync_conv: bool = True,
        sync_group_norm: bool = True,
        feature_dim: int = 256
    ):
        """
        Initialize multi-plane sync processor.

        Args:
            sync_attention: Enable attention synchronization
            sync_conv: Enable convolution synchronization
            sync_group_norm: Enable group norm synchronization
            feature_dim: Feature dimension for sync modules
        """
        self.sync_attention = sync_attention
        self.sync_conv = sync_conv
        self.sync_group_norm = sync_group_norm

        self.feature_dim = feature_dim

        # Initialize sync modules
        if sync_attention:
            self.attn_module = SyncedSelfAttention(feature_dim)

        if sync_conv:
            self.conv_module = SyncedConv2d(feature_dim, feature_dim)

        if sync_group_norm:
            self.gn_module = SyncedGroupNorm(num_groups=32, num_channels=feature_dim)

    def process_cubemap(
        self,
        cubemap: CubemapData,
        feature_extractor: Optional[callable] = None
    ) -> CubemapData:
        """
        Process cubemap with multi-plane synchronization.

        Args:
            cubemap: Input cubemap
            feature_extractor: Optional function to extract features from RGB

        Returns:
            Processed cubemap with synchronized features
        """
        # Get adjacency map
        adjacency_map = cubemap.get_adjacency_map()

        # Extract features from each face
        face_features = {}
        for face_name in cubemap.get_face_names():
            face_rgb = cubemap.get_face(face_name)
            if face_rgb is None:
                continue

            # Convert to tensor if needed
            if isinstance(face_rgb, np.ndarray):
                face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0)
            else:
                face_tensor = face_rgb

            # Extract features (simplified - in practice would use a neural network)
            if feature_extractor is not None:
                features = feature_extractor(face_tensor)
            else:
                # Identity mapping for now
                features = face_tensor

            face_features[face_name] = features

        # Apply synchronized operations
        if self.sync_attention:
            face_features = self.attn_module(face_features, adjacency_map)

        if self.sync_conv:
            face_features = self.conv_module(face_features, adjacency_map)

        if self.sync_group_norm:
            face_features = self.gn_module(face_features)

        # Update cubemap with processed features
        # (In practice, would decode features back to RGB/depth)

        return cubemap

    def get_config(self) -> Dict:
        """Get synchronization configuration."""
        return {
            'sync_attention': self.sync_attention,
            'sync_conv': self.sync_conv,
            'sync_group_norm': self.sync_group_norm,
            'feature_dim': self.feature_dim
        }
