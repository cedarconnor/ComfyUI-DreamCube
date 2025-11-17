"""
ComfyUI-DreamCube - Projection Mathematics

This module implements the core projection transformations between
equirectangular and cubemap representations for 360° panoramic images.

Coordinate Systems:
- Equirectangular: θ ∈ [0, 2π], φ ∈ [-π/2, π/2]
- Cubemap: 6 faces with UV ∈ [-1, 1] × [-1, 1] per face
- 3D Space: Right-handed, Z-forward

Author: Cedar
License: Apache 2.0
"""

import numpy as np
import torch
from typing import Tuple, Optional
from scipy.ndimage import map_coordinates

from .cubemap import CubemapData, create_empty_cubemap


def face_coords_to_vector(face: str, x: int, y: int, size: int) -> np.ndarray:
    """
    Convert 2D face pixel coordinates to 3D unit vector.

    Coordinate system:
    - X: right
    - Y: up
    - Z: forward (front face normal)

    Args:
        face: Face name ('front', 'back', 'left', 'right', 'top', 'bottom')
        x: Pixel x-coordinate [0, size)
        y: Pixel y-coordinate [0, size)
        size: Face resolution

    Returns:
        3D unit vector as numpy array [x, y, z]
    """
    # Normalize to [-1, 1]
    u = (2.0 * x / size) - 1.0
    v = 1.0 - (2.0 * y / size)  # Flip Y (image coords are top-down)

    # Map to 3D vectors based on face
    vectors = {
        'front':  np.array([u, v, 1.0]),
        'back':   np.array([-u, v, -1.0]),
        'left':   np.array([-1.0, v, u]),
        'right':  np.array([1.0, v, -u]),
        'top':    np.array([u, 1.0, -v]),
        'bottom': np.array([u, -1.0, v])
    }

    vec = vectors[face]
    # Normalize to unit vector
    return vec / np.linalg.norm(vec)


def vector_to_lonlat(vec: np.ndarray) -> Tuple[float, float]:
    """
    Convert 3D unit vector to longitude/latitude (spherical coordinates).

    Args:
        vec: 3D unit vector [x, y, z]

    Returns:
        Tuple of (longitude, latitude) in radians
        - longitude: [-π, π]
        - latitude: [-π/2, π/2]
    """
    lon = np.arctan2(vec[0], vec[2])
    lat = np.arcsin(np.clip(vec[1], -1.0, 1.0))
    return lon, lat


def lonlat_to_vector(lon: float, lat: float) -> np.ndarray:
    """
    Convert longitude/latitude to 3D unit vector.

    Args:
        lon: Longitude in radians [-π, π]
        lat: Latitude in radians [-π/2, π/2]

    Returns:
        3D unit vector [x, y, z]
    """
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    return np.array([x, y, z])


def vector_to_face_coords(vec: np.ndarray, face_size: int) -> Tuple[str, int, int]:
    """
    Determine which cubemap face a 3D vector points to and its pixel coordinates.

    Args:
        vec: 3D unit vector [x, y, z]
        face_size: Resolution of each cube face

    Returns:
        Tuple of (face_name, x, y) where x, y are pixel coordinates [0, face_size)
    """
    abs_vec = np.abs(vec)
    max_axis = np.argmax(abs_vec)

    # Determine face and local UV coordinates
    if max_axis == 0:  # X dominant
        if vec[0] > 0:
            face = 'right'
            u = -vec[2] / abs_vec[0]
            v = vec[1] / abs_vec[0]
        else:
            face = 'left'
            u = vec[2] / abs_vec[0]
            v = vec[1] / abs_vec[0]

    elif max_axis == 1:  # Y dominant
        if vec[1] > 0:
            face = 'top'
            u = vec[0] / abs_vec[1]
            v = -vec[2] / abs_vec[1]
        else:
            face = 'bottom'
            u = vec[0] / abs_vec[1]
            v = vec[2] / abs_vec[1]

    else:  # Z dominant
        if vec[2] > 0:
            face = 'front'
            u = vec[0] / abs_vec[2]
            v = vec[1] / abs_vec[2]
        else:
            face = 'back'
            u = -vec[0] / abs_vec[2]
            v = vec[1] / abs_vec[2]

    # Convert from [-1, 1] to pixel coordinates [0, face_size)
    x = int((u + 1.0) * face_size / 2.0)
    y = int((1.0 - v) * face_size / 2.0)

    # Clamp to valid range
    x = np.clip(x, 0, face_size - 1)
    y = np.clip(y, 0, face_size - 1)

    return face, x, y


def equirect_to_cubemap(
    equirect_img: np.ndarray,
    face_size: int,
    interpolation: str = 'linear'
) -> CubemapData:
    """
    Convert equirectangular (2:1) image to cubemap.

    Args:
        equirect_img: Input image array [H, W, C] where H = W/2
        face_size: Resolution of each cube face
        interpolation: 'linear' or 'nearest'

    Returns:
        CubemapData object with 6 populated faces

    Raises:
        ValueError: If input image has invalid aspect ratio
    """
    h, w = equirect_img.shape[:2]
    channels = equirect_img.shape[2] if equirect_img.ndim == 3 else 1

    # Validate aspect ratio
    if abs(w / h - 2.0) > 0.1:
        raise ValueError(
            f"Equirectangular image should have 2:1 aspect ratio, "
            f"got {w}x{h} ({w/h:.2f}:1)"
        )

    # Create cubemap
    cubemap = create_empty_cubemap(face_size)

    # For each face
    for face_name in cubemap.get_face_names():
        face_img = np.zeros((face_size, face_size, channels), dtype=equirect_img.dtype)

        # Pre-compute all coordinates for this face
        for y in range(face_size):
            for x in range(face_size):
                # Convert face coordinates to 3D unit vector
                vec = face_coords_to_vector(face_name, x, y, face_size)

                # Convert 3D vector to equirectangular coordinates
                lon, lat = vector_to_lonlat(vec)

                # Map to pixel coordinates in equirectangular image
                eq_x = (lon + np.pi) / (2 * np.pi) * w
                eq_y = (np.pi / 2 - lat) / np.pi * h

                # Sample from equirectangular image
                if interpolation == 'linear':
                    # Bilinear interpolation
                    x0, y0 = int(eq_x), int(eq_y)
                    x1, y1 = x0 + 1, y0 + 1

                    # Handle wrapping at x boundaries
                    x0 = x0 % w
                    x1 = x1 % w
                    y0 = np.clip(y0, 0, h - 1)
                    y1 = np.clip(y1, 0, h - 1)

                    # Interpolation weights
                    wx = eq_x - int(eq_x)
                    wy = eq_y - int(eq_y)

                    # Bilinear interpolation
                    face_img[y, x] = (
                        (1 - wx) * (1 - wy) * equirect_img[y0, x0] +
                        wx * (1 - wy) * equirect_img[y0, x1] +
                        (1 - wx) * wy * equirect_img[y1, x0] +
                        wx * wy * equirect_img[y1, x1]
                    )
                else:
                    # Nearest neighbor
                    eq_x = int(eq_x) % w
                    eq_y = int(eq_y)
                    eq_y = np.clip(eq_y, 0, h - 1)
                    face_img[y, x] = equirect_img[eq_y, eq_x]

        cubemap.set_face(face_name, face_img)

    return cubemap


def equirect_to_cubemap_fast(
    equirect_img: np.ndarray,
    face_size: int
) -> CubemapData:
    """
    Vectorized version of equirect_to_cubemap for better performance.

    Uses numpy array operations instead of nested loops.

    Args:
        equirect_img: Input image array [H, W, C]
        face_size: Resolution of each cube face

    Returns:
        CubemapData object with 6 populated faces
    """
    h, w = equirect_img.shape[:2]
    channels = equirect_img.shape[2] if equirect_img.ndim == 3 else 1

    cubemap = create_empty_cubemap(face_size)

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:face_size, 0:face_size]

    for face_name in cubemap.get_face_names():
        # Vectorized coordinate conversion
        u = (2.0 * x_coords / face_size) - 1.0
        v = 1.0 - (2.0 * y_coords / face_size)

        # Convert UV to 3D vectors (vectorized)
        if face_name == 'front':
            vecs_x, vecs_y, vecs_z = u, v, np.ones_like(u)
        elif face_name == 'back':
            vecs_x, vecs_y, vecs_z = -u, v, -np.ones_like(u)
        elif face_name == 'left':
            vecs_x, vecs_y, vecs_z = -np.ones_like(u), v, u
        elif face_name == 'right':
            vecs_x, vecs_y, vecs_z = np.ones_like(u), v, -u
        elif face_name == 'top':
            vecs_x, vecs_y, vecs_z = u, np.ones_like(u), -v
        elif face_name == 'bottom':
            vecs_x, vecs_y, vecs_z = u, -np.ones_like(u), v

        # Normalize vectors
        norm = np.sqrt(vecs_x**2 + vecs_y**2 + vecs_z**2)
        vecs_x /= norm
        vecs_y /= norm
        vecs_z /= norm

        # Convert to lon/lat
        lon = np.arctan2(vecs_x, vecs_z)
        lat = np.arcsin(np.clip(vecs_y, -1.0, 1.0))

        # Map to equirect coordinates
        eq_x = (lon + np.pi) / (2 * np.pi) * w
        eq_y = (np.pi / 2 - lat) / np.pi * h

        # Sample using map_coordinates (handles interpolation)
        face_img = np.zeros((face_size, face_size, channels), dtype=equirect_img.dtype)

        for c in range(channels):
            # Wrap x coordinates, clamp y coordinates
            eq_x_wrapped = eq_x % w
            eq_y_clamped = np.clip(eq_y, 0, h - 1)

            # Use scipy's map_coordinates for interpolation
            coords = np.array([eq_y_clamped, eq_x_wrapped])
            face_img[:, :, c] = map_coordinates(
                equirect_img[:, :, c],
                coords,
                order=1,  # Linear interpolation
                mode='wrap',
                prefilter=False
            )

        cubemap.set_face(face_name, face_img)

    return cubemap


def cubemap_to_equirect(
    cubemap: CubemapData,
    width: int,
    height: int,
    use_depth: bool = False
) -> np.ndarray:
    """
    Convert cubemap back to equirectangular format.

    Args:
        cubemap: CubemapData object with populated faces
        width: Output image width (typically 2 * height)
        height: Output image height
        use_depth: If True, convert depth faces instead of RGB

    Returns:
        Equirectangular image array [height, width, C]

    Raises:
        ValueError: If cubemap faces are not set
    """
    # Validate cubemap
    valid, msg = cubemap.validate()
    if not valid:
        raise ValueError(f"Invalid cubemap: {msg}")

    # Determine which faces to use
    if use_depth:
        if not cubemap.has_depth:
            raise ValueError("Cubemap has no depth information")
        face_dict = cubemap.depth_faces
        channels = 1
    else:
        face_dict = cubemap.faces
        # Get channels from first face
        first_face = next(iter(face_dict.values()))
        channels = first_face.shape[2] if first_face.ndim == 3 else 1

    # Create output image
    equirect = np.zeros((height, width, channels), dtype=first_face.dtype)

    for y in range(height):
        # Latitude from pixel y
        lat = (np.pi / 2) - (y * np.pi / height)

        for x in range(width):
            # Longitude from pixel x
            lon = (x * 2 * np.pi / width) - np.pi

            # Convert to 3D vector
            vec = lonlat_to_vector(lon, lat)

            # Determine which face and coordinates
            face_name, face_x, face_y = vector_to_face_coords(vec, cubemap.resolution)

            # Sample from appropriate face
            face_img = face_dict[face_name]
            equirect[y, x] = face_img[face_y, face_x]

    return equirect


def cubemap_to_equirect_fast(
    cubemap: CubemapData,
    width: int,
    height: int,
    use_depth: bool = False
) -> np.ndarray:
    """
    Vectorized version of cubemap_to_equirect for better performance.

    Args:
        cubemap: CubemapData object with populated faces
        width: Output image width
        height: Output image height
        use_depth: If True, convert depth faces instead of RGB

    Returns:
        Equirectangular image array [height, width, C]
    """
    # Determine which faces to use
    if use_depth:
        if not cubemap.has_depth:
            raise ValueError("Cubemap has no depth information")
        face_dict = cubemap.depth_faces
    else:
        face_dict = cubemap.faces

    first_face = next(iter(face_dict.values()))
    channels = first_face.shape[2] if first_face.ndim == 3 else 1
    dtype = first_face.dtype

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Convert to lon/lat
    lon = (x_coords * 2 * np.pi / width) - np.pi
    lat = (np.pi / 2) - (y_coords * np.pi / height)

    # Convert to 3D vectors
    vecs_x = np.cos(lat) * np.sin(lon)
    vecs_y = np.sin(lat)
    vecs_z = np.cos(lat) * np.cos(lon)

    # Initialize output
    equirect = np.zeros((height, width, channels), dtype=dtype)

    # For each face, find pixels that map to it and sample
    abs_x = np.abs(vecs_x)
    abs_y = np.abs(vecs_y)
    abs_z = np.abs(vecs_z)

    # Determine dominant axis for each pixel
    max_axis = np.argmax(np.stack([abs_x, abs_y, abs_z], axis=0), axis=0)

    # Process each face
    for face_idx, (face_name, face_check) in enumerate([
        ('right', (max_axis == 0) & (vecs_x > 0)),
        ('left', (max_axis == 0) & (vecs_x < 0)),
        ('top', (max_axis == 1) & (vecs_y > 0)),
        ('bottom', (max_axis == 1) & (vecs_y < 0)),
        ('front', (max_axis == 2) & (vecs_z > 0)),
        ('back', (max_axis == 2) & (vecs_z < 0))
    ]):
        mask = face_check
        if not np.any(mask):
            continue

        # Get vectors for this face
        x_masked = vecs_x[mask]
        y_masked = vecs_y[mask]
        z_masked = vecs_z[mask]

        # Calculate UV coordinates based on face
        if 'right' in face_name:
            u = -z_masked / abs_x[mask]
            v = y_masked / abs_x[mask]
        elif 'left' in face_name:
            u = z_masked / abs_x[mask]
            v = y_masked / abs_x[mask]
        elif 'top' in face_name:
            u = x_masked / abs_y[mask]
            v = -z_masked / abs_y[mask]
        elif 'bottom' in face_name:
            u = x_masked / abs_y[mask]
            v = z_masked / abs_y[mask]
        elif 'front' in face_name:
            u = x_masked / abs_z[mask]
            v = y_masked / abs_z[mask]
        elif 'back' in face_name:
            u = -x_masked / abs_z[mask]
            v = y_masked / abs_z[mask]

        # Convert to pixel coordinates
        face_x = ((u + 1.0) * cubemap.resolution / 2.0).astype(int)
        face_y = ((1.0 - v) * cubemap.resolution / 2.0).astype(int)

        # Clamp
        face_x = np.clip(face_x, 0, cubemap.resolution - 1)
        face_y = np.clip(face_y, 0, cubemap.resolution - 1)

        # Sample from face
        face_img = face_dict[face_name]
        equirect[mask] = face_img[face_y, face_x]

    return equirect
