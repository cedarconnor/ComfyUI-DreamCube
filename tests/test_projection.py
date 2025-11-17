"""
Test suite for projection mathematics.

Run with: pytest tests/test_projection.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.projection import (
    equirect_to_cubemap,
    cubemap_to_equirect,
    face_coords_to_vector,
    vector_to_lonlat,
    lonlat_to_vector,
    vector_to_face_coords
)
from core.cubemap import CubemapData


def test_face_coords_to_vector():
    """Test face coordinate to vector conversion."""
    # Center of front face should point forward (z=1)
    vec = face_coords_to_vector('front', 256, 256, 512)
    assert abs(vec[2]) > 0.9  # Mostly Z
    assert np.linalg.norm(vec) == pytest.approx(1.0)  # Unit vector


def test_vector_to_lonlat():
    """Test 3D vector to longitude/latitude conversion."""
    # Forward vector should be lon=0, lat=0
    vec = np.array([0.0, 0.0, 1.0])
    lon, lat = vector_to_lonlat(vec)
    assert lon == pytest.approx(0.0, abs=0.01)
    assert lat == pytest.approx(0.0, abs=0.01)


def test_lonlat_to_vector():
    """Test longitude/latitude to vector conversion."""
    lon, lat = 0.0, 0.0
    vec = lonlat_to_vector(lon, lat)
    assert vec[2] == pytest.approx(1.0, abs=0.01)  # Forward
    assert np.linalg.norm(vec) == pytest.approx(1.0)


def test_roundtrip_conversion():
    """Test vector → lonlat → vector roundtrip."""
    original_vec = np.array([0.5, 0.3, 0.7])
    original_vec /= np.linalg.norm(original_vec)

    lon, lat = vector_to_lonlat(original_vec)
    recovered_vec = lonlat_to_vector(lon, lat)

    assert np.allclose(original_vec, recovered_vec, atol=0.01)


def test_vector_to_face_coords():
    """Test vector to face coordinate conversion."""
    # Forward vector should map to front face
    vec = np.array([0.0, 0.0, 1.0])
    face, x, y = vector_to_face_coords(vec, 512)
    assert face == 'front'
    # Should be near center
    assert 200 < x < 312
    assert 200 < y < 312


def test_cubemap_creation():
    """Test cubemap data structure creation."""
    cubemap = CubemapData(resolution=512)
    assert cubemap.resolution == 512
    assert len(cubemap.faces) == 6
    assert cubemap.has_depth == False


def test_cubemap_face_operations():
    """Test setting and getting cubemap faces."""
    cubemap = CubemapData(resolution=256)
    test_face = np.random.rand(256, 256, 3)

    cubemap.set_face('front', test_face)
    retrieved = cubemap.get_face('front')

    assert np.array_equal(test_face, retrieved)


def test_equirect_to_cubemap_basic():
    """Test basic equirectangular to cubemap conversion."""
    # Create simple equirect image (2:1 ratio)
    equirect = np.random.rand(512, 1024, 3)

    # Convert to cubemap
    cubemap = equirect_to_cubemap(equirect, face_size=256)

    # Check all faces are set
    assert cubemap.all_faces_set()
    assert cubemap.resolution == 256

    # Check each face has correct shape
    for face_name in cubemap.get_face_names():
        face = cubemap.get_face(face_name)
        assert face.shape == (256, 256, 3)


def test_cubemap_to_equirect_basic():
    """Test basic cubemap to equirectangular conversion."""
    # Create cubemap with random faces
    cubemap = CubemapData(resolution=256)
    for face_name in cubemap.get_face_names():
        face = np.random.rand(256, 256, 3)
        cubemap.set_face(face_name, face)

    # Convert to equirect
    equirect = cubemap_to_equirect(cubemap, width=1024, height=512)

    # Check output shape
    assert equirect.shape == (512, 1024, 3)


def test_roundtrip_projection():
    """Test equirect → cubemap → equirect roundtrip."""
    # Create test equirect image
    original = np.random.rand(256, 512, 3)

    # Convert to cubemap and back
    cubemap = equirect_to_cubemap(original, face_size=128)
    recovered = cubemap_to_equirect(cubemap, width=512, height=256)

    # Check shape preservation
    assert recovered.shape == original.shape

    # Note: Perfect reconstruction isn't expected due to sampling,
    # but shapes and rough values should match
    assert recovered.min() >= 0.0
    assert recovered.max() <= 1.0


def test_cubemap_validation():
    """Test cubemap validation."""
    cubemap = CubemapData(resolution=256)

    # Should fail - no faces set
    valid, msg = cubemap.validate()
    assert not valid

    # Set all faces
    for face_name in cubemap.get_face_names():
        cubemap.set_face(face_name, np.zeros((256, 256, 3)))

    # Should pass
    valid, msg = cubemap.validate()
    assert valid


def test_cubemap_adjacency():
    """Test cubemap adjacency mapping."""
    cubemap = CubemapData(resolution=256)
    adj_map = cubemap.get_adjacency_map()

    # Front face should have 4 neighbors
    assert len(adj_map['front']) == 4
    assert 'left' in adj_map['front']
    assert 'right' in adj_map['front']
    assert 'top' in adj_map['front']
    assert 'bottom' in adj_map['front']

    # Check all faces have adjacency info
    for face_name in cubemap.get_face_names():
        assert face_name in adj_map
        assert len(adj_map[face_name]) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
