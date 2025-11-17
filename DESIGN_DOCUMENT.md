# ComfyUI-DreamCube Design Document

## Project Overview

**Project Name**: ComfyUI-DreamCube  
**Repository**: ComfyUI_DreamCube  
**Author**: Cedar  
**Version**: 1.0.0  
**Target**: ComfyUI custom node pack for 360° panoramic depth estimation using cubemap-based multi-plane synchronization

### Objectives

1. Implement DreamCube's multi-plane synchronization framework within ComfyUI
2. Enable modular depth estimation on cubemap faces using any external depth model
3. Provide seamless conversion between equirectangular and cubemap formats
4. Maintain cross-face consistency through synchronized processing
5. Support integration with existing ComfyUI depth estimation workflows

---

## Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ComfyUI-DreamCube                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │   Projection     │      │   Multi-plane    │           │
│  │   Conversion     │──────│   Sync Engine    │           │
│  └──────────────────┘      └──────────────────┘           │
│          │                          │                      │
│          │                          │                      │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │   Cubemap        │      │   Depth Model    │           │
│  │   Processing     │──────│   Interface      │           │
│  └──────────────────┘      └──────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Module Breakdown

#### 1. **Projection Conversion Module**
- **Equirect to Cubemap**: Converts 2:1 equirectangular images to 6-face cubemap
- **Cubemap to Equirect**: Reconstructs equirectangular from cubemap faces
- **Face Extraction**: Isolates individual faces for processing
- **Coordinate Mapping**: Handles pixel-accurate projection transformations

#### 2. **Multi-plane Synchronization Engine**
- **Attention Synchronization**: Coordinates cross-face attention mechanisms
- **Convolution Sync**: Ensures boundary consistency in convolutional operations
- **Group Norm Sync**: Normalizes features across all faces simultaneously
- **Seam Blending**: Manages edge transitions between adjacent faces

#### 3. **Cubemap Processing Module**
- **Face Orientation**: Manages 6 face coordinate systems (front, back, left, right, top, bottom)
- **Adjacency Graph**: Tracks spatial relationships between faces
- **Padding Management**: Applies appropriate padding at face boundaries
- **Batch Processing**: Processes all 6 faces efficiently

#### 4. **Depth Model Interface**
- **Generic Depth Input**: Accepts depth maps from any ComfyUI depth node
- **Per-Face Processing**: Applies depth estimation independently per face
- **Depth Normalization**: Scales and aligns depth values across faces
- **Consistency Validation**: Checks depth continuity at seams

---

## Node Architecture

### Node Types

#### Input/Output Nodes

**1. EquirectToCubemap**
```python
INPUT_TYPES = {
    "required": {
        "image": ("IMAGE",),  # Equirectangular input
        "cube_resolution": ("INT", {"default": 1024, "min": 256, "max": 4096}),
    }
}
RETURN_TYPES = ("CUBEMAP",)
```

**2. CubemapToEquirect**
```python
INPUT_TYPES = {
    "required": {
        "cubemap": ("CUBEMAP",),
        "output_width": ("INT", {"default": 2048, "min": 512, "max": 8192}),
        "output_height": ("INT", {"default": 1024, "min": 256, "max": 4096}),
    }
}
RETURN_TYPES = ("IMAGE",)
```

**3. ExtractCubemapFace**
```python
INPUT_TYPES = {
    "required": {
        "cubemap": ("CUBEMAP",),
        "face": (["front", "back", "left", "right", "top", "bottom"],),
    }
}
RETURN_TYPES = ("IMAGE",)
```

#### Processing Nodes

**4. ApplyDepthToCubemapFace**
```python
INPUT_TYPES = {
    "required": {
        "cubemap": ("CUBEMAP",),
        "depth_map": ("IMAGE",),  # Depth from ANY depth node
        "face": (["front", "back", "left", "right", "top", "bottom"],),
    }
}
RETURN_TYPES = ("CUBEMAP",)
```

**5. BatchCubemapDepthEstimation**
```python
INPUT_TYPES = {
    "required": {
        "cubemap": ("CUBEMAP",),
        "depth_model_name": (folder_paths.get_filename_list("depth_anything"),),
    },
    "optional": {
        "sync_enabled": ("BOOLEAN", {"default": True}),
        "sync_method": (["attention", "conv", "full"],),
    }
}
RETURN_TYPES = ("CUBEMAP",)  # Returns cubemap with depth channel
```

**6. MultiplaneSyncProcessor**
```python
INPUT_TYPES = {
    "required": {
        "cubemap_rgb": ("CUBEMAP",),
        "cubemap_depth": ("CUBEMAP",),
        "sync_self_attn": ("BOOLEAN", {"default": True}),
        "sync_conv2d": ("BOOLEAN", {"default": True}),
        "sync_group_norm": ("BOOLEAN", {"default": True}),
    }
}
RETURN_TYPES = ("CUBEMAP",)
```

#### Utility Nodes

**7. CubemapPreview**
```python
INPUT_TYPES = {
    "required": {
        "cubemap": ("CUBEMAP",),
        "layout": (["horizontal", "cross", "vertical"],),
    }
}
RETURN_TYPES = ("IMAGE",)
```

**8. CubemapSeamValidator**
```python
INPUT_TYPES = {
    "required": {
        "cubemap": ("CUBEMAP",),
        "threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0}),
    }
}
RETURN_TYPES = ("BOOLEAN", "FLOAT",)  # is_valid, max_error
```

**9. MergeCubemapDepth**
```python
INPUT_TYPES = {
    "required": {
        "cubemap_rgb": ("CUBEMAP",),
        "cubemap_depth": ("CUBEMAP",),
    }
}
RETURN_TYPES = ("CUBEMAP",)  # 4-channel RGBD cubemap
```

---

## Data Structures

### Cubemap Data Type

```python
class CubemapData:
    """
    Represents a 6-face cubemap with optional depth channel
    """
    def __init__(self, resolution: int):
        self.resolution = resolution
        self.faces = {
            'front': None,   # [H, W, C] numpy array
            'back': None,
            'left': None,
            'right': None,
            'top': None,
            'bottom': None
        }
        self.depth_faces = {
            'front': None,   # [H, W, 1] depth maps
            'back': None,
            'left': None,
            'right': None,
            'top': None,
            'bottom': None
        }
        self.has_depth = False
        
    def get_adjacency_map(self):
        """
        Returns dict of adjacent faces for each face with edge mapping
        """
        return {
            'front': {'right': 'left', 'left': 'right', 'top': 'bottom', 'bottom': 'top'},
            'back': {'right': 'right', 'left': 'left', 'top': 'top', 'bottom': 'bottom'},
            'left': {'front': 'right', 'back': 'right', 'top': 'left', 'bottom': 'left'},
            'right': {'front': 'left', 'back': 'left', 'top': 'right', 'bottom': 'right'},
            'top': {'front': 'top', 'back': 'bottom', 'left': 'top', 'right': 'top'},
            'bottom': {'front': 'bottom', 'back': 'top', 'left': 'bottom', 'right': 'bottom'}
        }
```

---

## Implementation Details

### Phase 1: Core Infrastructure

#### 1.1 Projection Mathematics

**Equirectangular to Cubemap**
```python
def equirect_to_cubemap(equirect_img: np.ndarray, face_size: int) -> CubemapData:
    """
    Convert equirectangular (2:1) to cubemap
    
    Args:
        equirect_img: [H, W, C] array where H = W/2
        face_size: Resolution of each cube face
        
    Returns:
        CubemapData object with 6 populated faces
    """
    h, w = equirect_img.shape[:2]
    cubemap = CubemapData(face_size)
    
    # For each face
    for face_name in ['front', 'back', 'left', 'right', 'top', 'bottom']:
        face_img = np.zeros((face_size, face_size, equirect_img.shape[2]))
        
        # Generate coordinate grid for this face
        for y in range(face_size):
            for x in range(face_size):
                # Convert face coordinates to 3D unit vector
                vec = face_coords_to_vector(face_name, x, y, face_size)
                
                # Convert 3D vector to equirectangular coordinates
                lon, lat = vector_to_lonlat(vec)
                
                # Sample from equirectangular image
                eq_x = int((lon + np.pi) / (2 * np.pi) * w) % w
                eq_y = int((np.pi / 2 - lat) / np.pi * h)
                eq_y = np.clip(eq_y, 0, h - 1)
                
                face_img[y, x] = equirect_img[eq_y, eq_x]
        
        cubemap.faces[face_name] = face_img
    
    return cubemap


def face_coords_to_vector(face: str, x: int, y: int, size: int) -> np.ndarray:
    """
    Convert 2D face coordinates to 3D unit vector
    
    Coordinate system:
    - X: right
    - Y: up
    - Z: forward (front face normal)
    """
    # Normalize to [-1, 1]
    u = (2.0 * x / size) - 1.0
    v = 1.0 - (2.0 * y / size)  # Flip Y
    
    vectors = {
        'front':  np.array([u, v, 1.0]),
        'back':   np.array([-u, v, -1.0]),
        'left':   np.array([-1.0, v, u]),
        'right':  np.array([1.0, v, -u]),
        'top':    np.array([u, 1.0, -v]),
        'bottom': np.array([u, -1.0, v])
    }
    
    vec = vectors[face]
    return vec / np.linalg.norm(vec)


def vector_to_lonlat(vec: np.ndarray) -> tuple:
    """
    Convert 3D unit vector to longitude/latitude
    
    Returns:
        (longitude, latitude) in radians
        longitude: [-π, π]
        latitude: [-π/2, π/2]
    """
    lon = np.arctan2(vec[0], vec[2])
    lat = np.arcsin(vec[1])
    return lon, lat
```

**Cubemap to Equirectangular**
```python
def cubemap_to_equirect(cubemap: CubemapData, width: int, height: int) -> np.ndarray:
    """
    Convert cubemap back to equirectangular
    
    Args:
        cubemap: CubemapData object
        width: Output width (typically 2 * height)
        height: Output height
        
    Returns:
        [height, width, C] equirectangular image
    """
    equirect = np.zeros((height, width, 3))
    
    for y in range(height):
        lat = (np.pi / 2) - (y * np.pi / height)
        
        for x in range(width):
            lon = (x * 2 * np.pi / width) - np.pi
            
            # Convert to 3D vector
            vec = lonlat_to_vector(lon, lat)
            
            # Determine which face and coordinates
            face_name, face_x, face_y = vector_to_face_coords(vec, cubemap.resolution)
            
            # Sample from appropriate face
            face_img = cubemap.faces[face_name]
            equirect[y, x] = face_img[face_y, face_x]
    
    return equirect


def lonlat_to_vector(lon: float, lat: float) -> np.ndarray:
    """Convert longitude/latitude to 3D unit vector"""
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    return np.array([x, y, z])


def vector_to_face_coords(vec: np.ndarray, face_size: int) -> tuple:
    """
    Determine which face a vector points to and its coordinates
    
    Returns:
        (face_name, x, y) where x, y are in [0, face_size)
    """
    abs_vec = np.abs(vec)
    max_axis = np.argmax(abs_vec)
    
    # Determine face
    if max_axis == 0:  # X dominant
        face = 'right' if vec[0] > 0 else 'left'
        u, v = -vec[2] / abs_vec[0], vec[1] / abs_vec[0]
        if face == 'left':
            u = -u
    elif max_axis == 1:  # Y dominant
        face = 'top' if vec[1] > 0 else 'bottom'
        u, v = vec[0] / abs_vec[1], -vec[2] / abs_vec[1]
        if face == 'bottom':
            v = -v
    else:  # Z dominant
        face = 'front' if vec[2] > 0 else 'back'
        u, v = vec[0] / abs_vec[2], vec[1] / abs_vec[2]
        if face == 'back':
            u = -u
    
    # Convert from [-1, 1] to pixel coordinates
    x = int((u + 1.0) * face_size / 2.0)
    y = int((1.0 - v) * face_size / 2.0)
    
    # Clamp to valid range
    x = np.clip(x, 0, face_size - 1)
    y = np.clip(y, 0, face_size - 1)
    
    return face, x, y
```

#### 1.2 Multi-plane Synchronization

**Attention Synchronization**
```python
class SyncedSelfAttention:
    """
    Synchronized self-attention across cubemap faces
    Ensures boundary pixels attend to adjacent face pixels
    """
    def __init__(self, dim: int, num_heads: int = 8):
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Standard attention components
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, face_features: dict, adjacency_map: dict):
        """
        Args:
            face_features: Dict of {face_name: [B, H, W, C]}
            adjacency_map: Dict defining face adjacencies
            
        Returns:
            Dict of {face_name: [B, H, W, C]} synchronized features
        """
        B, H, W, C = next(iter(face_features.values())).shape
        
        synced_features = {}
        
        for face_name, features in face_features.items():
            # Get QKV for this face
            qkv = self.qkv(features)  # [B, H, W, 3*C]
            q, k, v = qkv.chunk(3, dim=-1)
            
            # Reshape for multi-head attention
            q = q.reshape(B, H, W, self.num_heads, -1)
            k = k.reshape(B, H, W, self.num_heads, -1)
            v = v.reshape(B, H, W, self.num_heads, -1)
            
            # Collect K, V from adjacent faces at boundaries
            boundary_kv = self._collect_boundary_kv(
                face_name, face_features, adjacency_map
            )
            
            # Compute attention with boundary context
            attn_out = self._compute_boundary_aware_attention(
                q, k, v, boundary_kv
            )
            
            # Project back
            synced_features[face_name] = self.proj(attn_out)
        
        return synced_features
    
    def _collect_boundary_kv(self, face_name: str, face_features: dict, 
                              adjacency_map: dict, boundary_width: int = 8):
        """
        Collect key/value features from boundary regions of adjacent faces
        """
        adjacent_faces = adjacency_map[face_name]
        boundary_kv = {}
        
        for adj_face, edge_mapping in adjacent_faces.items():
            adj_features = face_features[adj_face]
            
            # Extract boundary strip from adjacent face
            if edge_mapping == 'left':
                boundary = adj_features[:, :, :boundary_width, :]
            elif edge_mapping == 'right':
                boundary = adj_features[:, :, -boundary_width:, :]
            elif edge_mapping == 'top':
                boundary = adj_features[:, :boundary_width, :, :]
            elif edge_mapping == 'bottom':
                boundary = adj_features[:, -boundary_width:, :, :]
            
            boundary_kv[adj_face] = boundary
        
        return boundary_kv
    
    def _compute_boundary_aware_attention(self, q, k, v, boundary_kv):
        """
        Compute attention with additional context from boundaries
        """
        # Standard self-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        
        # Add boundary attention for edge pixels
        # (Implementation details for merging boundary attention)
        
        return out
```

**Convolution Synchronization**
```python
class SyncedConv2d:
    """
    Synchronized 2D convolution with cross-face padding
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, padding: int = 1):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=0)  # We handle padding manually
        self.kernel_size = kernel_size
        self.padding = padding
        
    def forward(self, face_features: dict, adjacency_map: dict):
        """
        Apply convolution with padding from adjacent faces
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
    
    def _add_cross_face_padding(self, face_name: str, features: torch.Tensor,
                                face_features: dict, adjacency_map: dict):
        """
        Pad feature map with data from adjacent faces
        
        Args:
            face_name: Current face
            features: [B, C, H, W]
            face_features: Dict of all face features
            adjacency_map: Face adjacency information
            
        Returns:
            [B, C, H+2p, W+2p] padded features
        """
        B, C, H, W = features.shape
        p = self.padding
        
        # Initialize padded tensor
        padded = torch.zeros(B, C, H + 2*p, W + 2*p, device=features.device)
        padded[:, :, p:H+p, p:W+p] = features
        
        adjacent_faces = adjacency_map[face_name]
        
        # Fill padding regions from adjacent faces
        for adj_face, edge in adjacent_faces.items():
            adj_features = face_features[adj_face]
            
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
```

### Phase 2: Depth Model Integration

#### 2.1 Generic Depth Interface

```python
class DepthModelInterface:
    """
    Generic interface for applying any depth model to cubemap faces
    """
    def __init__(self):
        self.supported_formats = ['comfyui_depth_anything', 'midas', 'custom']
    
    def estimate_depth_per_face(self, cubemap: CubemapData, 
                                depth_node_output: torch.Tensor,
                                face_name: str) -> np.ndarray:
        """
        Apply depth estimation to a single cubemap face
        
        Args:
            cubemap: CubemapData object
            depth_node_output: Output from any ComfyUI depth node
            face_name: Which face to process
            
        Returns:
            [H, W, 1] depth map for specified face
        """
        face_img = cubemap.faces[face_name]
        
        # Depth is already computed by upstream node
        # Just assign it to the appropriate face
        depth_map = depth_node_output
        
        # Normalize depth to [0, 1]
        depth_map = self._normalize_depth(depth_map)
        
        return depth_map
    
    def estimate_all_faces(self, cubemap: CubemapData, 
                          depth_estimator_callback) -> CubemapData:
        """
        Process all 6 faces through depth estimation
        
        Args:
            cubemap: Input cubemap with RGB faces
            depth_estimator_callback: Function that takes an image and returns depth
            
        Returns:
            CubemapData with depth_faces populated
        """
        for face_name in cubemap.faces.keys():
            face_rgb = cubemap.faces[face_name]
            
            # Call external depth estimator (e.g., Depth Anything node)
            depth_map = depth_estimator_callback(face_rgb)
            
            cubemap.depth_faces[face_name] = depth_map
        
        cubemap.has_depth = True
        return cubemap
    
    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth to [0, 1] range"""
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min < 1e-6:
            return depth
        return (depth - d_min) / (d_max - d_min)
```

#### 2.2 Depth Consistency Enforcement

```python
class DepthConsistencyEnforcer:
    """
    Ensures depth continuity across cubemap face boundaries
    """
    def __init__(self, boundary_width: int = 16):
        self.boundary_width = boundary_width
        
    def enforce_consistency(self, cubemap: CubemapData, 
                          adjacency_map: dict) -> CubemapData:
        """
        Smooth depth values at face boundaries
        
        Args:
            cubemap: CubemapData with depth_faces populated
            adjacency_map: Face adjacency information
            
        Returns:
            CubemapData with smoothed depth at boundaries
        """
        for face_name, depth_face in cubemap.depth_faces.items():
            adjacent_faces = adjacency_map[face_name]
            
            for adj_face, edge in adjacent_faces.items():
                # Get depth from adjacent face boundary
                adj_depth = cubemap.depth_faces[adj_face]
                
                # Blend depths at boundary
                blended_depth = self._blend_boundary(
                    depth_face, adj_depth, edge
                )
                
                cubemap.depth_faces[face_name] = blended_depth
        
        return cubemap
    
    def _blend_boundary(self, depth: np.ndarray, adj_depth: np.ndarray, 
                       edge: str) -> np.ndarray:
        """
        Blend depth values at face boundary
        
        Uses weighted average based on distance from boundary
        """
        H, W = depth.shape[:2]
        w = self.boundary_width
        
        # Create blend weights (1.0 at center, 0.5 at boundary)
        weights = np.linspace(0.5, 1.0, w)
        
        if edge == 'left':
            # Blend left edge with right edge of adjacent face
            for i in range(w):
                alpha = weights[i]
                depth[:, i] = alpha * depth[:, i] + (1 - alpha) * adj_depth[:, -(w-i)]
                
        elif edge == 'right':
            # Blend right edge with left edge of adjacent face
            for i in range(w):
                alpha = weights[w - i - 1]
                depth[:, -(i+1)] = alpha * depth[:, -(i+1)] + (1 - alpha) * adj_depth[:, i]
                
        elif edge == 'top':
            # Blend top edge with bottom edge of adjacent face
            for i in range(w):
                alpha = weights[i]
                depth[i, :] = alpha * depth[i, :] + (1 - alpha) * adj_depth[-(w-i), :]
                
        elif edge == 'bottom':
            # Blend bottom edge with top edge of adjacent face
            for i in range(w):
                alpha = weights[w - i - 1]
                depth[-(i+1), :] = alpha * depth[-(i+1), :] + (1 - alpha) * adj_depth[i, :]
        
        return depth
    
    def validate_seams(self, cubemap: CubemapData, 
                      adjacency_map: dict) -> tuple[bool, float]:
        """
        Validate depth continuity at seams
        
        Returns:
            (is_valid, max_error) where max_error is maximum depth difference at seams
        """
        max_error = 0.0
        
        for face_name, depth_face in cubemap.depth_faces.items():
            adjacent_faces = adjacency_map[face_name]
            
            for adj_face, edge in adjacent_faces.items():
                adj_depth = cubemap.depth_faces[adj_face]
                
                # Compare boundary values
                error = self._compute_boundary_error(depth_face, adj_depth, edge)
                max_error = max(max_error, error)
        
        is_valid = max_error < 0.05  # Threshold for valid seams
        return is_valid, max_error
    
    def _compute_boundary_error(self, depth: np.ndarray, 
                               adj_depth: np.ndarray, edge: str) -> float:
        """
        Compute maximum depth difference at boundary
        """
        if edge == 'left':
            diff = np.abs(depth[:, 0] - adj_depth[:, -1])
        elif edge == 'right':
            diff = np.abs(depth[:, -1] - adj_depth[:, 0])
        elif edge == 'top':
            diff = np.abs(depth[0, :] - adj_depth[-1, :])
        elif edge == 'bottom':
            diff = np.abs(depth[-1, :] - adj_depth[0, :])
        
        return float(np.max(diff))
```

### Phase 3: ComfyUI Integration

#### 3.1 Node Implementation Examples

```python
class EquirectToCubemapNode:
    """
    ComfyUI node to convert equirectangular to cubemap
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cube_resolution": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 256
                }),
            }
        }
    
    RETURN_TYPES = ("CUBEMAP",)
    FUNCTION = "convert"
    CATEGORY = "DreamCube/Projection"
    
    def convert(self, image, cube_resolution):
        # Convert from ComfyUI tensor format [B, H, W, C]
        equirect_np = image.cpu().numpy()[0]  # Take first batch
        
        # Convert to cubemap
        cubemap = equirect_to_cubemap(equirect_np, cube_resolution)
        
        return (cubemap,)


class BatchCubemapDepthNode:
    """
    Process all 6 cubemap faces through external depth model
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap_rgb": ("CUBEMAP",),
                "depth_front": ("IMAGE",),   # From external depth node
                "depth_back": ("IMAGE",),
                "depth_left": ("IMAGE",),
                "depth_right": ("IMAGE",),
                "depth_top": ("IMAGE",),
                "depth_bottom": ("IMAGE",),
                "enforce_consistency": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("CUBEMAP",)
    FUNCTION = "apply_depth"
    CATEGORY = "DreamCube/Depth"
    
    def apply_depth(self, cubemap_rgb, depth_front, depth_back, 
                   depth_left, depth_right, depth_top, depth_bottom,
                   enforce_consistency):
        
        # Assign depth maps to cubemap faces
        face_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
        depth_maps = [depth_front, depth_back, depth_left, 
                     depth_right, depth_top, depth_bottom]
        
        for face_name, depth_map in zip(face_names, depth_maps):
            depth_np = depth_map.cpu().numpy()[0]
            cubemap_rgb.depth_faces[face_name] = depth_np
        
        cubemap_rgb.has_depth = True
        
        # Enforce boundary consistency if requested
        if enforce_consistency:
            adjacency_map = cubemap_rgb.get_adjacency_map()
            enforcer = DepthConsistencyEnforcer()
            cubemap_rgb = enforcer.enforce_consistency(
                cubemap_rgb, adjacency_map
            )
        
        return (cubemap_rgb,)


class CubemapToEquirectNode:
    """
    Convert cubemap back to equirectangular
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap": ("CUBEMAP",),
                "output_width": ("INT", {"default": 2048, "min": 512, "max": 8192}),
                "output_height": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "output_type": (["rgb", "depth", "rgbd"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "DreamCube/Projection"
    
    def convert(self, cubemap, output_width, output_height, output_type):
        if output_type == "rgb":
            equirect = cubemap_to_equirect(cubemap, output_width, output_height)
        elif output_type == "depth":
            # Create temporary cubemap with depth as RGB
            depth_cubemap = CubemapData(cubemap.resolution)
            for face in cubemap.faces.keys():
                d = cubemap.depth_faces[face]
                depth_cubemap.faces[face] = np.stack([d, d, d], axis=-1)
            equirect = cubemap_to_equirect(depth_cubemap, output_width, output_height)
        elif output_type == "rgbd":
            # Combine RGB and depth
            rgb_equirect = cubemap_to_equirect(cubemap, output_width, output_height)
            depth_cubemap = CubemapData(cubemap.resolution)
            for face in cubemap.faces.keys():
                depth_cubemap.faces[face] = cubemap.depth_faces[face]
            depth_equirect = cubemap_to_equirect(depth_cubemap, output_width, output_height)
            equirect = np.concatenate([rgb_equirect, depth_equirect[..., :1]], axis=-1)
        
        # Convert back to ComfyUI tensor format
        tensor = torch.from_numpy(equirect).unsqueeze(0)  # Add batch dim
        
        return (tensor,)
```

---

## Example Workflows

### Workflow 1: Basic Equirect → Cubemap → Depth → Equirect

```
[Load Image]
    ↓ (IMAGE)
[EquirectToCubemap]
    ↓ (CUBEMAP)
[Extract Cubemap Faces] × 6
    ↓ (IMAGE) × 6
[Depth Anything V2 Node] × 6
    ↓ (IMAGE) × 6
[Batch Cubemap Depth]
    ↓ (CUBEMAP with depth)
[Cubemap To Equirect]
    ↓ (IMAGE)
[Save Image]
```

### Workflow 2: With Multi-plane Synchronization

```
[Load Image]
    ↓
[Equirect To Cubemap]
    ↓ (CUBEMAP)
    ├─→ [Extract Face: Front] → [DA3 Node] ─┐
    ├─→ [Extract Face: Back]  → [DA3 Node] ─┤
    ├─→ [Extract Face: Left]  → [DA3 Node] ─┤
    ├─→ [Extract Face: Right] → [DA3 Node] ─┤
    ├─→ [Extract Face: Top]   → [DA3 Node] ─┤
    └─→ [Extract Face: Bottom]→ [DA3 Node] ─┘
                                             ↓
                              [Batch Cubemap Depth]
                                             ↓
                           [Multiplane Sync Processor]
                                             ↓
                              [Cubemap Seam Validator]
                                             ↓
                              [Cubemap To Equirect]
                                             ↓
                                      [Save Image]
```

### Workflow 3: Integration with Motion Transfer

```
[Load 360° Video Frame]
    ↓
[Equirect To Cubemap]
    ↓
[Batch Depth Estimation (6 faces)]
    ↓
[Multiplane Sync]
    ↓
[Cubemap To Equirect Depth]
    ↓
[ComfyUI_MotionTransfer Nodes]
    ↓
[Output]
```

---

## File Structure

```
ComfyUI_DreamCube/
├── __init__.py
├── nodes/
│   ├── __init__.py
│   ├── projection_nodes.py      # Equirect ↔ Cubemap conversion
│   ├── depth_nodes.py            # Depth estimation interfaces
│   ├── sync_nodes.py             # Multi-plane synchronization
│   ├── utility_nodes.py          # Preview, validation, etc.
│   └── integration_nodes.py     # High-level workflow nodes
├── core/
│   ├── __init__.py
│   ├── cubemap.py                # CubemapData class
│   ├── projection.py             # Projection math
│   ├── synchronization.py        # Sync algorithms
│   ├── depth_interface.py        # Generic depth interface
│   └── consistency.py            # Boundary consistency
├── utils/
│   ├── __init__.py
│   ├── visualization.py          # Cubemap preview layouts
│   └── validation.py             # Quality metrics
├── tests/
│   ├── test_projection.py
│   ├── test_synchronization.py
│   └── test_consistency.py
├── examples/
│   ├── workflows/
│   │   ├── basic_depth.json
│   │   ├── synced_depth.json
│   │   └── motion_transfer_integration.json
│   └── sample_images/
└── README.md
```

---

## Performance Considerations

### Memory Optimization

1. **Lazy Face Processing**: Only load faces into memory when needed
2. **Streaming**: Process faces sequentially for low-VRAM systems
3. **Resolution Scaling**: Offer dynamic resolution adjustment
4. **Batch Processing**: Group operations to minimize GPU transfers

### Speed Optimization

1. **CUDA Kernels**: Implement projection math in CUDA for speed
2. **Caching**: Cache cubemap conversions for video processing
3. **Parallel Face Processing**: Use `torch.nn.parallel` for 6 faces
4. **Optimized Sampling**: Use `torch.nn.functional.grid_sample` for reprojection

### Recommended Hardware

- **Minimum**: 8GB VRAM (1024² faces)
- **Recommended**: 16GB VRAM (2048² faces)
- **Optimal**: 24GB VRAM (4096² faces)

---

## Testing Strategy

### Unit Tests

1. **Projection Accuracy**: Verify equirect ↔ cubemap round-trip
2. **Seam Continuity**: Validate boundary alignment
3. **Depth Consistency**: Check cross-face depth matching
4. **Coordinate Mapping**: Test all edge cases

### Integration Tests

1. **End-to-end Workflows**: Test complete pipelines
2. **Multiple Depth Models**: Verify compatibility
3. **Resolution Scaling**: Test various resolutions
4. **Error Handling**: Validate graceful failures

### Visual Quality Tests

1. **Seam Visibility**: Manual inspection of boundaries
2. **Depth Smoothness**: Check for discontinuities
3. **Round-trip Quality**: Compare input vs. output
4. **Comparative Analysis**: Benchmark against DreamCube reference

---

## Development Roadmap

### Phase 1: Foundation (Weeks 1-2)
- ✅ Implement projection math
- ✅ Create CubemapData structure
- ✅ Basic conversion nodes
- ✅ Unit tests

### Phase 2: Depth Integration (Weeks 3-4)
- ✅ Generic depth interface
- ✅ Per-face depth nodes
- ✅ Batch processing node
- ✅ Consistency enforcement

### Phase 3: Synchronization (Weeks 5-6)
- ✅ Attention synchronization
- ✅ Convolution synchronization
- ✅ Multi-plane sync node
- ✅ Validation tools

### Phase 4: Polish (Weeks 7-8)
- ✅ Optimize performance
- ✅ Add visualization tools
- ✅ Write documentation
- ✅ Create example workflows

### Phase 5: Release (Week 9)
- ✅ Final testing
- ✅ Package for ComfyUI Manager
- ✅ Publish to GitHub
- ✅ Community announcement

---

## Dependencies

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
pillow>=9.5.0
opencv-python>=4.7.0
einops>=0.6.1
```

---

## Licensing

- **Core Framework**: Apache 2.0 (compatible with DreamCube)
- **Node Pack**: Apache 2.0
- **Example Workflows**: CC BY 4.0

---

## Future Extensions

1. **Video Support**: Temporal consistency across frames
2. **Normal Map Generation**: Surface normals from depth
3. **3D Mesh Export**: Convert to OBJ/PLY/GLTF
4. **3D Gaussian Splatting**: Direct integration
5. **LoRA Support**: Fine-tuning for specific styles
6. **Outpainting**: Extend limited FOV to 360°

---

## References

- **DreamCube Paper**: https://arxiv.org/abs/2506.17206
- **DreamCube GitHub**: https://github.com/Yukun-Huang/DreamCube
- **Depth Anything V3**: https://github.com/DepthAnything/Depth-Anything-V3
- **ComfyUI Documentation**: https://github.com/comfyanonymous/ComfyUI
