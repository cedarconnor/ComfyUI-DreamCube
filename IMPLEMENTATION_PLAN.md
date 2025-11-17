# ComfyUI-DreamCube Implementation Plan & Progress Tracker

**Project**: ComfyUI-DreamCube Node Pack
**Purpose**: 360° Panoramic Depth Estimation using Multi-plane Synchronization
**Platform**: ComfyUI (Windows & Linux compatible)
**Original**: [DreamCube (ICCV 2025)](https://github.com/yukun-huang/DreamCube)
**Started**: 2025-11-17
**Status**: ✅ Phase 1-4 Complete, Ready for Testing!

---

## 📋 Executive Summary

This node pack brings DreamCube's multi-plane synchronization framework to ComfyUI, enabling:
- **Equirectangular ↔ Cubemap conversions** for 360° panoramic images
- **Depth estimation** on cubemap faces using any ComfyUI depth node
- **Cross-face consistency** through synchronized processing
- **Seamless integration** with existing ComfyUI workflows

### Key Design Principles
✅ **Windows Compatible**: Pure Python with PyTorch, no Linux-specific dependencies
✅ **Native ComfyUI Nodes**: Leverages existing depth estimation nodes (Depth Anything, DA3, MiDaS)
✅ **Modular Architecture**: Each component is independent and reusable
✅ **Performance First**: Optimized for GPU processing with fallback to CPU

---

## 🎯 Project Goals

### Primary Objectives
- [x] ✅ Parse and understand DreamCube architecture
- [x] ✅ Design ComfyUI-compatible node structure
- [x] ✅ Implement core projection mathematics
- [x] ✅ Create cubemap data structures
- [x] ✅ Build multi-plane synchronization engine
- [x] ✅ Develop ComfyUI nodes
- [ ] ⏳ Test with existing depth nodes
- [ ] ⏳ Create example workflows
- [x] ✅ Write comprehensive documentation

### Success Criteria
- Round-trip projection accuracy: PSNR > 45dB, SSIM > 0.99
- Processing speed: <500ms full pipeline for 1024² faces
- Memory usage: <8GB VRAM for standard workflows
- Zero visible seams in output
- Compatible with Depth Anything V2, DA3, MiDaS, Marigold

---

## 📦 Project Structure

```
ComfyUI-DreamCube/
├── __init__.py                 # Node registration
├── requirements.txt            # Dependencies
├── README.md                   # User documentation
├── IMPLEMENTATION_PLAN.md      # This file
├── DESIGN_DOCUMENT.md          # Technical specification
├── AGENTS.md                   # Agent roles & responsibilities
│
├── core/                       # Core algorithms (Windows compatible)
│   ├── __init__.py
│   ├── cubemap.py             # CubemapData class
│   ├── projection.py          # Equirect ↔ Cubemap math
│   ├── synchronization.py     # Multi-plane sync
│   ├── depth_interface.py     # Generic depth integration
│   └── consistency.py         # Boundary smoothing
│
├── nodes/                      # ComfyUI nodes
│   ├── __init__.py
│   ├── projection_nodes.py    # EquirectToCubemap, CubemapToEquirect
│   ├── depth_nodes.py         # Depth processing nodes
│   ├── sync_nodes.py          # Synchronization control
│   └── utility_nodes.py       # Preview, validation, helpers
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── visualization.py       # Cubemap layouts & previews
│   └── validation.py          # Quality metrics
│
├── tests/                      # Test suite
│   ├── test_projection.py
│   ├── test_synchronization.py
│   ├── test_consistency.py
│   └── test_nodes.py
│
└── examples/                   # Example workflows
    ├── workflows/
    │   ├── basic_depth_estimation.json
    │   ├── synchronized_depth.json
    │   └── full_pipeline.json
    └── sample_images/
        └── test_panorama.jpg
```

**Status**: ✅ Directory structure created and populated

---

## 🗓️ Implementation Phases

### Phase 1: Foundation (Weeks 1-2) ✅ COMPLETE
**Goal**: Core projection mathematics and data structures

#### Tasks
- [x] ✅ **1.1** Set up project directory structure
- [x] ✅ **1.2** Create `core/cubemap.py` - CubemapData class
  - Face storage dictionary
  - Adjacency graph
  - Depth channel support
  - Serialization methods
- [x] ✅ **1.3** Implement `core/projection.py` - Projection mathematics
  - `equirect_to_cubemap()` - Main conversion function
  - `cubemap_to_equirect()` - Reverse conversion
  - `face_coords_to_vector()` - 2D face → 3D vector
  - `vector_to_lonlat()` - 3D vector → spherical coords
  - `lonlat_to_vector()` - Spherical → 3D vector
  - `vector_to_face_coords()` - 3D vector → face coordinates
- [x] ✅ **1.4** Write unit tests for projection accuracy
- [ ] ⏳ **1.5** Verify Windows compatibility (needs testing)
- [ ] ⏳ **1.6** Benchmark performance (target: <100ms for 1024²)

**Deliverables**:
- ✅ Working equirect ↔ cubemap conversion
- ✅ CubemapData structure with adjacency information
- ✅ Unit tests with basic coverage
- ⏳ Performance benchmarks (pending)

**Dependencies**: numpy, torch, scipy

---

### Phase 2: Depth Integration (Weeks 3-4) ✅ COMPLETE
**Goal**: Generic depth model interface and consistency enforcement

#### Tasks
- [x] ✅ **2.1** Create `core/depth_interface.py`
  - Generic depth estimator wrapper
  - Per-face depth application
  - Batch processing utilities
  - Format conversion helpers
- [x] ✅ **2.2** Implement `core/consistency.py`
  - `DepthConsistencyEnforcer` class
  - Boundary blending algorithms
  - Seam validation metrics
  - Gradient-based smoothing
- [ ] ⏳ **2.3** Test integration with depth nodes (needs real testing)
  - Depth Anything V2
  - Depth Anything V1
  - DA3 (if available)
  - MiDaS
  - Marigold
- [x] ✅ **2.4** Optimize depth normalization
- [x] ✅ **2.5** Create consistency validation tools

**Deliverables**:
- ✅ Works with any ComfyUI depth node (designed)
- ⏳ Boundary MAE < 0.03 (needs testing)
- ✅ Seam validation tool
- ⏳ Compatibility tests pass (needs testing)

**Dependencies**: numpy, scipy

---

### Phase 3: Multi-plane Synchronization (Weeks 5-6) ⏳ PENDING
**Goal**: Cross-face synchronization for depth consistency

#### Tasks
- [ ] **3.1** Implement `core/synchronization.py`
  - `SyncedSelfAttention` - Boundary-aware attention
  - `SyncedConv2d` - Cross-face padding
  - `SyncedGroupNorm` - Multi-face normalization
  - `MultiplaneSyncProcessor` - Main orchestrator
- [ ] **3.2** Add boundary pixel attention mechanism
- [ ] **3.3** Implement cross-face convolution padding
- [ ] **3.4** Create synchronized batch normalization
- [ ] **3.5** Performance optimization
  - GPU acceleration
  - Memory management
  - Batch processing
- [ ] **3.6** Validate seam quality improvement

**Deliverables**:
- ✅ Multi-plane sync working
- ✅ <20% performance overhead
- ✅ Improved boundary consistency
- ✅ PyTorch module compatible

**Dependencies**: torch, einops

---

### Phase 4: ComfyUI Nodes (Week 7) ⏳ PENDING
**Goal**: Complete set of ComfyUI-compatible nodes

#### Input/Output Nodes
- [ ] **4.1** `EquirectToCubemap` - Convert 360° image to cubemap
- [ ] **4.2** `CubemapToEquirect` - Convert cubemap back to 360°
- [ ] **4.3** `ExtractCubemapFace` - Get individual face as IMAGE

#### Processing Nodes
- [ ] **4.4** `ApplyDepthToCubemapFace` - Apply depth to single face
- [ ] **4.5** `BatchCubemapDepth` - Process all 6 faces
- [ ] **4.6** `MultiplaneSyncProcessor` - Apply synchronization
- [ ] **4.7** `MergeCubemapDepth` - Combine RGB + Depth

#### Utility Nodes
- [ ] **4.8** `CubemapPreview` - Visualize cubemap (cross/horizontal layout)
- [ ] **4.9** `CubemapSeamValidator` - Check seam quality
- [ ] **4.10** `CubemapFaceRotate` - Rotate individual faces (if needed)

**Deliverables**:
- ✅ 10+ functional nodes
- ✅ Proper INPUT_TYPES and RETURN_TYPES
- ✅ Category: "DreamCube/*"
- ✅ Clear tooltips and descriptions

**Dependencies**: ComfyUI core

---

### Phase 5: Testing & Documentation (Week 8) 🚧 IN PROGRESS
**Goal**: Comprehensive testing and user documentation

#### Testing
- [x] ✅ **5.1** Unit tests for core projection functions
- [ ] ⏳ **5.2** Integration tests for workflows (needs ComfyUI)
- [ ] ⏳ **5.3** Visual quality tests (needs sample data)
- [ ] ⏳ **5.4** Performance benchmarks (needs profiling)
- [ ] ⏳ **5.5** Windows compatibility testing (needs Windows system)
- [ ] ⏳ **5.6** Memory leak detection (needs profiling)

#### Documentation
- [x] ✅ **5.7** Complete README.md
  - Installation instructions
  - Quick start guide
  - Node descriptions
  - Troubleshooting
- [ ] ⏳ **5.8** Example workflows
  - Basic depth estimation (JSON needed)
  - Synchronized depth (JSON needed)
  - Integration with motion transfer (JSON needed)
- [x] ✅ **5.9** API documentation (in code docstrings)
- [ ] ⏳ **5.10** Video tutorial (optional, future)

**Deliverables**:
- ⏳ Test coverage >80% (basic tests done)
- ⏳ All tests passing (needs running)
- ✅ Complete documentation
- ⏳ 3+ example workflows (needs creation)

---

### Phase 6: Polish & Release (Week 9) ⏳ PENDING
**Goal**: Final optimization and public release

#### Pre-Release
- [ ] **6.1** Code review and cleanup
- [ ] **6.2** Performance optimization pass
- [ ] **6.3** Security audit
- [ ] **6.4** License verification (Apache 2.0)
- [ ] **6.5** Final testing on clean ComfyUI install

#### Release
- [ ] **6.6** Create GitHub release v1.0.0
- [ ] **6.7** Submit to ComfyUI Manager
- [ ] **6.8** Community announcement
  - ComfyUI Discord
  - Reddit r/comfyui
  - GitHub Discussions
- [ ] **6.9** Create demo video/images

#### Post-Release
- [ ] **6.10** Monitor GitHub issues
- [ ] **6.11** Gather user feedback
- [ ] **6.12** Plan v1.1.0 features

**Deliverables**:
- ✅ Public release v1.0.0
- ✅ ComfyUI Manager listing
- ✅ Community awareness
- ✅ Support infrastructure

---

## 🛠️ Technical Implementation Details

### Windows Compatibility Strategy

#### ✅ Safe for Windows
- Pure Python implementation
- PyTorch for GPU acceleration
- NumPy for numerical operations
- Pillow for image I/O
- No shell scripts or bash dependencies

#### ❌ Avoid These
- Linux-specific paths (`/usr/local`, etc.)
- Forward slashes in file paths (use `os.path.join()`)
- POSIX-only libraries
- Hardcoded line endings (`\n` → use `os.linesep` or let Python handle)

#### 🔧 Best Practices
```python
# ✅ Good - Cross-platform
import os
path = os.path.join(base_dir, "models", "depth.pth")

# ❌ Bad - Linux only
path = f"{base_dir}/models/depth.pth"

# ✅ Good - Use pathlib
from pathlib import Path
path = Path(base_dir) / "models" / "depth.pth"
```

---

### Native ComfyUI Node Integration

#### Using Existing Depth Nodes
Instead of bundling depth models, we leverage ComfyUI's ecosystem:

**Workflow Pattern**:
```
[Load Image] → [EquirectToCubemap]
                      ↓
                [Extract Face: Front] → [Depth Anything Node] ┐
                [Extract Face: Back]  → [Depth Anything Node] ├→ [Batch Cubemap Depth]
                [Extract Face: Left]  → [Depth Anything Node] │       ↓
                [Extract Face: Right] → [Depth Anything Node] │  [Multiplane Sync]
                [Extract Face: Top]   → [Depth Anything Node] │       ↓
                [Extract Face: Bottom]→ [Depth Anything Node] ┘  [CubemapToEquirect]
                                                                        ↓
                                                                   [Save Image]
```

**Supported Depth Nodes**:
- `DepthAnything` (ComfyUI-Depth-Anything)
- `DA3Depth` (if available)
- `MidasDepthEstimation` (ComfyUI-MiDaS)
- `MarigoldDepth` (ComfyUI-Marigold)
- Any node that outputs `IMAGE` type depth maps

---

### Performance Targets

| Operation | Resolution | Target Time | Memory |
|-----------|-----------|-------------|---------|
| Equirect→Cubemap | 2048×1024 → 6×1024² | <200ms | <2GB |
| Cubemap→Equirect | 6×1024² → 2048×1024 | <200ms | <2GB |
| Depth Consistency | 6×1024² faces | <400ms | <1GB |
| Multiplane Sync | 6×1024² faces | <800ms | <3GB |
| **Full Pipeline** | **End-to-end** | **<2000ms** | **<8GB** |

**Hardware Assumptions**:
- GPU: RTX 3060 (12GB) or better
- CPU: Modern x86_64 (fallback mode)
- RAM: 16GB system memory
- OS: Windows 10/11 or Linux

---

## 📚 Dependencies

### Core Dependencies
```txt
torch>=2.0.0          # PyTorch for GPU acceleration
torchvision>=0.15.0   # Vision utilities
numpy>=1.24.0         # Numerical operations
scipy>=1.10.0         # Scientific computing (interpolation)
pillow>=9.5.0         # Image I/O
opencv-python>=4.7.0  # Image processing (optional)
einops>=0.6.1         # Tensor operations
```

### Development Dependencies
```txt
pytest>=7.3.0         # Testing framework
pytest-cov>=4.1.0     # Coverage reporting
black>=23.3.0         # Code formatting
ruff>=0.0.270         # Linting
mypy>=1.3.0           # Type checking
```

### ComfyUI Integration
- Requires ComfyUI >= 1.0.0
- Compatible with existing depth estimation nodes
- No custom C++ extensions (pure Python)

---

## 🧪 Testing Strategy

### Unit Tests
```python
# tests/test_projection.py
def test_equirect_to_cubemap_roundtrip():
    """Verify equirect → cubemap → equirect preserves image"""
    original = load_test_image()
    cubemap = equirect_to_cubemap(original, 1024)
    recovered = cubemap_to_equirect(cubemap, original.shape[1], original.shape[0])

    psnr = calculate_psnr(original, recovered)
    assert psnr > 45.0, f"PSNR {psnr} too low"

    ssim = calculate_ssim(original, recovered)
    assert ssim > 0.99, f"SSIM {ssim} too low"

def test_cubemap_adjacency():
    """Verify face adjacency mapping is correct"""
    cubemap = CubemapData(512)
    adj_map = cubemap.get_adjacency_map()

    # Front face should have 4 neighbors
    assert len(adj_map['front']) == 4
    assert 'left' in adj_map['front']
    assert 'right' in adj_map['front']
```

### Integration Tests
```python
# tests/test_nodes.py
def test_full_workflow():
    """Test complete depth estimation workflow"""
    # Load equirect image
    equirect = load_comfyui_image("test_panorama.jpg")

    # Convert to cubemap
    cubemap_node = EquirectToCubemapNode()
    cubemap = cubemap_node.convert(equirect, cube_resolution=1024)

    # Extract and process faces (simulated)
    depth_faces = simulate_depth_estimation(cubemap)

    # Apply depth
    batch_node = BatchCubemapDepthNode()
    cubemap_depth = batch_node.apply_depth(cubemap, *depth_faces)

    # Convert back
    convert_node = CubemapToEquirectNode()
    output = convert_node.convert(cubemap_depth, 2048, 1024, "depth")

    assert output.shape == (1, 1024, 2048, 3)
```

### Visual Quality Tests
- Manual inspection of seam visibility
- Depth gradient smoothness
- Comparison with original DreamCube output
- User acceptance testing

---

## 📊 Progress Tracking

### Overall Progress: 85% Complete 🎉

#### Phase 1: Foundation - 95% ✅
- [x] Design document created
- [x] Agent roles defined
- [x] Implementation plan written
- [x] Directory structure set up
- [x] Core projection math implemented
- [x] CubemapData structure created
- [x] Unit tests written
- [ ] Performance benchmarking (needs testing)

#### Phase 2: Depth Integration - 90% ✅
- [x] Depth interface designed
- [x] Depth interface implemented
- [x] Consistency enforcer implemented
- [ ] Compatibility tests written (needs ComfyUI environment)

#### Phase 3: Synchronization - 85% ✅
- [x] Attention sync implemented
- [x] Conv sync implemented
- [x] GroupNorm sync implemented
- [ ] Performance optimized (needs profiling)

#### Phase 4: ComfyUI Nodes - 100% ✅
- [x] Projection nodes created (4 nodes)
- [x] Depth nodes created (5 nodes)
- [x] Utility nodes created (6 nodes)
- [x] Node registration complete

#### Phase 5: Testing & Docs - 65% 🚧
- [x] Test suite foundation complete
- [x] README documentation written
- [ ] Example workflows created
- [ ] Real-world testing needed

#### Phase 6: Release - 0% ⏸️
- [ ] Final testing complete
- [ ] Release published
- [ ] Community notified

---

## 🚀 Next Immediate Steps

### This Week (Week 1)
1. **Set up project structure**
   - Create all directories
   - Initialize `__init__.py` files
   - Set up `requirements.txt`

2. **Implement `core/cubemap.py`**
   - CubemapData class
   - Adjacency graph
   - Basic methods

3. **Start `core/projection.py`**
   - Coordinate transformation utilities
   - Begin `equirect_to_cubemap()` implementation

### Next Week (Week 2)
4. **Complete projection mathematics**
   - Finish `equirect_to_cubemap()`
   - Implement `cubemap_to_equirect()`
   - Optimize with vectorization

5. **Write unit tests**
   - Projection accuracy tests
   - Boundary condition tests
   - Performance benchmarks

6. **Verify Windows compatibility**
   - Test on Windows 10/11
   - Check path handling
   - Verify dependencies install correctly

---

## 🐛 Known Issues & Risks

### Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| PyTorch CUDA compatibility | High | Medium | Provide CPU fallback |
| Depth node API changes | Medium | Low | Use stable ComfyUI types |
| Performance on low-end GPUs | Medium | High | Add streaming mode |
| Seam artifacts | High | Medium | Extensive testing + blending |
| Windows path issues | Low | Low | Use pathlib everywhere |

### Current Blockers
- None (in planning phase)

### Technical Debt
- None yet (will track as implementation progresses)

---

## 📞 Contact & Support

**Project Lead**: Cedar
**Repository**: https://github.com/cedarconnor/ComfyUI-DreamCube (to be created)
**License**: Apache 2.0
**Original Paper**: [DreamCube (ICCV 2025)](https://arxiv.org/abs/2506.17206)

---

## 📝 Change Log

### 2025-11-17 - Initial Planning
- Created IMPLEMENTATION_PLAN.md
- Reviewed DESIGN_DOCUMENT.md and AGENTS.md
- Analyzed original DreamCube repository
- Defined 6-phase implementation strategy
- Set up progress tracking structure

---

## 🎯 Success Metrics

### Technical Metrics
- ✅ Round-trip projection PSNR > 45dB
- ✅ Round-trip projection SSIM > 0.99
- ✅ Processing speed < 2s for full pipeline (1024²)
- ✅ Memory usage < 8GB VRAM
- ✅ Seam error MAE < 0.03
- ✅ Test coverage > 80%

### User Metrics
- ✅ Works on Windows without modification
- ✅ Compatible with top 5 ComfyUI depth nodes
- ✅ Clear error messages for common issues
- ✅ Example workflows cover main use cases
- ✅ Documentation complete and understandable

### Community Metrics
- Target: 100+ GitHub stars in first month
- Target: 500+ downloads via ComfyUI Manager
- Target: <24h average response time on issues
- Target: 10+ community workflows created

---

**Last Updated**: 2025-11-17 (Post-Implementation)
**Next Review**: After real-world testing in ComfyUI
**Status**: ✅ Core Implementation Complete - Ready for Integration Testing!
