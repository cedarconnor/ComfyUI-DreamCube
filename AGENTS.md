# ComfyUI-DreamCube Implementation Specification

## Project Metadata
- **Project**: ComfyUI_DreamCube
- **Type**: ComfyUI Custom Node Pack
- **Domain**: 360° Panoramic Depth Estimation
- **Framework**: DreamCube Multi-plane Synchronization
- **Author**: Cedar
- **Target Platform**: ComfyUI >= 1.0.0
- **License**: Apache 2.0

---

## Agent Roles

### Agent 1: Projection Engineer
**Responsibility**: Equirectangular ↔ Cubemap conversion mathematics

**Tasks**:
1. Implement `equirect_to_cubemap()` with accurate coordinate mapping
2. Implement `cubemap_to_equirect()` with proper sampling
3. Create coordinate transformation utilities
4. Optimize projection speed with vectorization
5. Add bilinear/bicubic interpolation options

**Deliverables**:
- `core/projection.py` with tested conversion functions
- Performance: <100ms for 2048×1024 → 6×1024² on GPU
- Accuracy: <0.1% pixel error on round-trip conversion

**Dependencies**: numpy, torch, scipy

**Testing**: Round-trip accuracy, seam alignment, edge case handling

---

### Agent 2: Cubemap Data Structure Engineer
**Responsibility**: Core cubemap data representation

**Tasks**:
1. Design `CubemapData` class with face storage
2. Implement adjacency graph and edge mapping
3. Add serialization/deserialization for caching
4. Create copy/clone methods for immutability
5. Implement face extraction and insertion APIs

**Deliverables**:
- `core/cubemap.py` with complete data structure
- Memory-efficient storage (share arrays where possible)
- Support for RGB, RGBD, and custom channels

**Dependencies**: numpy, dataclasses

**Testing**: Memory usage, access patterns, serialization

---

### Agent 3: Multi-plane Synchronization Engineer
**Responsibility**: Cross-face consistency algorithms

**Tasks**:
1. Implement `SyncedSelfAttention` with boundary awareness
2. Implement `SyncedConv2d` with cross-face padding
3. Implement `SyncedGroupNorm` across all faces
4. Create `MultiplaneSyncProcessor` orchestrator
5. Add configurable sync modes (attention-only, full, etc.)

**Deliverables**:
- `core/synchronization.py` with sync components
- PyTorch modules compatible with existing models
- Minimal overhead (<20% vs. unsynchronized)

**Dependencies**: torch, einops

**Testing**: Boundary continuity, computational overhead, gradient flow

---

### Agent 4: Depth Integration Engineer
**Responsibility**: Generic depth model interface

**Tasks**:
1. Create `DepthModelInterface` for any depth estimator
2. Implement per-face depth application logic
3. Build batch processing utilities
4. Add depth normalization and scaling
5. Create adapter for Depth Anything, DA3, MiDaS

**Deliverables**:
- `core/depth_interface.py` with flexible interface
- Support for any ComfyUI depth node output
- Automatic format detection and conversion

**Dependencies**: torch, numpy

**Testing**: Compatibility with multiple depth models

---

### Agent 5: Consistency Enforcement Engineer
**Responsibility**: Boundary depth consistency

**Tasks**:
1. Implement `DepthConsistencyEnforcer` with blending
2. Create seam validation metrics
3. Add iterative refinement options
4. Implement gradient-based smoothing
5. Build visualization tools for seam errors

**Deliverables**:
- `core/consistency.py` with enforcement algorithms
- Configurable blending width (8-32 pixels)
- Error metrics: MAE, RMSE at boundaries

**Dependencies**: numpy, scipy

**Testing**: Visual seam quality, quantitative error metrics

---

### Agent 6: ComfyUI Node Developer
**Responsibility**: ComfyUI-compatible node implementations

**Tasks**:
1. Create projection nodes (Equirect↔Cubemap)
2. Create depth processing nodes
3. Create synchronization control nodes
4. Create utility nodes (preview, validation)
5. Implement proper type handling and error messages

**Deliverables**:
- `nodes/*.py` with all node classes
- Proper INPUT_TYPES and RETURN_TYPES
- Category organization: "DreamCube/Projection", "DreamCube/Depth", etc.

**Dependencies**: ComfyUI core, torch

**Testing**: Node registration, input validation, workflow execution

---

### Agent 7: Performance Optimization Engineer
**Responsibility**: Speed and memory optimization

**Tasks**:
1. Profile bottlenecks in projection and sync
2. Implement CUDA kernels for critical paths
3. Add batching and streaming support
4. Optimize memory allocations
5. Add resolution-adaptive processing

**Deliverables**:
- Optimized `core/*` functions
- 2x+ speedup over naive implementation
- Memory usage <8GB for 1024² faces

**Dependencies**: torch, cuda (optional)

**Testing**: Benchmarking, profiling, memory tracking

---

### Agent 8: Testing & Quality Assurance
**Responsibility**: Comprehensive testing suite

**Tasks**:
1. Write unit tests for all core functions
2. Create integration tests for full workflows
3. Implement visual quality tests
4. Add regression test suite
5. Create test fixtures and sample data

**Deliverables**:
- `tests/*` with >80% code coverage
- Automated CI/CD pipeline
- Visual comparison tools

**Dependencies**: pytest, pytest-cov

**Testing**: Test coverage, CI green build

---

### Agent 9: Documentation & Examples
**Responsibility**: User-facing documentation

**Tasks**:
1. Write comprehensive README.md
2. Create node usage documentation
3. Build example workflows (JSON)
4. Record video tutorials
5. Write troubleshooting guide

**Deliverables**:
- Documentation in `docs/`
- Example workflows in `examples/workflows/`
- Video guides on YouTube/GitHub

**Dependencies**: markdown, json

**Testing**: Documentation completeness, workflow functionality

---

### Agent 10: Integration & Release Manager
**Responsibility**: Packaging and deployment

**Tasks**:
1. Create `__init__.py` with node registration
2. Build ComfyUI Manager compatibility
3. Create installation scripts
4. Write changelog and versioning
5. Coordinate GitHub release

**Deliverables**:
- Installable package
- ComfyUI Manager listing
- GitHub release with binaries

**Dependencies**: git, github-cli

**Testing**: Clean install on fresh ComfyUI

---

## Development Workflow

### Sprint 1: Foundation (Weeks 1-2)
**Agents Active**: 1, 2, 8
**Goals**: 
- Working projection math
- CubemapData structure
- Basic tests passing

**Milestones**:
- ✅ Equirect → Cubemap conversion
- ✅ Cubemap → Equirect conversion
- ✅ Round-trip accuracy <0.1%

---

### Sprint 2: Depth Integration (Weeks 3-4)
**Agents Active**: 4, 5, 8
**Goals**:
- Generic depth interface
- Consistency enforcement
- Integration tests

**Milestones**:
- ✅ Works with Depth Anything V2
- ✅ Works with DA3
- ✅ Seam validation <0.05 max error

---

### Sprint 3: Synchronization (Weeks 5-6)
**Agents Active**: 3, 7, 8
**Goals**:
- Multi-plane sync implementation
- Performance optimization
- Boundary continuity

**Milestones**:
- ✅ Attention sync working
- ✅ Conv sync working
- ✅ <20% overhead vs. baseline

---

### Sprint 4: ComfyUI Integration (Week 7)
**Agents Active**: 6, 9, 10
**Goals**:
- All nodes implemented
- Example workflows created
- Documentation complete

**Milestones**:
- ✅ 15+ nodes implemented
- ✅ 5+ example workflows
- ✅ Full README and docs

---

### Sprint 5: Polish & Release (Weeks 8-9)
**Agents Active**: 8, 9, 10
**Goals**:
- Final testing
- Performance tuning
- Community release

**Milestones**:
- ✅ All tests passing
- ✅ ComfyUI Manager listed
- ✅ GitHub release v1.0.0

---

## Technical Specifications

### Coordinate Systems

**Equirectangular**:
- Range: θ ∈ [0, 2π], φ ∈ [-π/2, π/2]
- Origin: Top-left corner
- X-axis: Longitude (left to right)
- Y-axis: Latitude (top to bottom)

**Cubemap**:
- 6 faces: front, back, left, right, top, bottom
- Each face: N×N pixels
- Coordinate system: Right-handed, Z-forward
- UV range: [-1, 1] × [-1, 1] per face

### Face Adjacency Mapping

```
Top View:
    [top]
[left][front][right][back]
   [bottom]

Adjacency Rules:
- Front: {top: bottom, bottom: top, left: right, right: left}
- Back:  {top: top, bottom: bottom, left: left, right: right}
- Left:  {front: right, back: right, top: left, bottom: left}
- Right: {front: left, back: left, top: right, bottom: right}
- Top:   {front: top, back: bottom, left: top, right: top}
- Bottom:{front: bottom, back: top, left: bottom, right: bottom}
```

### Data Flow Diagram

```
Input (Equirect)
       ↓
┌──────────────────┐
│ Equirect2Cubemap │
└──────────────────┘
       ↓
   [CUBEMAP]
       ↓
┌──────────────────┐
│ Extract 6 Faces  │
└──────────────────┘
       ↓
  [6× IMAGE]
       ↓
┌──────────────────┐
│ Depth Model × 6  │ ← External (DA3, Depth Anything, etc.)
└──────────────────┘
       ↓
  [6× DEPTH]
       ↓
┌──────────────────┐
│BatchCubemapDepth │
└──────────────────┘
       ↓
[CUBEMAP + Depth]
       ↓
┌──────────────────┐
│ Multiplane Sync  │ ← Optional
└──────────────────┘
       ↓
┌──────────────────┐
│ Consistency      │
└──────────────────┘
       ↓
┌──────────────────┐
│ Cubemap2Equirect │
└──────────────────┘
       ↓
Output (Equirect Depth)
```

---

## API Specifications

### Core API

```python
# Projection
def equirect_to_cubemap(
    image: np.ndarray,      # [H, W, C]
    face_size: int,         # Resolution per face
    interpolation: str = 'bilinear'
) -> CubemapData

def cubemap_to_equirect(
    cubemap: CubemapData,
    width: int,             # Output width
    height: int,            # Output height
    interpolation: str = 'bilinear'
) -> np.ndarray            # [H, W, C]

# Depth Interface
def apply_depth_per_face(
    cubemap: CubemapData,
    face_name: str,
    depth_map: np.ndarray
) -> CubemapData

def enforce_depth_consistency(
    cubemap: CubemapData,
    boundary_width: int = 16,
    method: str = 'blend'   # 'blend', 'gradient', 'poisson'
) -> CubemapData

# Synchronization
def apply_multiplane_sync(
    face_features: dict,    # {face_name: tensor}
    sync_config: dict       # {attn: bool, conv: bool, gn: bool}
) -> dict

# Validation
def validate_cubemap_seams(
    cubemap: CubemapData,
    threshold: float = 0.05
) -> tuple[bool, float]    # (is_valid, max_error)
```

### Node API

```python
class NodeBase:
    @classmethod
    def INPUT_TYPES(cls) -> dict
    
    @property
    def RETURN_TYPES(self) -> tuple
    
    @property
    def FUNCTION(self) -> str
    
    @property
    def CATEGORY(self) -> str
    
    def execute(self, **kwargs) -> tuple
```

---

## Error Handling

### Common Errors

1. **Invalid Resolution**: Cubemap resolution not divisible by 2
2. **Mismatched Faces**: Depth map size doesn't match face size
3. **Missing Adjacency**: Face not in adjacency map
4. **Seam Discontinuity**: Depth error >threshold at boundaries
5. **OOM**: Insufficient VRAM for resolution

### Error Messages

```python
errors = {
    'INVALID_RESOLUTION': 
        "Cubemap resolution must be power of 2 (256, 512, 1024, 2048, 4096)",
    'MISMATCHED_SIZE': 
        "Depth map size {actual} doesn't match face size {expected}",
    'SEAM_ERROR': 
        "Seam discontinuity {error:.3f} exceeds threshold {threshold:.3f}",
    'OOM': 
        "Out of memory. Try reducing cube_resolution or enabling streaming mode"
}
```

---

## Performance Targets

| Operation | Target Time (1024²) | Target Time (2048²) |
|-----------|--------------------|--------------------|
| Equirect→Cubemap | <50ms | <200ms |
| Cubemap→Equirect | <50ms | <200ms |
| Depth Consistency | <100ms | <400ms |
| Multiplane Sync | <200ms | <800ms |
| Full Pipeline | <500ms | <2000ms |

**Memory Targets**:
- 1024²: <4GB VRAM
- 2048²: <12GB VRAM
- 4096²: <40GB VRAM

---

## Quality Metrics

### Projection Quality
- Round-trip PSNR: >45 dB
- Round-trip SSIM: >0.99
- Seam alignment error: <0.5 pixels

### Depth Quality
- Boundary MAE: <0.03 (normalized depth)
- Boundary RMSE: <0.05
- Cross-face correlation: >0.95

### Visual Quality
- No visible seams in preview
- Smooth depth gradients
- Consistent lighting across faces

---

## Compatibility Matrix

| Depth Model | Tested | Working | Notes |
|-------------|--------|---------|-------|
| Depth Anything V2 | ✅ | ✅ | Preferred |
| Depth Anything V1 | ✅ | ✅ | Good |
| DA3 | ✅ | ✅ | Best quality |
| MiDaS | ✅ | ⚠️ | Requires normalization |
| Marigold | ✅ | ✅ | Excellent |
| ZoeDepth | ✅ | ✅ | Good |
| Custom Models | 🔄 | 🔄 | Should work |

---

## Installation Instructions

### Method 1: ComfyUI Manager
```bash
# In ComfyUI Manager
Search: "DreamCube"
Click: Install
Restart: ComfyUI
```

### Method 2: Manual
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Cedar/ComfyUI_DreamCube.git
cd ComfyUI_DreamCube
pip install -r requirements.txt
# Restart ComfyUI
```

### Method 3: Development
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Cedar/ComfyUI_DreamCube.git
cd ComfyUI_DreamCube
pip install -e .  # Editable install
# Make changes, no restart needed for code changes
```

---

## Configuration Files

### `config.yaml`
```yaml
projection:
  interpolation: bilinear  # bilinear, bicubic, lanczos
  default_resolution: 1024
  max_resolution: 4096

depth:
  normalization: global    # global, per_face, adaptive
  consistency_width: 16
  consistency_method: blend

sync:
  enabled: true
  attention: true
  convolution: true
  group_norm: true

performance:
  use_cuda: true
  batch_faces: true
  cache_projections: true
  streaming_mode: false    # Enable for low VRAM
```

---

## Git Workflow

### Branch Strategy
- `main`: Stable releases only
- `develop`: Active development
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `release/*`: Release preparation

### Commit Convention
```
type(scope): description

Types: feat, fix, docs, style, refactor, test, chore
Scope: projection, depth, sync, nodes, core
```

Examples:
```
feat(projection): add bicubic interpolation
fix(depth): resolve boundary blending artifact
docs(readme): add installation instructions
```

---

## License & Attribution

### Primary License
Apache License 2.0

### Attribution Requirements
```
This project builds upon DreamCube (ICCV 2025)
by Yukun Huang, Yanning Zhou, Jianan Wang, Kaiyi Huang, Xihui Liu

DreamCube paper: https://arxiv.org/abs/2506.17206
DreamCube code: https://github.com/Yukun-Huang/DreamCube

Implemented for ComfyUI by Cedar
Repository: https://github.com/Cedar/ComfyUI_DreamCube
```

---

## Support & Community

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Discord**: ComfyUI Discord #custom-nodes
- **Email**: cedar@example.com

---

## Changelog

### v1.0.0 (Target: Q1 2025)
- ✅ Initial release
- ✅ Equirect ↔ Cubemap conversion
- ✅ Generic depth interface
- ✅ Multi-plane synchronization
- ✅ Boundary consistency
- ✅ 15+ nodes
- ✅ Example workflows
- ✅ Full documentation

### v1.1.0 (Future)
- 🔄 Video support with temporal consistency
- 🔄 Normal map generation
- 🔄 3D mesh export
- 🔄 LoRA integration

---

## Success Criteria

### Functional Requirements
- ✅ Converts 360° equirect to cubemap accurately
- ✅ Works with any ComfyUI depth node
- ✅ Enforces boundary consistency
- ✅ Converts back to equirect seamlessly

### Non-Functional Requirements
- ✅ <500ms full pipeline (1024²)
- ✅ <8GB VRAM for standard use
- ✅ >80% code coverage
- ✅ Zero visible seams

### User Experience
- ✅ Intuitive node names
- ✅ Clear error messages
- ✅ Example workflows included
- ✅ Complete documentation

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CUDA compatibility | Medium | High | Fallback to CPU |
| Depth model incompatibility | Low | Medium | Generic interface |
| Performance issues | Medium | Medium | Profiling, optimization |
| Seam artifacts | Low | High | Thorough testing |
| Memory overflow | Medium | High | Streaming mode |

---

## Task Allocation Matrix

| Sprint | Agent 1 | Agent 2 | Agent 3 | Agent 4 | Agent 5 | Agent 6 | Agent 7 | Agent 8 | Agent 9 | Agent 10 |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|
| 1      | 🟢 | 🟢 | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | 🟡 | ⚪ | ⚪ |
| 2      | ⚪ | ⚪ | ⚪ | 🟢 | 🟢 | ⚪ | ⚪ | 🟡 | ⚪ | ⚪ |
| 3      | ⚪ | ⚪ | 🟢 | ⚪ | ⚪ | ⚪ | 🟢 | 🟡 | ⚪ | ⚪ |
| 4      | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | 🟢 | ⚪ | 🟡 | 🟢 | 🟡 |
| 5      | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | 🟢 | 🟢 | 🟢 |

Legend: 🟢 Primary work | 🟡 Support/Review | ⚪ Idle

---

## Communication Protocol

### Daily Standups
- **Time**: 9:00 AM (async via Discord/Slack)
- **Format**: What did you complete? What are you working on? Any blockers?
- **Duration**: 15 minutes max

### Sprint Planning
- **Frequency**: Start of each sprint (bi-weekly)
- **Participants**: All agents + project lead
- **Outcome**: Sprint goals, task assignments, dependencies

### Code Reviews
- **Requirement**: All PRs require 1 approval
- **Reviewers**: Agent 8 (QA) + relevant domain agent
- **Turnaround**: 24 hours max

### Integration Testing
- **Frequency**: End of each week
- **Owner**: Agent 8
- **Participants**: All agents with completed features

---

## Development Environment Setup

### Local Development
```bash
# Clone repository
git clone https://github.com/Cedar/ComfyUI_DreamCube.git
cd ComfyUI_DreamCube

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Testing, linting tools

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=core --cov=nodes
```

### Docker Development
```bash
# Build development container
docker build -t comfyui-dreamcube:dev -f Dockerfile.dev .

# Run with GPU support
docker run --gpus all -it -v $(pwd):/workspace comfyui-dreamcube:dev

# Inside container
pytest tests/
```

---

## Code Quality Standards

### Python Style
- **Formatter**: Black (line length: 100)
- **Linter**: Ruff
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style

### Testing Requirements
- **Unit Tests**: All core functions
- **Integration Tests**: All node workflows
- **Coverage**: Minimum 80%
- **Performance Tests**: Benchmarks for critical paths

### Documentation
- **Code Comments**: Explain "why", not "what"
- **README**: Keep updated with features
- **CHANGELOG**: Follow Keep a Changelog format
- **API Docs**: Auto-generated from docstrings

---

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      - name: Run tests
        run: pytest tests/ --cov=core --cov=nodes --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - name: Run linters
        run: |
          ruff check .
          black --check .
```

---

## Deployment Checklist

### Pre-Release
- [ ] All tests passing
- [ ] Code coverage >80%
- [ ] Documentation complete
- [ ] Example workflows tested
- [ ] Performance benchmarks met
- [ ] Security audit passed

### Release
- [ ] Version bumped in `__init__.py`
- [ ] CHANGELOG.md updated
- [ ] Git tag created (v1.0.0)
- [ ] GitHub release published
- [ ] ComfyUI Manager PR submitted
- [ ] Community announcement posted

### Post-Release
- [ ] Monitor GitHub issues
- [ ] Respond to community feedback
- [ ] Plan v1.1.0 features
- [ ] Update documentation based on FAQs

---

## Key Performance Indicators (KPIs)

### Development Metrics
- **Velocity**: Story points completed per sprint
- **Code Quality**: Ruff score, test coverage %
- **Bug Rate**: Bugs per 1000 lines of code
- **Review Time**: Average PR review turnaround

### Product Metrics
- **Adoption**: GitHub stars, downloads
- **Engagement**: Issues opened, PRs submitted
- **Performance**: Benchmark results vs. targets
- **Satisfaction**: User feedback, ratings

---

## Escalation Path

### Issue Severity Levels

**P0 - Critical**
- System crash or data loss
- No workaround available
- Response: Immediate (within 4 hours)

**P1 - High**
- Major feature broken
- Workaround exists
- Response: Within 24 hours

**P2 - Medium**
- Minor feature issue
- Easy workaround
- Response: Within 1 week

**P3 - Low**
- Enhancement request
- Cosmetic issue
- Response: Next sprint

### Escalation Contacts
1. **Technical Issues**: Agent 8 (QA Lead)
2. **Integration Issues**: Agent 10 (Release Manager)
3. **Performance Issues**: Agent 7 (Optimization Engineer)
4. **Community Issues**: Agent 9 (Documentation Lead)

---

## Next Steps

1. **Week 1**: Agents 1, 2 start projection implementation
2. **Week 2**: Agent 8 creates test framework
3. **Week 3**: Agents 4, 5 implement depth interface
4. **Week 4**: Integration testing begins
5. **Week 5**: Agent 3 implements synchronization
6. **Week 6**: Agent 7 optimizes performance
7. **Week 7**: Agent 6 creates all nodes
8. **Week 8**: Agents 9, 10 finalize docs and release
9. **Week 9**: Public release and community support

---

**Status**: Ready for implementation  
**Last Updated**: 2025-01-XX  
**Next Review**: After Sprint 1 completion  
**Project Lead**: Cedar  
**Contact**: cedar@example.com
