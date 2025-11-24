"""
ComfyUI-DreamCube - 360-degree Panoramic Depth Estimation with Multi-plane Synchronization

A ComfyUI custom node pack implementing DreamCube's multi-plane synchronization
framework for consistent depth estimation on 360-degree panoramic images.

Author: Cedar
License: Apache 2.0
Based on: DreamCube (ICCV 2025) by Yukun Huang et al.
"""

import sys
import traceback
from pathlib import Path

__version__ = "1.0.0"
__author__ = "Cedar"
__license__ = "Apache 2.0"


_DEBUG_LOG = Path(__file__).with_name("dreamcube_import.log")


def _log_debug(message: str):
    """
    Lightweight logger that writes to both stdout and a local log file.
    File writes are best-effort and will never raise.
    """
    line = f"[ComfyUI-DreamCube][DEBUG] {message}"
    try:
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Ignore file write failures
        pass
    try:
        print(line)
    except Exception:
        # Ignore console encoding issues
        pass


def _debug_dependency_status():
    """Print minimal dependency info to help diagnose import issues."""
    deps = ("torch", "numpy", "scipy", "einops")
    for name in deps:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "unknown")
            _log_debug(f"{name} available (version {version})")
        except Exception as exc:  # pragma: no cover - debugging helper
            _log_debug(f"{name} import failed: {exc}")


try:
    _log_debug(f"Import start; file={Path(__file__).resolve()}")
    _log_debug(f"sys.path={sys.path}")
    from .nodes import projection_nodes, depth_nodes, utility_nodes
except Exception as exc:
    _log_debug("Failed to import nodes package")
    _log_debug(f"Exception: {exc}")
    _log_debug(f"Module file: {Path(__file__).resolve()}")
    _log_debug(f"sys.path: {sys.path}")
    _debug_dependency_status()
    try:
        traceback.print_exc()
    except Exception:
        pass
    try:
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            traceback.print_exc(file=f)
    except Exception:
        pass
    # Re-raise so ComfyUI surfaces the failure
    raise

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

# Use ASCII-only logging to avoid Windows console encoding errors
_log_debug(f"v{__version__} loaded successfully")
_log_debug(f"{len(NODE_CLASS_MAPPINGS)} nodes registered")
_log_debug(f"Loaded from {Path(__file__).resolve()}")
