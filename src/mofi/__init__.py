"""MOFI: Multi-Omics Fate Inference

A computational framework that reconstructs coordinated multi-omics
developmental trajectories from discrete paired snapshots.

This package is a thin wrapper around the CytoBridge engine.
Users can import via ``import mofi`` or ``import CytoBridge``.
"""

from CytoBridge import pp, tl, pl, utils, Map  # noqa: F401

__all__ = ["pp", "tl", "pl", "utils", "Map"]
__version__ = "0.1.0"
