"""Package containing robot assets, actuators and terrain helpers for Mongo0909.

This package plays the same role as ``agent_world`` in the wheeled-legged_RL
reference project: it owns the Isaac Lab side of the world (robot descriptions,
USD paths, ...) and knows nothing about reinforcement learning.
"""

import os

# Parent path of the package: ``<repo>/source/mongo_world/mongo_world``
RootPath = os.path.dirname(os.path.abspath(__file__))
# Folder that contains the importable assets (``<...>/mongo_world/assets``)
AssetPath = os.path.join(RootPath, "assets")

# Repository root: ``<...>/Mongo_RL``
REPO_ROOT = os.path.abspath(os.path.join(RootPath, "..", "..", ".."))

__all__ = ["RootPath", "AssetPath", "REPO_ROOT"]
