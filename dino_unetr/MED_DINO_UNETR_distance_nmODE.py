"""Compatibility imports for the former model module.

Use :mod:`dino_unetr.dosedino` for the paper-aligned names.
These imports use the paper CDD implementation, not the historical solver.
"""

from .dosedino import (
    CDDDynamics as nmODEFunc,
    DoseDINO as MED_DINO_UNETR_Distance_nmODE,
    DINOv3Encoder as MedDINOv3Backbone,
    DirectedAttention as CrossAttentionFusion,
    FixedStepCDD as nmODEBlock,
    PGIEncoder as GeometryEncoder,
)

__all__ = [
    "MED_DINO_UNETR_Distance_nmODE",
    "MedDINOv3Backbone",
    "GeometryEncoder",
    "CrossAttentionFusion",
    "nmODEFunc",
    "nmODEBlock",
]
