"""
Modules package for the Semantic Twin Engine.

This package contains all pluggable probe modules. Each probe module
must implement the ProbeInterface contract defined in core.interfaces.

Available modules:
- vector_probe: Embeddings & Geometry analysis
- logit_probe: Logprobs & Subconscious analysis
"""

from modules.base_probe import BaseProbe

__all__ = ["BaseProbe"]
