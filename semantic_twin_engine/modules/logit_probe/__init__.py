"""
Logit Probe module for the Semantic Twin Engine.

This module analyzes token probabilities (logprobs) to probe the
"subconscious" associations the LLM makes with the entity.
"""

from modules.logit_probe.logic import LogitProbe

__all__ = ["LogitProbe"]
