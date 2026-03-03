"""Gymnasium wrappers for Robotron 2084 environment."""

from .multi_discrete_wrapper import MultiDiscreteToDiscrete
from .frame_skip_wrapper import FrameSkipWrapper

__all__ = ["MultiDiscreteToDiscrete", "FrameSkipWrapper"]
