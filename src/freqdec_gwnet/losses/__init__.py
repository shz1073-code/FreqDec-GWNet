"""Loss functions for FreqDec-GWNet."""

from .motion_losses import MotionLoss, dilate_binary

__all__ = ["MotionLoss", "dilate_binary"]
