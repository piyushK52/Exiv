"""Diffusion-related handler helpers."""

from typing import Any, Dict, Optional

import torch
class DiffusionMixin:
    """Mixin containing diffusion execution helpers.

    Required host attributes:
    - ``device``: torch device string used for output tensor placement.
    - ``dtype``: torch dtype used for output tensor conversion.
    """
