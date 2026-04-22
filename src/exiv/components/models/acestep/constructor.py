import os
from pathlib import Path
from typing import Optional, Tuple

import torch

from ....utils.file import ensure_model_availability
from ....utils.file_path import FilePaths
from ....utils.logging import app_logger


def _resolve_checkpoint_path(config_name: str) -> str:
    """Resolve a checkpoint path, using local checkpoints dir first."""
    from .model_downloader import get_checkpoints_dir
    checkpoints_dir = get_checkpoints_dir()
    local_path = checkpoints_dir / config_name
    if local_path.exists():
        return str(local_path)

    registry_name = config_name.replace("-", "_") + ".safetensors"
    try:
        path_data = FilePaths.get_path(registry_name, file_type="checkpoint")
        if path_data.path and os.path.exists(path_data.path):
            return str(Path(path_data.path).parent)
    except Exception:
        pass

    return str(local_path)


def get_acestep_dit_instance(
    model_path: Optional[str] = None,
    force_load_mode: Optional[str] = None,
    force_dtype: Optional[torch.dtype] = None,
) -> Tuple[object, str]:
    """Initialize an ACE-Step DiT handler."""
    from .handler import AceStepHandler

    if model_path is None:
        model_path = "acestep-v15-sft"

    checkpoint_path = _resolve_checkpoint_path(model_path)
    app_logger.info(f"ACE-Step DiT checkpoint resolved to: {checkpoint_path}")

    handler = AceStepHandler()
    dtype = force_dtype or torch.bfloat16
    status, ok = handler.initialize_service(
        config_path=model_path,
        device="cuda",
        use_flash_attention=True,
    )

    if not ok:
        raise RuntimeError(f"Failed to initialize ACE-Step DiT: {status}")

    return handler, status


def get_acestep_lm_instance(
    model_path: Optional[str] = None,
    force_load_mode: Optional[str] = None,
    force_dtype: Optional[torch.dtype] = None,
) -> Tuple[object, str]:
    """Initialize an ACE-Step 5Hz LM handler."""
    from .llm_inference import LLMHandler
    from .model_downloader import get_checkpoints_dir

    if model_path is None:
        model_path = "acestep-5Hz-lm-1.7B"

    checkpoint_dir = str(get_checkpoints_dir())
    app_logger.info(f"ACE-Step LM checkpoint resolved to: {checkpoint_dir}/{model_path}")

    handler = LLMHandler()
    dtype = force_dtype or torch.bfloat16
    status, ok = handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=model_path,
        backend="pt",
        device="cuda",
    )

    if not ok:
        raise RuntimeError(f"Failed to initialize ACE-Step LM: {status}")

    return handler, status
