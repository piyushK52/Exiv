"""ACE-Step constructor — follows the qwen3_tts constructor pattern."""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .model import AceStepDiT
from .lm import AceStepLM
from .pipeline import AceStepPipeline
from .utils import ensure_acestep_checkpoint
from .core.configuration_acestep_v15 import AceStepConfig
from ....utils.file_path import FilePaths
from ....utils.logging import app_logger


# Mapping from checkpoint directory name → (variant_tag, model_enum)
_DIT_VARIANTS = {
    "acestep-v15-base": "base",
    "acestep-v15-sft": "sft",
    "acestep-v15-turbo": "turbo",
    "acestep-v15-xl-base": "xl_base",
    "acestep-v15-xl-sft": "xl_sft",
    "acestep-v15-xl-turbo": "xl_turbo",
}


def _detect_dit_variant(checkpoint_dir: Path) -> str:
    """Infer DiT variant from directory name or config.json."""
    name = checkpoint_dir.name
    if name in _DIT_VARIANTS:
        return _DIT_VARIANTS[name]

    # Fallback: read config.json is_turbo / hidden_size heuristics
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if cfg.get("is_turbo"):
            return "turbo"
        hidden = cfg.get("hidden_size", 2048)
        if hidden == 4096:
            return "base"  # could be xl, but user can override
    return "turbo"  # safest default


def _resolve_path(name: str, checkpoint_root: Path, file_type: str = "checkpoint") -> Path:
    """Resolve a model path: local first, then registry, then return expected path."""
    local = checkpoint_root / name
    if local.exists() and any(local.iterdir()):
        return local

    registry_name = name.replace("-", "_") + ".safetensors"
    try:
        path_data = FilePaths.get_path(registry_name, file_type=file_type)
        if path_data.path and os.path.exists(path_data.path):
            return Path(path_data.path).parent
    except Exception:
        pass

    return local


def get_acestep_instance(
    dit_path: str = "acestep-v15-turbo",
    lm_path: str = "acestep-5Hz-lm-1.7B",
    text_encoder_path: str = "Qwen3-Embedding-0.6B",
    vae_path: str = "vae",
    checkpoint_root: str = "models/checkpoints",
    force_dtype: Optional[torch.dtype] = None,
) -> Tuple[AceStepPipeline, str]:
    """Initialize an ACE-Step inference pipeline.

    Returns:
        (pipeline, status_message)
    """
    checkpoint_root = Path(checkpoint_root)
    dtype = force_dtype or torch.bfloat16

    # ------------------------------------------------------------------
    # 1. DiT
    # ------------------------------------------------------------------
    app_logger.info("[ACE-Step] Loading DiT ...")
    dit_dir = _resolve_path(dit_path, checkpoint_root)
    if not dit_dir.exists() or not any(dit_dir.iterdir()):
        app_logger.info(f"[ACE-Step] Downloading DiT {dit_path} ...")
        dit_dir = ensure_acestep_checkpoint(
            repo_id=f"ACE-Step/{dit_path}",
            local_dir=str(dit_dir),
        )

    dit_config_path = dit_dir / "config.json"
    dit_cfg = AceStepConfig.from_json_file(str(dit_config_path)) if dit_config_path.exists() else AceStepConfig()
    dit_variant = _detect_dit_variant(dit_dir)

    dit = AceStepDiT(config=dit_cfg, variant=dit_variant, dtype=dtype)
    dit.model_path = str(dit_dir)
    # Find weights file
    weights_file = dit_dir / "model.safetensors"
    if not weights_file.exists():
        weights_file = dit_dir / "pytorch_model.bin"
    if weights_file.exists():
        dit.load_model(model_path=str(weights_file), dtype=dtype)
    else:
        app_logger.warning(f"[ACE-Step] No weights file found in {dit_dir}")

    # ------------------------------------------------------------------
    # 2. VAE
    # ------------------------------------------------------------------
    app_logger.info("[ACE-Step] Loading VAE ...")
    vae_dir = _resolve_path(vae_path, checkpoint_root)
    if not vae_dir.exists() or not any(vae_dir.iterdir()):
        app_logger.info(f"[ACE-Step] Downloading VAE to {vae_dir} ...")
        # VAE is bundled with the main Ace-Step1.5 repo
        vae_dir = ensure_acestep_checkpoint(
            repo_id="ACE-Step/Ace-Step1.5",
            local_dir=str(vae_dir),
        )

    # NOTE: AutoencoderOobleck is from diffusers — this is the existing
    # dependency already used by the original ACE-Step integration.
    try:
        from diffusers.models import AutoencoderOobleck
        vae = AutoencoderOobleck.from_pretrained(str(vae_dir))
        vae = vae.to(dit.gpu_device).to(dtype)
        vae.eval()
    except Exception as e:
        app_logger.error(f"[ACE-Step] Failed to load VAE: {e}")
        vae = None

    # ------------------------------------------------------------------
    # 3. Text encoder + tokenizer
    # ------------------------------------------------------------------
    app_logger.info("[ACE-Step] Loading text encoder ...")
    te_dir = _resolve_path(text_encoder_path, checkpoint_root)
    if not te_dir.exists() or not any(te_dir.iterdir()):
        app_logger.info(f"[ACE-Step] Downloading text encoder {text_encoder_path} ...")
        te_dir = ensure_acestep_checkpoint(
            repo_id=text_encoder_path,
            local_dir=str(te_dir),
        )

    text_tokenizer = AutoTokenizer.from_pretrained(str(te_dir), trust_remote_code=True)
    text_encoder = AutoModel.from_pretrained(str(te_dir), trust_remote_code=True)
    text_encoder = text_encoder.to(dit.gpu_device).to(dtype)
    text_encoder.eval()

    # ------------------------------------------------------------------
    # 4. 5Hz LM + tokenizer
    # ------------------------------------------------------------------
    app_logger.info("[ACE-Step] Loading 5Hz LM ...")
    lm_dir = _resolve_path(lm_path, checkpoint_root)
    if not lm_dir.exists() or not any(lm_dir.iterdir()):
        app_logger.info(f"[ACE-Step] Downloading LM {lm_path} ...")
        lm_dir = ensure_acestep_checkpoint(
            repo_id=f"ACE-Step/{lm_path}",
            local_dir=str(lm_dir),
        )

    lm_tokenizer = AutoTokenizer.from_pretrained(str(lm_dir), use_fast=True)
    lm_config = AutoConfig.from_pretrained(str(lm_dir), trust_remote_code=True)

    lm = AceStepLM(config=lm_config, dtype=dtype)
    lm.model_path = str(lm_dir)
    lm_weights = lm_dir / "model.safetensors"
    if not lm_weights.exists():
        lm_weights = lm_dir / "pytorch_model.bin"
    if lm_weights.exists():
        lm.load_model(model_path=str(lm_weights), dtype=dtype)
    else:
        app_logger.warning(f"[ACE-Step] No LM weights found in {lm_dir}")

    # ------------------------------------------------------------------
    # 5. Assemble pipeline
    # ------------------------------------------------------------------
    pipeline = AceStepPipeline(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        text_tokenizer=text_tokenizer,
        lm=lm,
        lm_tokenizer=lm_tokenizer,
        device=dit.gpu_device,
    )

    status = f"ACE-Step ready: DiT={dit_variant} LM={lm_path} device={dit.gpu_device}"
    app_logger.info(f"[ACE-Step] {status}")
    return pipeline, status
