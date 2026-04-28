"""Exiv-style ModelMixin wrapper for ACE-Step DiT."""

from typing import Optional, Union

import torch
from torch import Tensor

from exiv.model_utils.model_mixin import ModelMixin, ModelArchConfig
from exiv.components.latent_format import LatentFormat
from exiv.components.enum import ModelType
from exiv.utils.logging import app_logger
from exiv.utils.dtype import cast_to


class AceStepLatentFormat(LatentFormat):
    """Latent format for ACE-Step audio latents."""

    def __init__(self):
        # ACE-Step DCAE latents are roughly [-8, 8] range
        self.scale_factor = 1.0
        self.latent_channels = 64

    def process_in(self, latent: Tensor) -> Tensor:
        return latent

    def process_out(self, latent: Tensor) -> Tensor:
        return latent


class AceStepDiT(ModelMixin):
    """
    Exiv ModelMixin wrapper around ACE-Step DiT (AceStepConditionGenerationModel).

    The inner DiT is a transformers.PreTrainedModel subclass with custom
    diffusion-specific architecture.  This wrapper gives it exiv's:
    - zero-init meta loading
    - quantization hook integration
    - dtype / device casting
    - memory footprint reporting
    """

    VARIANT_MAP = {
        "acestep-v15-base": ("base", "core.modeling_acestep_v15_base"),
        "acestep-v15-sft": ("sft", "core.modeling_acestep_v15_sft"),
        "acestep-v15-turbo": ("turbo", "core.modeling_acestep_v15_turbo"),
        "acestep-v15-xl-base": ("xl_base", "core.modeling_acestep_v15_xl_base"),
        "acestep-v15-xl-sft": ("xl_sft", "core.modeling_acestep_v15_xl_sft"),
        "acestep-v15-xl-turbo": ("xl_turbo", "core.modeling_acestep_v15_xl_turbo"),
    }

    def __init__(
        self,
        config,
        variant: str = "turbo",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        quant_type=None,
        quant_config=None,
        force_load_mode=None,
        **kwargs,
    ):
        # ModelMixin metaclass pops dtype/quant_type/force_load_mode, but we
        # keep variant/config for inner model creation.
        super().__init__(
            device=device,
            quant_type=quant_type,
            quant_config=quant_config,
            force_load_mode=force_load_mode,
            dtype=dtype,
            **kwargs,
        )
        self.config = config
        self.variant = variant
        self.transformer = self._create_inner_model(config, variant)

        # exiv model arch config
        self.model_arch_config = ModelArchConfig(model_type=ModelType.FLOW)
        self.model_arch_config.latent_format = AceStepLatentFormat()

    def _create_inner_model(self, config, variant: str):
        app_logger.info(f"Creating AceStep DiT inner model (variant={variant})")
        if variant == "turbo":
            from .core.modeling_acestep_v15_turbo import AceStepConditionGenerationModel
        elif variant == "base":
            from .core.modeling_acestep_v15_base import AceStepConditionGenerationModel
        elif variant == "sft":
            from .core.modeling_acestep_v15_sft import AceStepConditionGenerationModel
        elif variant == "xl_base":
            from .core.modeling_acestep_v15_xl_base import AceStepConditionGenerationModel
        elif variant == "xl_sft":
            from .core.modeling_acestep_v15_xl_sft import AceStepConditionGenerationModel
        elif variant == "xl_turbo":
            from .core.modeling_acestep_v15_xl_turbo import AceStepConditionGenerationModel
        else:
            raise ValueError(f"Unknown AceStep DiT variant: {variant}")

        # Suppress post_init() because ModelMixin's metaclass creates the model
        # on the meta device and post_init()->init_weights() can fail when
        # writing to meta tensors.
        _original_post_init = AceStepConditionGenerationModel.post_init
        AceStepConditionGenerationModel.post_init = lambda self: None
        try:
            model = AceStepConditionGenerationModel(config)
        finally:
            AceStepConditionGenerationModel.post_init = _original_post_init
        return model

    def load_model(self, model_path=None, force_download=False, download_url=None, dtype=None, model_type=None):
        """Load weights into the inner DiT using load_state_dict.

        Overrides ModelMixin.load_model because PreTrainedModel handles
        param/buffer loading more robustly than manual per-tensor assignment.
        """
        from exiv.model_utils.helper_methods import get_state_dict
        from exiv.utils.file import ensure_model_availability
        from exiv.utils.device import ProcDevice

        model_path = model_path or self.model_path
        assert model_path is not None, "model_path is required"

        self._set_quantization(model_path)
        self.dtype = dtype or self.dtype

        model_path = ensure_model_availability(model_path, download_url, force_download)
        state_dict = get_state_dict(model_path, model_type=model_type)

        # PreTrainedModel.load_state_dict handles meta-device materialisation
        # via assign=True (PyTorch >= 2.1) and strict=False lets us skip
        # keys that may be present in the checkpoint but not in this variant.
        missing, unexpected = self.transformer.load_state_dict(
            state_dict, strict=False, assign=True
        )
        if missing:
            app_logger.warning(f"AceStepDiT missing keys: {missing}")
        if unexpected:
            app_logger.info(f"AceStepDiT unexpected keys (skipped): {unexpected}")

        # Move to target device / dtype
        self.transformer = self.transformer.to(self.dtype).to(self.gpu_device)
        self.transformer.eval()

        app_logger.info(f"AceStepDiT loaded from {model_path}")

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def generate_audio(self, *args, **kwargs):
        """Delegate to the DiT's native diffusion generation."""
        return self.transformer.generate_audio(*args, **kwargs)

    def get_memory_footprint_params(self):
        """Return memory params for exiv activation-size calculations."""
        cfg = self.config
        return {
            "hidden_size": getattr(cfg, "hidden_size", 2048),
            "num_layers": getattr(cfg, "num_hidden_layers", 24),
            "seq_len_factor": 1.0,
            "vocab_size": getattr(cfg, "vocab_size", 64003),
        }

    def get_model_sampling_obj(self):
        # DiT handles its own flow-matching sampling inside generate_audio()
        from exiv.components.samplers.sampler_types import get_model_sampling
        return get_model_sampling(ModelType.FLOW)
