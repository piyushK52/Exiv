"""Exiv ARModelMixin wrapper for ACE-Step 5Hz LM.

Built from transformers Qwen3 primitives but owned by Exiv.
"""

from typing import Optional

import torch
import torch.nn as nn

from transformers import AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RMSNorm

from exiv.model_utils.autoregressive_model_mixin import ARModelMixin, ARModelArchConfig
from exiv.components.models.common import AROutput
from exiv.components.enum import Model
from exiv.utils.logging import app_logger
from exiv.utils.device import ProcDevice


class AceStepLM(ARModelMixin):
    """
    ACE-Step 5Hz Language Model.

    Architecture: Qwen3-based decoder stack + LM head.
    We compose ``transformers.Qwen3Model`` as the backbone and add our own
    ``lm_head``, giving us an Exiv-native ARModelMixin while reusing
    battle-tested Qwen3 layer implementations.
    """

    def __init__(
        self,
        config,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        quant_type=None,
        quant_config=None,
        force_load_mode=None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            quant_type=quant_type,
            quant_config=quant_config,
            force_load_mode=force_load_mode,
            dtype=dtype,
            **kwargs,
        )
        self.config = config
        self.vocab_size = config.vocab_size

        # Qwen3 backbone (embeddings + decoder layers + norm + rotary)
        # Suppress post_init() because ARModuleMeta creates the model on the
        # meta device and post_init()->init_weights() can fail when writing
        # to meta tensors.
        _original_post_init = Qwen3Model.post_init
        Qwen3Model.post_init = lambda self: None
        try:
            self.model = Qwen3Model(config)
        finally:
            Qwen3Model.post_init = _original_post_init

        # Language modelling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.model_arch_config = ARModelArchConfig(model_type=Model.ACESTEP_5HZ_LM_1_7B)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> AROutput:
        """Single forward step compatible with ARModelMixin.generate()."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return AROutput(
            logits=logits,
            past_key_values=outputs.past_key_values,
            extra={},
        )

    def load_model(self, model_path=None, force_download=False, download_url=None, dtype=None, model_type=None):
        """Load weights.

        Checkpoint keys follow HF Qwen3 format:
        ``model.embed_tokens.weight``, ``model.layers.0...``, ``lm_head.weight``.
        Our module hierarchy matches exactly, so no remapping is required.
        """
        from exiv.model_utils.helper_methods import get_state_dict
        from exiv.utils.file import ensure_model_availability

        model_path = model_path or self.model_path
        assert model_path is not None, "model_path is required"
        self.dtype = dtype or self.dtype

        model_path = ensure_model_availability(model_path, download_url, force_download)
        state_dict = get_state_dict(model_path, model_type=model_type)
        model_state_dict = self.state_dict()

        loaded = 0
        skipped = 0
        for param_name, param in state_dict.items():
            if param_name not in model_state_dict:
                app_logger.warning(f"AceStepLM skipping unknown key: {param_name}")
                skipped += 1
                continue

            param = param.to(self.dtype)
            old_param = getattr(self, param_name, None)
            if isinstance(old_param, (torch.nn.Parameter, torch.Tensor)) and old_param.is_contiguous():
                param = param.contiguous()

            # Use ARModelMixin's helper for device placement
            from exiv.model_utils.helper_methods import set_module_tensor_to_device
            set_module_tensor_to_device(self, param_name, ProcDevice.CPU.value, value=param, dtype=self.dtype)
            loaded += 1

        # Clean up any leftover meta tensors
        for name, p in self.named_parameters():
            if p.device.type == "meta":
                from exiv.model_utils.helper_methods import set_module_tensor_to_device
                set_module_tensor_to_device(
                    self, name, ProcDevice.CPU.value,
                    value=torch.zeros_like(p, device=ProcDevice.CPU.value, dtype=self.dtype),
                    dtype=self.dtype,
                )

        app_logger.info(f"AceStepLM loaded {loaded} params, skipped {skipped} from {model_path}")

    @torch.no_grad()
    def generate_audio_codes(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: int = 512,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: int = 0,
        logits_processor=None,
        **kwargs,
    ):
        """Convenience wrapper for LM generation with constrained decoding."""
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.gpu_device)
        attention_mask = inputs["attention_mask"].to(self.gpu_device)

        # Build default logits processors if none provided
        if logits_processor is None:
            from transformers.generation.logits_process import LogitsProcessorList
            logits_processor = LogitsProcessorList()

        generated = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logits_processors=logits_processor,
            **kwargs,
        )
        return generated
