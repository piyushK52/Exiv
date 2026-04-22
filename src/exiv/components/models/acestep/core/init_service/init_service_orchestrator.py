"""Top-level initialization orchestration for the handler."""

import os
import traceback
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger

from exiv.utils import gpu as gpu_config

def _cuda_supports_bfloat16() -> bool:
    """Return whether the active CUDA device supports native bfloat16 kernels."""
    return gpu_config.cuda_supports_bfloat16()


class InitServiceOrchestratorMixin:
    """Public ``initialize_service`` orchestration entrypoint."""

    def initialize_service(
        self,
        config_path: str,
        project_root: str = "",
        device: str = "auto",
        use_flash_attention: bool = False,
        compile_model: bool = False,
        offload_to_cpu: bool = False,
        offload_dit_to_cpu: bool = False,
        quantization: Optional[str] = None,
        prefer_source: Optional[str] = None,
        use_mlx_dit: bool = True,
    ) -> Tuple[str, bool]:
        """Initialize model artifacts and runtime backends for generation.

        This method intentionally supports repeated calls to reinitialize models
        with new settings; it does not short-circuit when components are already loaded.
        """
        try:
            if config_path is None:
                config_path = "acestep-v15-turbo"
                logger.warning(
                    "[initialize_service] config_path not set; defaulting to 'acestep-v15-turbo'."
                )

            resolved_device = self._resolve_initialize_device(device)
            self.device = resolved_device
            self.offload_to_cpu = offload_to_cpu
            self.offload_dit_to_cpu = offload_dit_to_cpu

            normalized_compile, normalized_quantization, mlx_compile_requested = self._configure_initialize_runtime(
                device=resolved_device,
                compile_model=compile_model,
                quantization=quantization,
            )
            self.compiled = normalized_compile
            if resolved_device == "cuda":
                if gpu_config.cuda_supports_bfloat16():
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16
                    logger.info(
                        "[initialize_service] Pre-Ampere CUDA detected: "
                        "using float16 instead of bfloat16."
                    )
            else:
                self.dtype = torch.bfloat16 if resolved_device == "xpu" else torch.float32
            self.quantization = normalized_quantization
            try:
                self._validate_quantization_setup(
                    quantization=self.quantization,
                    compile_model=normalized_compile,
                )
            except ImportError as exc:
                if self.quantization is not None:
                    logger.warning(
                        "[initialize_service] Quantization disabled: {}",
                        exc,
                    )
                    self.quantization = None
                else:
                    raise

            from exiv.components.models.acestep.model_downloader import get_checkpoints_dir
            checkpoint_dir = str(get_checkpoints_dir())
            checkpoint_path = Path(checkpoint_dir)

            precheck_failure = self._ensure_models_present(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                prefer_source=prefer_source,
            )
            if precheck_failure is not None:
                self.model = None
                self.vae = None
                self.text_encoder = None
                self.text_tokenizer = None
                self.config = None
                self.silence_latent = None
                return precheck_failure

            self._sync_model_code_if_needed(config_path, checkpoint_path)

            model_path = os.path.join(checkpoint_dir, config_path)
            self._load_main_model_from_checkpoint(
                model_checkpoint_path=model_path,
                device=resolved_device,
                use_flash_attention=use_flash_attention,
                compile_model=normalized_compile,
                quantization=self.quantization,
            )
            vae_path = self._load_vae_model(
                checkpoint_dir=checkpoint_dir,
                device=resolved_device,
                compile_model=normalized_compile,
            )
            text_encoder_path = self._load_text_encoder_and_tokenizer(
                checkpoint_dir=checkpoint_dir,
                device=resolved_device,
            )

            status_msg = self._build_initialize_status_message(
                device=resolved_device,
                model_path=model_path,
                vae_path=vae_path,
                text_encoder_path=text_encoder_path,
                dtype=self.dtype,
                attention=getattr(self.config, "_attn_implementation", "eager"),
                compile_model=normalized_compile,
                offload_to_cpu=offload_to_cpu,
                offload_dit_to_cpu=offload_dit_to_cpu,
                quantization=self.quantization,
            )

            self.last_init_params = {
                "project_root": project_root,
                "config_path": config_path,
                "device": resolved_device,
                "use_flash_attention": use_flash_attention,
                "compile_model": normalized_compile,
                "offload_to_cpu": offload_to_cpu,
                "offload_dit_to_cpu": offload_dit_to_cpu,
                "quantization": self.quantization,
                "prefer_source": prefer_source,
            }

            return status_msg, True
        except Exception as exc:
            self.model = None
            self.vae = None
            self.text_encoder = None
            self.text_tokenizer = None
            self.config = None
            self.silence_latent = None
            error_msg = f"Error initializing model: {str(exc)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.exception(error_msg)
            return error_msg, False
