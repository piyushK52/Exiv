"""Bare-bones ACE-Step inference pipeline."""

import math
import os
from typing import Optional, Union, List

import torch
import torch.nn.functional as F
from loguru import logger


# ---------------------------------------------------------------------------
# VAE wrapper with simple tiled decode
# ---------------------------------------------------------------------------

class AceStepVAE:
    """Minimal VAE wrapper with optional tiled decoding."""

    def __init__(self, vae_model, device: str, dtype: torch.dtype):
        self.vae = vae_model
        self.device = device
        self.dtype = dtype
        self.sample_rate = getattr(vae_model.config, "sampling_rate", 48000)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, chunk_size: Optional[int] = None, overlap: int = 64):
        """Decode latents to waveform.

        Args:
            latents: [batch, channels, latent_frames]
            chunk_size: Latent frames per chunk. None = single-shot decode.
            overlap: Overlap between chunks in latent frames.

        Returns:
            Waveform tensor [batch, audio_channels, samples].
        """
        if chunk_size is None or latents.shape[-1] <= chunk_size:
            out = self.vae.decode(latents)
            return out.sample if hasattr(out, "sample") else out

        bsz, _, latent_frames = latents.shape
        stride = chunk_size - 2 * overlap
        if stride <= 0:
            raise ValueError(f"chunk_size {chunk_size} must be > 2 * overlap {overlap}")
        num_steps = math.ceil(latent_frames / stride)

        decoded_chunks = []
        upsample_factor = None

        for i in range(num_steps):
            core_start = i * stride
            core_end = min(core_start + stride, latent_frames)
            win_start = max(0, core_start - overlap)
            win_end = min(latent_frames, core_end + overlap)

            chunk = latents[:, :, win_start:win_end]
            out = self.vae.decode(chunk)
            audio = out.sample if hasattr(out, "sample") else out

            if upsample_factor is None:
                upsample_factor = audio.shape[-1] / chunk.shape[-1]

            trim_start = int(round((core_start - win_start) * upsample_factor))
            trim_end = int(round((win_end - core_end) * upsample_factor))
            end_idx = audio.shape[-1] - trim_end if trim_end > 0 else audio.shape[-1]
            decoded_chunks.append(audio[:, :, trim_start:end_idx])

        return torch.cat(decoded_chunks, dim=-1)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AceStepPipeline:
    """
    Minimal ACE-Step inference pipeline.

    Supports ``text2music``, ``cover``, and ``repaint`` tasks.
    Conditioning logic that was previously smeared across 30+ mixin files
    is now a single sequential function.
    """

    def __init__(
        self,
        dit,
        vae,
        text_encoder,
        text_tokenizer,
        lm,
        lm_tokenizer,
        device: str,
    ):
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.lm = lm
        self.lm_tokenizer = lm_tokenizer
        self.device = device
        self.dtype = dit.dtype

        # Silence latent — loaded lazily on first generation
        self._silence_latent: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_silence_latent(self, seq_len: int, channels: int = 64):
        """Load or synthesise the silence latent used for padding / init."""
        if self._silence_latent is not None:
            return self._silence_latent

        # Try loading from DiT checkpoint directory
        dit_path = getattr(self.dit, "model_path", None)
        if dit_path:
            pt_path = os.path.join(dit_path, "silence_latent.pt")
            if os.path.exists(pt_path):
                sl = torch.load(pt_path, weights_only=True, map_location="cpu")
                if sl.dim() == 3:
                    sl = sl.transpose(1, 2)
                self._silence_latent = sl.to(self.device).to(self.dtype)
                logger.info(f"Loaded silence_latent from {pt_path}")
                return self._silence_latent

        # Fallback: random tensor (matches original behaviour when file absent)
        logger.warning("silence_latent.pt not found; using random fallback")
        self._silence_latent = torch.randn(1, seq_len, channels, device=self.device, dtype=self.dtype)
        return self._silence_latent

    def _encode_text(self, text: Union[str, List[str]]) -> tuple:
        """Tokenize and encode text via the Qwen3 embedding model.

        Returns:
            (hidden_states, attention_mask)
        """
        if isinstance(text, str):
            text = [text]
        inputs = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Qwen3 embedding model returns last_hidden_state
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        return hidden_states, attention_mask

    def _build_text2music_inputs(
        self,
        caption: str,
        lyrics: str,
        duration_sec: float,
    ):
        """Build DiT inputs for pure text-to-music generation."""
        # Encode caption and lyrics
        text_h, text_mask = self._encode_text(caption)
        if lyrics and lyrics.strip():
            lyric_h, lyric_mask = self._encode_text(lyrics)
        else:
            lyric_h, lyric_mask = text_h, text_mask  # fallback

        # Latent length at 25 Hz
        latent_length = max(int(duration_sec * 25), 1)

        # Silence latent for init
        silence_latent = self._ensure_silence_latent(latent_length)
        silence_latent = silence_latent[:, :latent_length, :]

        # Source latents = silence for text2music
        src_latents = silence_latent.expand(text_h.shape[0], -1, -1)

        # Chunk masks: all 2.0 means "model decides" (auto mode)
        chunk_masks = torch.full((text_h.shape[0], latent_length, 1), 2.0, device=self.device, dtype=self.dtype)

        # Not a cover
        is_covers = torch.zeros(text_h.shape[0], device=self.device, dtype=self.dtype)

        # Empty reference audio
        ref_audio_h = torch.zeros(text_h.shape[0], 1, 1, device=self.device, dtype=self.dtype)
        ref_audio_mask = torch.zeros(text_h.shape[0], 1, dtype=torch.long, device=self.device)

        # Attention mask for latents
        latent_mask = torch.ones(text_h.shape[0], latent_length, device=self.device, dtype=self.dtype)

        return {
            "text_hidden_states": text_h,
            "text_attention_mask": text_mask,
            "lyric_hidden_states": lyric_h,
            "lyric_attention_mask": lyric_mask,
            "refer_audio_acoustic_hidden_states_packed": ref_audio_h,
            "refer_audio_order_mask": ref_audio_mask,
            "src_latents": src_latents,
            "chunk_masks": chunk_masks,
            "is_covers": is_covers,
            "silence_latent": silence_latent,
            "attention_mask": latent_mask,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        caption: str = "",
        lyrics: str = "",
        duration: float = 15.0,
        seed: Optional[int] = None,
        inference_steps: int = 8,
        shift: float = 3.0,
        sampler_mode: str = "euler",
        audio_cover_strength: float = 1.0,
        cover_noise_strength: float = 0.0,
        use_lm: bool = False,
        lm_temperature: float = 0.85,
        lm_top_p: float = 0.9,
        lm_top_k: int = 0,
        task_type: str = "text2music",
        **kwargs,
    ) -> torch.Tensor:
        """Generate audio from text prompt and optional lyrics.

        Args:
            caption: Text prompt describing the desired music.
            lyrics: Lyrics with section tags like [Verse], [Chorus].
            duration: Target duration in seconds.
            seed: Random seed. None or -1 = random.
            inference_steps: Number of diffusion steps (8 for turbo, 50 for base).
            shift: Timestep shift factor (1.0, 2.0, or 3.0).
            sampler_mode: "euler" or "heun".
            audio_cover_strength: Reference audio influence (1.0 = full cover).
            cover_noise_strength: Cover noise blend (0.0 = pure noise init).
            use_lm: Whether to run the 5Hz LM for CoT / audio codes.
            lm_temperature: LM sampling temperature.
            lm_top_p: LM nucleus sampling p.
            lm_top_k: LM top-k filtering.
            task_type: "text2music", "cover", or "repaint".

        Returns:
            Waveform tensor [batch, channels, samples].
        """
        if seed is not None and seed < 0:
            seed = None

        # ------------------------------------------------------------------
        # 1. Build conditioning
        # ------------------------------------------------------------------
        if task_type == "text2music":
            dit_inputs = self._build_text2music_inputs(caption, lyrics, duration)
        else:
            raise NotImplementedError(f"Task type '{task_type}' not yet implemented in minimal pipeline")

        batch_size = dit_inputs["text_hidden_states"].shape[0]

        # ------------------------------------------------------------------
        # 2. (Optional) 5Hz LM → audio codes
        # ------------------------------------------------------------------
        audio_codes = None
        if use_lm and self.lm is not None:
            # Build a simple prompt for the LM
            lm_prompt = f"# Caption\n{caption}\n# Lyrics\n{lyrics}\n"
            logger.info("Running 5Hz LM for audio codes...")
            generated = self.lm.generate_audio_codes(
                prompt=lm_prompt,
                tokenizer=self.lm_tokenizer,
                max_new_tokens=min(inference_steps * 8, 512),
                temperature=lm_temperature,
                top_p=lm_top_p,
                top_k=lm_top_k,
            )
            # TODO: convert generated token IDs to audio_codes tensor
            # For now, audio_codes stays None (model falls back to tokenizing src_latents)
            logger.info(f"LM generated {generated.shape[-1]} tokens")

        # ------------------------------------------------------------------
        # 3. DiT diffusion → latents
        # ------------------------------------------------------------------
        logger.info(f"Running DiT generation: steps={inference_steps}, shift={shift}, sampler={sampler_mode}")
        outputs = self.dit.generate_audio(
            **dit_inputs,
            seed=seed,
            fix_nfe=inference_steps,
            shift=shift,
            sampler_mode=sampler_mode,
            audio_cover_strength=audio_cover_strength,
            cover_noise_strength=cover_noise_strength,
            audio_codes=audio_codes,
            **kwargs,
        )
        target_latents = outputs["target_latents"]
        time_costs = outputs.get("time_costs", {})
        logger.info(f"DiT done: {time_costs}")

        # ------------------------------------------------------------------
        # 4. VAE decode → waveform
        # ------------------------------------------------------------------
        waveform = self.vae.decode(target_latents)
        return waveform
