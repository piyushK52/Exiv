"""
Generate a distinct heavy metal reference audio for style transfer demo.
Uses acestep-v15-sft for highest quality.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from exiv.components.models.acestep.handler import AceStepHandler
from exiv.components.models.acestep.llm_inference import LLMHandler
from exiv.components.models.acestep.inference import GenerationParams, GenerationConfig, generate_music


def main():
    output_dir = REPO_ROOT / "outputs" / "04_style_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)

    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    print("[1/2] Initializing DiT (acestep-v15-sft) ...")
    status, ok = dit_handler.initialize_service(
          
        config_path="acestep-v15-sft",
        device="cuda",
        use_flash_attention=True,
    )
    if not ok:
        print(f"Failed: {status}")
        sys.exit(1)
    print(f"OK: {status}")

    print("\n[2/2] Initializing 5Hz LM ...")
    from exiv.components.models.acestep.model_downloader import get_checkpoints_dir
    status, ok = llm_handler.initialize(
        checkpoint_dir=str(get_checkpoints_dir()),
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device="cuda",
    )
    if not ok:
        print(f"Failed: {status}")
        sys.exit(1)
    print(f"OK: {status}")

    print("\n[Generating] Heavy metal reference audio ...")
    params = GenerationParams(
        task_type="text2music",
        caption="aggressive heavy metal with distorted electric guitars, fast double-bass drums, screaming vocals, dark atmosphere",
        lyrics="[Instrumental]",
        duration=20,
        inference_steps=50,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
    result = generate_music(
        dit_handler, llm_handler, params, config,
        save_dir=str(output_dir / "reference_heavy_metal")
    )
    if result.success:
        print(f"\n  -> {result.audios[0]['path']}")
    else:
        print(f"\n  FAILED: {result.error}")


if __name__ == "__main__":
    main()
