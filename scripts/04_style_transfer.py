"""
Script 04: Style Transfer with Reference Audio

Demonstrates ACE-Step's reference-audio style transfer feature:
- Generate a new song while borrowing the style/timbre from an existing audio file.
- The reference audio guides the sonic character (instruments, texture, energy)
  without changing the melody or structure dictated by the text prompt.

This is different from Cover (Script 03):
- Cover: re-synthesizes the SAME song with new style
- Style Transfer: generates a COMPLETELY NEW song influenced by reference style

Models:
- DiT: acestep-v15-sft (2B, highest quality, ~4.7GB)
- LM: acestep-5Hz-lm-1.7B (~3.4GB)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from exiv.components.models.acestep.handler import AceStepHandler
from exiv.components.models.acestep.llm_inference import LLMHandler
from exiv.components.models.acestep.inference import GenerationParams, GenerationConfig, generate_music


def _get_reference_audio_path() -> str:
    """Return path to the pre-generated heavy metal reference audio."""
    path = (
        REPO_ROOT
        / "outputs"
        / "04_style_transfer"
        / "reference_heavy_metal"
        / "43b4e6d9-4099-f054-8c61-f3e8302f9f81.wav"
    )
    if not path.exists():
        # Fallback: discover newest wav in the reference directory
        ref_dir = REPO_ROOT / "outputs" / "04_style_transfer" / "reference_heavy_metal"
        files = sorted(ref_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if files:
            path = files[-1]
        else:
            raise FileNotFoundError(
                f"Reference audio not found. Run scripts/generate_reference_heavy_metal.py first."
            )
    return str(path)


def main():
    output_dir = REPO_ROOT / "outputs" / "04_style_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACE-Step 1.5 - Script 04: Style Transfer with Reference Audio")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Initialize handlers
    # ------------------------------------------------------------------
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    print("\n[1/2] Initializing DiT (acestep-v15-sft) ...")
    status, ok = dit_handler.initialize_service(
          
        config_path="acestep-v15-sft",
        device="cuda",
        use_flash_attention=True,
    )
    if not ok:
        print(f"Failed to initialize DiT: {status}")
        sys.exit(1)
    print(f"DiT initialized: {status}")

    print("\n[2/2] Initializing 5Hz LM (acestep-5Hz-lm-1.7B) ...")
    from exiv.components.models.acestep.model_downloader import get_checkpoints_dir

    checkpoint_dir = str(get_checkpoints_dir())
    status, ok = llm_handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device="cuda",
    )
    if not ok:
        print(f"Failed to initialize LM: {status}")
        sys.exit(1)
    print(f"LM initialized: {status}")

    reference_audio_path = _get_reference_audio_path()
    print(f"\n[SETUP] Reference audio: {reference_audio_path}")

    caption = "melancholic acoustic guitar ballad with soft strings and gentle piano"

    # ------------------------------------------------------------------
    # Demo A: Baseline — no reference audio
    # ------------------------------------------------------------------
    print(f"\n[Demo A] Baseline — generate WITHOUT reference ...")
    print(f"  Prompt: '{caption}'")
    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics="[Instrumental]",
        duration=20,
        inference_steps=50,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
    result = generate_music(
        dit_handler, llm_handler, params, config,
        save_dir=str(output_dir / "demo_a_baseline")
    )
    if result.success:
        print(f"  -> {result.audios[0]['path']}")
    else:
        print(f"  FAILED: {result.error}")

    # ------------------------------------------------------------------
    # Demo B: Style Transfer — with heavy metal reference
    # ------------------------------------------------------------------
    print(f"\n[Demo B] Style Transfer — generate WITH heavy metal reference ...")
    print(f"  Prompt: '{caption}'")
    print(f"  Reference: heavy metal | cover_strength: 0.5")
    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics="[Instrumental]",
        duration=20,
        inference_steps=50,
        reference_audio=reference_audio_path,
        audio_cover_strength=0.5,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
    result = generate_music(
        dit_handler, llm_handler, params, config,
        save_dir=str(output_dir / "demo_b_with_reference")
    )
    if result.success:
        print(f"  -> {result.audios[0]['path']}")
    else:
        print(f"  FAILED: {result.error}")

    print("\n" + "=" * 60)
    print("Comparison:")
    print("  Demo A (baseline)     : Pure melancholic ballad")
    print("  Demo B (style transfer): Same ballad prompt, but influenced by")
    print("                           the heavy metal reference's timbre/energy")
    print(f"\nAll outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
