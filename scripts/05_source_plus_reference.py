"""
Script 05: Source + Reference — Dual-Influence Generation

Demonstrates the combined power of `src_audio` and `reference_audio` working
together:
- src_audio      = the SONG to transform (content/structure)
- reference_audio = the STYLE to borrow (timbre/texture)

This is different from previous scripts:
- Script 03 (Cover):          Same song, new style (uses src_audio only)
- Script 04 (Style Transfer): New song, borrowed style (uses reference_audio only)
- Script 05 (This script):    Same song, borrowed style (uses BOTH)

Think of it as a "remix" or "re-imagining" — taking an existing track and
re-synthesizing it through the lens of another track's sonic character.

Known Behavior (from testing):
- The reference_audio effect is SUBTLE when combined with src_audio in a cover task.
- The model strongly prioritizes reconstructing the source structure; reference only
  provides a mild timbre nudge. Increasing audio_cover_strength above 0.5-0.7
  does NOT amplify the reference's influence — it actually makes the output more
  faithful to the source. For dramatic style changes, use Style Transfer (Script 04)
  or Cover without src_audio (Script 03).

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


def _get_source_audio_path() -> str:
    """Return path to the pre-generated indie folk source audio."""
    # Check prune2 subdirectory first (where generate_music saves outputs)
    src_dir = REPO_ROOT / "outputs" / "prune2" / "01_text2music" / "demo_b_with_lyrics"
    path = src_dir / "1c41f385-c029-8e8c-51b1-2cf6a1848cc6.wav"
    if not path.exists():
        files = sorted(src_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if files:
            path = files[-1]
        else:
            raise FileNotFoundError(
                "Source audio not found. Run scripts/01_text2music.py first."
            )
    return str(path)


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
        ref_dir = REPO_ROOT / "outputs" / "04_style_transfer" / "reference_heavy_metal"
        files = sorted(ref_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if files:
            path = files[-1]
        else:
            raise FileNotFoundError(
                "Reference audio not found. Run scripts/generate_reference_heavy_metal.py first."
            )
    return str(path)


def main():
    output_dir = REPO_ROOT / "outputs" / "05_source_plus_reference"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACE-Step 1.5 - Script 05: Source + Reference Dual-Influence")
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

    src_audio_path = _get_source_audio_path()
    ref_audio_path = _get_reference_audio_path()
    print(f"\n[SETUP] Source audio:     {src_audio_path}")
    print(f"[SETUP] Reference audio:  {ref_audio_path}")

    # ------------------------------------------------------------------
    # Demo A: Cover of source — no style reference
    # ------------------------------------------------------------------
    print("\n[Demo A] Cover of source (indie folk) — NO style reference ...")
    params = GenerationParams(
        task_type="cover",
        src_audio=src_audio_path,
        caption="melancholic indie folk with female vocals, piano, and strings",
        audio_cover_strength=0.7,
        inference_steps=50,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
    result = generate_music(
        dit_handler, llm_handler, params, config,
        save_dir=str(output_dir / "demo_a_cover_no_ref")
    )
    if result.success:
        print(f"  -> {result.audios[0]['path']}")
    else:
        print(f"  FAILED: {result.error}")

    # ------------------------------------------------------------------
    # Demo B: Cover of source WITH heavy metal style reference
    # ------------------------------------------------------------------
    print("\n[Demo B] Cover of source (indie folk) WITH heavy metal reference ...")
    print("  This re-synthesizes the source song while borrowing the reference's")
    print("  timbre and energy — effectively a 'metal cover' of the folk track.")
    params = GenerationParams(
        task_type="cover",
        src_audio=src_audio_path,
        reference_audio=ref_audio_path,
        caption="melancholic indie folk with female vocals, piano, and strings",
        audio_cover_strength=0.5,
        inference_steps=50,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
    result = generate_music(
        dit_handler, llm_handler, params, config,
        save_dir=str(output_dir / "demo_b_cover_with_ref")
    )
    if result.success:
        print(f"  -> {result.audios[0]['path']}")
    else:
        print(f"  FAILED: {result.error}")

    print("\n" + "=" * 60)
    print("Comparison:")
    print("  Demo A (cover, no ref) : Faithful re-synthesis of the indie folk song")
    print("  Demo B (cover + ref)   : Same song structure, but with heavy metal")
    print("                           timbre, distortion, and aggressive energy mixed in")
    print(f"\nAll outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
