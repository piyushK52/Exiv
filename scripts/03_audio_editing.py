"""
Script 03: Audio Editing

Demonstrates ACE-Step's versatile audio editing capabilities:
1. Cover Generation      - Re-synthesize audio with new style/timbre
2. Repaint               - Regenerate a specific time segment
3. Track Extraction      - Isolate a specific stem (vocals, drums, etc.)
4. Lego / Add Layer      - Add a new instrument layer to existing audio
5. Complete / Vocal-to-BGM - Auto-complete a partial track with accompaniment

Quality Expectations (honest labels):
- Cover:      Spectral restyling works, but output is ~1 dB quieter than source.
- Repaint:    Segment replacement works, but repainted region is ~2.5 dB quieter,
              creating an audible volume dip.
- Extract:    Percussive isolation is objectively correct (percussive % triples),
              but output is very quiet (~13 dB below source).
- Lego:       KNOWN LIMITATION on base model — output ignores source and generates
              unrelated audio (GitHub issue #1088). Works better on sft variant.
- Complete:   KNOWN LIMITATION on base model — output ignores source and generates
              unrelated audio (GitHub issue #1088). Works better on sft variant.

Model Note:
- This script uses acestep-v15-base because it is the ONLY variant that
  supports all advanced editing tasks (extract, lego, complete).
  The sft/turbo variants do NOT support extract, lego, or complete.
  Cover and repaint work on all variants.

Models:
- DiT: acestep-v15-base (2B, ~4.7GB)
- LM: acestep-5Hz-lm-1.7B (~3.4GB)
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from exiv.components.models.acestep.handler import AceStepHandler
from exiv.components.models.acestep.llm_inference import LLMHandler
from exiv.components.models.acestep.inference import GenerationParams, GenerationConfig, generate_music


def _get_source_audio_path() -> str:
    """Return path to the pre-generated source audio clip."""
    path = REPO_ROOT / "outputs" / "03_audio_editing" / "source" / "b22d30fa-d8e9-135a-1430-f183ed005807.wav"
    if not path.exists():
        src_dir = REPO_ROOT / "outputs" / "03_audio_editing" / "source"
        files = sorted(src_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if files:
            path = files[-1]
        else:
            raise FileNotFoundError(f"Source audio not found: {path}")
    print(f"\n[SETUP] Using pre-generated source audio: {path}")
    return str(path)


def main():
    output_dir = REPO_ROOT / "outputs" / "03_audio_editing"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACE-Step 1.5 - Script 03: Audio Editing")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Initialize handlers
    # ------------------------------------------------------------------
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    print("\n[1/2] Initializing DiT (acestep-v15-base) ...")
    status, ok = dit_handler.initialize_service(
          
        config_path="acestep-v15-base",
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

    # Use the pre-generated source track
    src_audio_path = _get_source_audio_path()

    # ------------------------------------------------------------------
    # Demo A: Cover Generation
    # ------------------------------------------------------------------
    print("\n[Demo A] Cover Generation — orchestral symphonic arrangement ...")
    params = GenerationParams(
        task_type="cover",
        src_audio=src_audio_path,
        caption="orchestral symphonic arrangement with strings and brass",
        audio_cover_strength=0.7,
        inference_steps=50,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
    result = generate_music(dit_handler, llm_handler, params, config,
                            save_dir=str(output_dir / "demo_a_cover"))
    if result.success:
        print(f"  -> {result.audios[0]['path']}")
    else:
        print(f"  FAILED: {result.error}")

    # ------------------------------------------------------------------
    # Demo B: Repaint (segment replacement)
    # ------------------------------------------------------------------
    print("\n[Demo B] Repaint — replace 5s to 10s with piano solo ...")
    params = GenerationParams(
        task_type="repaint",
        src_audio=src_audio_path,
        repainting_start=5.0,
        repainting_end=10.0,
        caption="smooth transition with piano solo",
        inference_steps=50,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
    result = generate_music(dit_handler, llm_handler, params, config,
                            save_dir=str(output_dir / "demo_b_repaint"))
    if result.success:
        print(f"  -> {result.audios[0]['path']}")
    else:
        print(f"  FAILED: {result.error}")

    # ------------------------------------------------------------------
    # Demo C: Track Extraction (stem separation)
    # ------------------------------------------------------------------
    print("\n[Demo C] Track Extraction — isolate drums ...")
    params = GenerationParams(
        task_type="extract",
        src_audio=src_audio_path,
        instruction="Extract the drums track from the audio:",
        inference_steps=50,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
    result = generate_music(dit_handler, llm_handler, params, config,
                            save_dir=str(output_dir / "demo_c_extract_drums"))
    if result.success:
        print(f"  -> {result.audios[0]['path']}")
    else:
        print(f"  FAILED: {result.error}")

    # ------------------------------------------------------------------
    # Demo D: Lego (add instrument layer)
    # ------------------------------------------------------------------
    print("\n[Demo D] Lego — add lead guitar layer ...")
    params = GenerationParams(
        task_type="lego",
        src_audio=src_audio_path,
        instruction="Generate the guitar track based on the audio context:",
        caption="lead guitar melody with bluesy feel",
        repainting_start=0.0,
        repainting_end=-1,
        inference_steps=50,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
    result = generate_music(dit_handler, llm_handler, params, config,
                            save_dir=str(output_dir / "demo_d_lego_guitar"))
    if result.success:
        print(f"  -> {result.audios[0]['path']}")
    else:
        print(f"  FAILED: {result.error}")

    # ------------------------------------------------------------------
    # Demo E: Complete / Vocal-to-BGM
    # ------------------------------------------------------------------
    print("\n[Demo E] Complete — auto-complete with drums, bass, synth ...")
    params = GenerationParams(
        task_type="complete",
        src_audio=src_audio_path,
        instruction="Complete the input track with drums, bass, and synth:",
        caption="synth-pop style completion",
        inference_steps=50,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
    result = generate_music(dit_handler, llm_handler, params, config,
                            save_dir=str(output_dir / "demo_e_complete"))
    if result.success:
        print(f"  -> {result.audios[0]['path']}")
    else:
        print(f"  FAILED: {result.error}")

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
