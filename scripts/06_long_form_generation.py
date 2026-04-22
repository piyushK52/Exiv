"""
Script 06: Long-Form Generation

Demonstrates ACE-Step's ability to generate full-length songs (90–120s) with
proper musical structure — intro, verse, chorus, bridge, and outro — rather than
short 15–20s clips.

This script uses the LM's Simple Mode to automatically generate a complete song
blueprint (caption, lyrics, BPM, key, duration) from a natural language query,
then feeds the blueprint into the DiT for long-form synthesis.

Models:
- DiT: acestep-v15-sft (2B, highest quality, ~4.7GB)
- LM: acestep-5Hz-lm-1.7B (~3.4GB)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from exiv.components.models.acestep.handler import AceStepHandler
from exiv.components.models.acestep.llm_inference import LLMHandler
from exiv.components.models.acestep.inference import GenerationParams, GenerationConfig, generate_music, create_sample


def main():
    output_dir = REPO_ROOT / "outputs" / "06_long_form"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACE-Step 1.5 - Script 06: Long-Form Generation")
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

    # ------------------------------------------------------------------
    # Demo A: Full song with lyrics (120s)
    # ------------------------------------------------------------------
    print("\n[Demo A] Full song with lyrics — generating blueprint via LM ...")
    query = "an emotional progressive rock epic about chasing dreams, with soaring vocals, dynamic piano, and building guitar solos"
    print(f"  User query: '{query}'")

    sample = create_sample(llm_handler, query=query, vocal_language="en")
    if not sample.success:
        print(f"  LM blueprint generation failed: {sample.error}")
        sys.exit(1)

    print(f"\n  LM-generated blueprint:")
    print(f"    Caption : {sample.caption[:80]}...")
    print(f"    Lyrics  : {sample.lyrics[:60]}...")
    print(f"    BPM     : {sample.bpm}")
    print(f"    Key     : {sample.keyscale}")
    print(f"    Duration: {sample.duration}s")

    duration = sample.duration if sample.duration and sample.duration > 60 else 120
    print(f"\n  Generating {duration}s audio ...")
    params = GenerationParams(
        task_type="text2music",
        caption=sample.caption,
        lyrics=sample.lyrics,
        bpm=sample.bpm,
        keyscale=sample.keyscale,
        duration=duration,
        inference_steps=50,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
    result = generate_music(
        dit_handler, llm_handler, params, config,
        save_dir=str(output_dir / "demo_a_full_song")
    )
    if result.success:
        print(f"  -> {result.audios[0]['path']}")
    else:
        print(f"  FAILED: {result.error}")

    # ------------------------------------------------------------------
    # Demo B: Instrumental epic (90s)
    # ------------------------------------------------------------------
    print("\n[Demo B] Instrumental epic — cinematic film score style ...")
    sample_b = create_sample(
        llm_handler,
        query="a cinematic film score building from quiet strings to a triumphant orchestral climax",
        instrumental=True,
    )
    if not sample_b.success:
        print(f"  LM blueprint generation failed: {sample_b.error}")
        sys.exit(1)

    duration_b = sample_b.duration if sample_b.duration and sample_b.duration > 60 else 90
    print(f"\n  Generating {duration_b}s instrumental ...")
    params = GenerationParams(
        task_type="text2music",
        caption=sample_b.caption,
        lyrics="[Instrumental]",
        bpm=sample_b.bpm,
        keyscale=sample_b.keyscale,
        duration=duration_b,
        inference_steps=50,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
    result = generate_music(
        dit_handler, llm_handler, params, config,
        save_dir=str(output_dir / "demo_b_instrumental_epic")
    )
    if result.success:
        print(f"  -> {result.audios[0]['path']}")
    else:
        print(f"  FAILED: {result.error}")

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
