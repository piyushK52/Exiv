"""
Script 01: Text-to-Music Generation

Demonstrates ACE-Step's core text-to-music capability:
1. Basic instrumental generation from a text prompt
2. Song generation with lyrics and explicit metadata (BPM, key, duration)

Models:
- DiT: acestep-v15-turbo (2B, ~4.7GB)
- LM: acestep-5Hz-lm-1.7B (~3.4GB)
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent

from exiv.components.models.acestep.handler import AceStepHandler
from exiv.components.models.acestep.llm_inference import LLMHandler
from exiv.components.models.acestep.inference import GenerationParams, GenerationConfig, generate_music


def main():
    output_dir = REPO_ROOT / "outputs" / "prune2" / "01_text2music"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACE-Step 1.5 - Script 01: Text-to-Music Generation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Initialize handlers
    # ------------------------------------------------------------------
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    print("\n[1/4] Initializing DiT (acestep-v15-turbo) ...")
    status, ok = dit_handler.initialize_service(
          
        config_path="acestep-v15-sft",
        device="cuda",
        use_flash_attention=True,
    )
    if not ok:
        print(f"Failed to initialize DiT: {status}")
        sys.exit(1)
    print(f"DiT initialized: {status}")

    print("\n[2/4] Initializing 5Hz LM (acestep-5Hz-lm-1.7B) ...")
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
    # Demo A: Basic instrumental generation
    # ------------------------------------------------------------------
    print("\n[3/4] Demo A: Generating instrumental music ...")
    params_a = GenerationParams(
        task_type="text2music",
        caption="upbeat electronic dance music with heavy bass and synthesizer leads",
        lyrics="[Instrumental]",
        duration=15,  # short demo clip
        inference_steps=50,
    )
    config_a = GenerationConfig(
        batch_size=3,
        audio_format="wav",
        use_random_seed=True,
    )

    result_a = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params_a,
        config=config_a,
        save_dir=str(output_dir / "demo_a_instrumental"),
    )

    if result_a.success:
        print("Demo A succeeded:")
        for audio in result_a.audios:
            print(f"  -> {audio['path']}")
    else:
        print(f"Demo A failed: {result_a.error}")

    # ------------------------------------------------------------------
    # Demo B: Song generation with lyrics and metadata
    # ------------------------------------------------------------------
    print("\n[4/4] Demo B: Generating song with lyrics and metadata ...")
    params_b = GenerationParams(
        task_type="text2music",
        caption="melancholic indie folk with acoustic guitar and soft vocals",
        lyrics="""Verse 1:
Walking down the empty road
Wind is whispering soft and low
Leaves are falling one by one
Chasing shadows from the sun

Chorus:
I will find my way back home
Through the stars and the unknown
""",
        vocal_language="en",
        bpm=90,
        keyscale="G Major",
        timesignature="4",
        duration=20,
        inference_steps=50,
        thinking=True,
    )
    config_b = GenerationConfig(
        batch_size=1,
        audio_format="wav",
        use_random_seed=True,
    )

    result_b = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params_b,
        config=config_b,
        save_dir=str(output_dir / "demo_b_with_lyrics"),
    )

    if result_b.success:
        print("Demo B succeeded:")
        for audio in result_b.audios:
            print(f"  -> {audio['path']}")
    else:
        print(f"Demo B failed: {result_b.error}")

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
