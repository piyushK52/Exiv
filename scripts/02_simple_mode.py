"""
Script 02: Simple Mode & Format Mode

Demonstrates the 5Hz LM acting as a "Composer Agent":
1. Simple Mode: A natural-language query is expanded into a full song blueprint
   (caption, lyrics, BPM, key, duration, language) via create_sample().
2. Format Mode: Raw user caption/lyrics are enhanced and structured metadata is
   auto-filled via format_sample().

Models:
- DiT: acestep-v15-sft
- LM: acestep-5Hz-lm-1.7B
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from exiv.components.models.acestep.llm_inference import LLMHandler
from exiv.components.models.acestep.inference import (
    create_sample,
    format_sample,
    GenerationParams,
    GenerationConfig,
    generate_music,
)
from exiv.components.models.acestep.handler import AceStepHandler


def main():
    output_dir = REPO_ROOT / "outputs" / "prune2" / "02_simple_mode"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACE-Step 1.5 - Script 02: Simple Mode & Format Mode")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Initialize handlers
    # ------------------------------------------------------------------
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    print("\n[1/4] Initializing DiT (acestep-v15-sft) ...")
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
    # Demo A: Simple Mode (create_sample)
    # ------------------------------------------------------------------
    print("\n[3/4] Demo A: Simple Mode — generating blueprint from query ...")
    sample_result = create_sample(
        llm_handler=llm_handler,
        query="a soft acoustic ballad about autumn rain",
        instrumental=False,
        vocal_language="en",
        temperature=0.85,
    )

    if sample_result.success:
        print("Generated blueprint:")
        print(f"  Caption : {sample_result.caption}")
        print(f"  BPM     : {sample_result.bpm}")
        print(f"  Key     : {sample_result.keyscale}")
        print(f"  Duration: {sample_result.duration}s")
        print(f"  Language: {sample_result.language}")
        print(f"  Lyrics preview: {sample_result.lyrics[:120].replace(chr(10), ' ')} ...")

        params = GenerationParams(
            task_type="text2music",
            caption=sample_result.caption,
            lyrics=sample_result.lyrics,
            bpm=sample_result.bpm,
            duration=sample_result.duration,
            keyscale=sample_result.keyscale,
            vocal_language=sample_result.language,
            inference_steps=50,
            thinking=True,
        )
        config = GenerationConfig(
            batch_size=1,
            audio_format="wav",
            use_random_seed=True,
        )

        gen_result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
            save_dir=str(output_dir / "demo_a_simple_mode"),
        )

        if gen_result.success:
            print("Audio generated:")
            for audio in gen_result.audios:
                print(f"  -> {audio['path']}")
        else:
            print(f"Audio generation failed: {gen_result.error}")
    else:
        print(f"Simple mode failed: {sample_result.error}")

    # ------------------------------------------------------------------
    # Demo B: Format Mode (format_sample)
    # ------------------------------------------------------------------
    print("\n[4/4] Demo B: Format Mode — enhancing raw user input ...")
    raw_caption = "lo-fi hip hop, chill beats"
    raw_lyrics = """Verse:
Rainy window, coffee cold
Another night I'm feeling old
The city hums a lullaby
Underneath the neon sky

Chorus:
Just let the beat play on and on
Until the quiet of the dawn
"""

    format_result = format_sample(
        llm_handler=llm_handler,
        caption=raw_caption,
        lyrics=raw_lyrics,
        temperature=0.85,
    )

    if format_result.success:
        print("Enhanced input:")
        print(f"  Caption : {format_result.caption}")
        print(f"  BPM     : {format_result.bpm}")
        print(f"  Key     : {format_result.keyscale}")
        print(f"  Duration: {format_result.duration}s")
        print(f"  Language: {format_result.language}")
        print(f"  Lyrics preview: {format_result.lyrics[:120].replace(chr(10), ' ')} ...")

        params = GenerationParams(
            task_type="text2music",
            caption=format_result.caption,
            lyrics=format_result.lyrics,
            bpm=format_result.bpm,
            duration=format_result.duration,
            keyscale=format_result.keyscale,
            vocal_language=format_result.language,
            inference_steps=50,
            thinking=True,
        )
        config = GenerationConfig(
            batch_size=1,
            audio_format="wav",
            use_random_seed=True,
        )

        gen_result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
            save_dir=str(output_dir / "demo_b_format_mode"),
        )

        if gen_result.success:
            print("Audio generated:")
            for audio in gen_result.audios:
                print(f"  -> {audio['path']}")
        else:
            print(f"Audio generation failed: {gen_result.error}")
    else:
        print(f"Format mode failed: {format_result.error}")

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
