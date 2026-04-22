"""
Script 07: Music Understanding

Demonstrates ACE-Step's ability to analyze existing audio and extract metadata:
- Caption (style, instruments, mood)
- Lyrics (if vocals present)
- BPM, key/scale, time signature
- Duration

The model "listens" to the audio by encoding it to semantic tokens,
then the 5Hz LM interprets those tokens into human-readable metadata.

This is the reverse of generation: instead of text -> audio, it's audio -> text.

Models:
- DiT: acestep-v15-sft (for VAE encoding + tokenization)
- LM: acestep-5Hz-lm-1.7B (for interpreting audio codes)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from exiv.components.models.acestep.handler import AceStepHandler
from exiv.components.models.acestep.llm_inference import LLMHandler
from exiv.components.models.acestep.inference import understand_music


def _get_audio_to_analyze() -> str:
    """Return path to the long-form song from Script 06 for analysis."""
    audio_dir = REPO_ROOT / "outputs" / "06_long_form" / "demo_a_full_song"
    path = audio_dir / "9db67b10-ed37-3159-ba36-b3d9a6043ab6.wav"
    if not path.exists():
        files = sorted(audio_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if files:
            path = files[-1]
        else:
            raise FileNotFoundError(
                "Audio file not found. Run scripts/06_long_form_generation.py first."
            )
    return str(path)


def main():
    output_dir = REPO_ROOT / "outputs" / "07_music_understanding"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACE-Step 1.5 - Script 07: Music Understanding")
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

    audio_path = _get_audio_to_analyze()
    print(f"\n[SETUP] Audio to analyze: {audio_path}")
    print("  (This is the 218s progressive rock epic from Script 06)")

    # ------------------------------------------------------------------
    # Step 1: Encode audio to semantic codes
    # ------------------------------------------------------------------
    print("\n[Step 1] Encoding audio to semantic tokens via VAE + quantizer ...")
    audio_codes = dit_handler.convert_src_audio_to_codes(audio_path)
    if audio_codes.startswith("❌"):
        print(f"  FAILED: {audio_codes}")
        sys.exit(1)

    num_codes = audio_codes.count("<|audio_code_")
    print(f"  Generated {num_codes} audio semantic tokens")

    # ------------------------------------------------------------------
    # Step 2: LM interprets the codes into metadata
    # ------------------------------------------------------------------
    print("\n[Step 2] LM interpreting audio codes into metadata ...")
    result = understand_music(
        llm_handler,
        audio_codes=audio_codes,
        temperature=0.85,
        use_constrained_decoding=True,
    )

    if not result.success:
        print(f"  FAILED: {result.error}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("EXTRACTED METADATA")
    print("=" * 60)
    print(f"  Caption        : {result.caption}")
    print(f"  BPM            : {result.bpm}")
    print(f"  Key/Scale      : {result.keyscale}")
    print(f"  Time Signature : {result.timesignature}")
    print(f"  Duration       : {result.duration}s")
    print(f"  Language       : {result.language}")
    print(f"\n  Lyrics ({len(result.lyrics)} chars):")
    # Print lyrics with line wrapping
    lyrics_lines = result.lyrics.split("\n")
    for line in lyrics_lines[:20]:
        print(f"    {line}")
    if len(lyrics_lines) > 20:
        print(f"    ... ({len(lyrics_lines) - 20} more lines)")

    print("\n" + "=" * 60)
    print("GROUND TRUTH (from Script 06 generation)")
    print("=" * 60)
    print("  Query          : emotional progressive rock epic about chasing dreams")
    print("  Original BPM   : (set by LM during generation)")
    print("  Original Key   : (set by LM during generation)")
    print("  Original Dur   : 218s")
    print("  Note           : The extracted metadata should roughly match the")
    print("                   style and structure of the original prompt.")
    print("=" * 60)


if __name__ == "__main__":
    main()
