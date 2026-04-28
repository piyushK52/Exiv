"""ACE-Step checkpoint download utilities."""

from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from exiv.utils.logging import app_logger


def ensure_acestep_checkpoint(
    repo_id: str,
    local_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """Download an ACE-Step checkpoint from HuggingFace Hub to a local directory.

    Args:
        repo_id: HuggingFace Hub repo ID (e.g. "ACE-Step/acestep-v15-turbo").
        local_dir: Exact local directory to download to. If the directory already
            exists and is non-empty, download is skipped.
        cache_dir: Fallback cache directory if local_dir is not provided.

    Returns:
        Path to the local checkpoint directory.
    """
    target = Path(local_dir) if local_dir else Path(cache_dir) / repo_id.replace("/", "--")
    if target.exists() and any(target.iterdir()):
        app_logger.info(f"Checkpoint already exists at {target}, skipping download.")
        return target

    app_logger.info(f"Downloading {repo_id} to {target} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    app_logger.info(f"Download complete: {target}")
    return target
