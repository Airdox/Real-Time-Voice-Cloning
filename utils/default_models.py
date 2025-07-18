import urllib.request
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from typing import Dict, Tuple, List, Optional
import hashlib
import logging

from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_models = {
    "encoder": (
        "https://drive.google.com/uc?export=download&id=1q8mEGwCkFy23KZsinbuvdKAQLqNKbYf1",
        17090379,
    ),
    "synthesizer": (
        "https://drive.google.com/u/0/uc?id=1EqFMIbvxffxtjiVrtykroF6_mUh-5Z3s&export=download&confirm=t",
        370554559,
    ),
    "vocoder": (
        "https://drive.google.com/uc?export=download&id=1cf2NO6FtI0jDuy8AV3Xgn6leO6dHjIgu",
        53845290,
    ),
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads with custom update method."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        """Update progress bar with download progress."""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, target: Path, bar_pos: int = 0) -> bool:
    """Download a file from URL to target path with progress bar.

    Args:
        url: Download URL
        target: Target file path
        bar_pos: Position of progress bar

    Returns:
        True if download successful, False otherwise
    """
    try:
        # Ensure the directory exists
        target.parent.mkdir(exist_ok=True, parents=True)

        desc = f"Downloading {target.name}"
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=desc, position=bar_pos, leave=False
        ) as t:
            urllib.request.urlretrieve(url, filename=target, reporthook=t.update_to)
        return True
    except HTTPError as e:
        logger.error(f"HTTP error downloading {target.name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error downloading {target.name}: {e}")
        return False


def verify_file_integrity(file_path: Path, expected_size: int) -> bool:
    """Verify file exists and has expected size.

    Args:
        file_path: Path to file to check
        expected_size: Expected file size in bytes

    Returns:
        True if file is valid, False otherwise
    """
    if not file_path.exists():
        return False

    actual_size = file_path.stat().st_size
    if actual_size != expected_size:
        logger.warning(
            f"File {file_path.name} size mismatch. Expected: {expected_size}, "
            f"Actual: {actual_size}"
        )
        return False

    return True


def ensure_default_models(models_dir: Path) -> None:
    """Ensure all default models are downloaded and valid.

    Args:
        models_dir: Directory to store models
    """
    # Define download tasks
    jobs: List[Tuple[Thread, Path, int]] = []

    for model_name, (url, size) in default_models.items():
        target_path = models_dir / "default" / f"{model_name}.pt"

        if verify_file_integrity(target_path, size):
            logger.info(f"Model {model_name} already exists and is valid")
            continue

        if target_path.exists():
            logger.info(f"File {target_path} is invalid, redownloading...")
            target_path.unlink()  # Remove invalid file

        logger.info(f"Downloading {model_name} model...")
        thread = Thread(target=download, args=(url, target_path, len(jobs)))
        thread.start()
        jobs.append((thread, target_path, size))

    # Wait for all downloads to complete
    for thread, target_path, size in jobs:
        thread.join()

        if not verify_file_integrity(target_path, size):
            error_msg = (
                f"Download for {target_path.name} failed. You may download models manually instead.\n"
                f"https://drive.google.com/drive/folders/1fU6umc5uQAVR2udZdHX-lDgXYzTyqG_j"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Successfully downloaded {target_path.name}")
