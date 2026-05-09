"""Robust dataset download helper with MD5 verification.

Datasets are downloaded with a stream-to-disk + chunked-MD5 pattern to
avoid blowing up memory on multi-GB archives. We deliberately do **not**
hardcode mirror URLs here because mirrors change over the lifetime of a
project; instead, the user is expected to pass the download URL via the
config / CLI.

Automotive SiL parallel:
    Same pattern as the artefact registries used for ADAS test
    datasets: every binary is hash-verified before it can flow into the
    training pipeline.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def md5_of(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream-compute the MD5 hex digest of a file."""
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def download_with_progress(url: str, dest: Path) -> None:
    """Stream-download ``url`` to ``dest`` with a Rich progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    columns = (
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )
    with urllib.request.urlopen(url) as resp, Progress(*columns) as progress:  # noqa: S310
        total_bytes_str = resp.headers.get("Content-Length")
        total = int(total_bytes_str) if total_bytes_str else None
        task = progress.add_task(f"Downloading {dest.name}", total=total)
        with tmp.open("wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                progress.update(task, advance=len(chunk))

    shutil.move(tmp, dest)


def fetch_archive(
    url: str,
    dest: Path,
    *,
    expected_md5: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Download a file (with caching) and verify its MD5.

    Args:
        url: Source URL.
        dest: Destination path.
        expected_md5: Optional MD5 checksum. If provided, the file is
            re-downloaded when the cached copy doesn't match.
        overwrite: Force re-download even if the cached MD5 matches.

    Returns:
        Path to the verified local file.
    """
    if dest.exists() and not overwrite:
        if expected_md5 is None or md5_of(dest) == expected_md5:
            return dest
    download_with_progress(url, dest)
    if expected_md5 is not None:
        actual = md5_of(dest)
        if actual != expected_md5:
            msg = (
                f"MD5 mismatch for {dest.name}: expected {expected_md5}, got {actual}. "
                "Re-run with --overwrite to retry."
            )
            raise RuntimeError(msg)
    return dest
