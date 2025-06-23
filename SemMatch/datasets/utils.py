import py7zr
import zipfile
import tarfile

from pathlib import Path
from typing import Union, List

def extract_archive(
    archive_path: Union[str, Path],
    output_dir: Union[str, Path],
    remove_after_extraction: bool = False
) -> List[Path]:
    """
    Extracts an archive (.zip, .tar, .tar.gz, .7z, etc.) to a given directory.

    Args:
        archive_path (str | Path): Path to the archive file.
        output_dir (str | Path): Directory where files will be extracted.
        remove_after_extraction (bool): Whether to delete the archive after extraction.

    Returns:
        List[Path]: List of extracted file paths.
    """
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(output_dir)

    elif archive_path.suffix.lower() == '.7z':
        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            archive.extractall(path=output_dir)

    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffixes}")

    if remove_after_extraction:
        archive_path.unlink()