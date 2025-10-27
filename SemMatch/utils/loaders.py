"""
Module: utils.datasets
-----------------------

This module provides utilities for downloading and extracting archive files of various formats
including .zip, .tar, .tar.gz, and .7z. It defines functions for downloading files from a URL
with a progress bar and extracting different types of compressed archives.

Functions:
----------
    download_from_url(url, archive_path, chunk_size=1024)
        Downloads a file from the specified URL to a local path with a progress bar.

    extract_archive(archive_path, output_dir, remove_after_extraction=False) -> List[Path]
        Extracts the contents of an archive file into a specified directory and optionally
        deletes the archive after extraction.
"""

import os
import zipfile
import tarfile
import urllib.request

from pathlib import Path
from typing import Union, List

import py7zr
from tqdm import tqdm


def download_from_url(
    url: str,
    archive_path: Union[str, Path],
    chunk_size: int = 1024
) -> None:
    """
    Downloads a file from the specified URL to the given local path.

    Parameters:
    ----------
    url : str
        The URL of the file to download.
    archive_path : str or Path
        The destination path to save the downloaded file.
    chunk_size : int, optional
        The size of each chunk to read during download (default is 1024 bytes).
    """
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get("Content-Length", 0))

        with open(archive_path, 'wb') as out_file, \
            tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading",
                ncols=80
        ) as progress_bar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                progress_bar.update(len(chunk))


def _extract_with_top_folder_strip(
    archive_members: list,
    read_member_func,
    output_dir: Path
) -> List[Path]:
    """
    Helper method to extract files from archive members,
    stripping the top-level folder if present.

    Parameters:
    ----------
    archive_members : list
        List of archive members (files) to extract.
    read_member_func : callable
        Function to read a member and return bytes.
        Signature: read_member_func(member) -> bytes
    output_dir : Path
        Directory where to extract files.

    Returns:
    -------
    List[Path]
        List of extracted file paths.
    """
    extracted_files = []

    # Detect common top-level folder
    member_names = [m for m in archive_members if m]  # skip empty
    if not member_names:
        return extracted_files

    top_folder = os.path.commonprefix(member_names).split('/')[0]

    for member_name in member_names:
        # Remove top-level folder if present
        filename = member_name
        if top_folder and filename.startswith(top_folder + '/'):
            filename = filename[len(top_folder)+1:]

        if not filename:
            continue

        target_path = output_dir / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        data = read_member_func(member_name)
        with open(target_path, 'wb') as f:
            f.write(data)

        extracted_files.append(target_path)

    return extracted_files


def extract_archive(
    archive_path: Union[str, Path],
    output_dir: Union[str, Path],
    remove_after_extraction: bool = False
) -> List[Path]:
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = []

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            members = [m.filename for m in zip_ref.infolist()
                       if not m.is_dir()]
            extracted_files = _extract_with_top_folder_strip(
                members,
                lambda name: zip_ref.read(name),
                output_dir
            )

    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            members = [m.name for m in tar_ref.getmembers() if m.isfile()]
            extracted_files = _extract_with_top_folder_strip(
                members,
                lambda name: tar_ref.extractfile(name).read(),
                output_dir
            )

    elif archive_path.suffix.lower() == '.7z':
        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            archive.extractall(path=output_dir)
            extracted_files = [output_dir /
                               name for name in archive.getnames()]

    else:
        raise ValueError(
            f"Unsupported archive format: {archive_path.suffixes}")

    if remove_after_extraction:
        archive_path.unlink()

    return extracted_files
