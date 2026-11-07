#!/usr/bin/env python3
"""
initializes shavian conversion tool
"""

import os
import sys
import requests
from tqdm import tqdm
from urllib.parse import urlparse
import time
import lzma
import zipfile
import tarfile
import gzip
import bz2
import hashlib

# Hardcoded dictionary of URLs, paths, and descriptions.
# Files with a "sha256" are pinned: a download that doesn't match is rejected.
# The dictionaries aren't pinned because dechifro adds words to them regularly.
# shaw.py isn't downloaded: src/shaw.py is a vendored copy tested with this pipeline.
DOWNLOADS = {
    "https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble/resolve/main/model.pt": {
        "path": "./models/model.pt",
        "sha256": "200141c44d4153662050531905ec3354d1d6302c28939dbbb019c037250ed610",
        "description": "Machine vision model for identifying and cutting out speech bubbles.",
        "extract": False
    },
    "https://dechifro.org/shavian/amer.dict.xz": {
        "path": "./dict/tmp/amer.dict.xz",
        "description": "English -> Shavian phonemes as pronounced by an American",
        "extract": True,
        "extract_to": "./dict/"
    },
    "https://dechifro.org/shavian/brit.dict": {
        "path": "./dict/brit.dict",
        "description": "English -> Shavian phonemes as pronounced by a Brit (well, with an RP accent)",
        "extract": False
    },
    "https://2gd4.me/tidbit/shavian-fonts/backlogshavian.otf": {
        "path": "./fonts/backlogshavian.otf",
        "sha256": "9575619c156fadd005dc2aac065c3a780738c86563ac5c4d3ccd390b038339ec",
        "description": "Font by Haley Halcyon ",
        "extract": False
    },
    "https://2gd4.me/tidbit/shavian-fonts/stampatello-faceto.otf": {
        "path": "./fonts/stampatello-faceto.otf",
        "sha256": "837b02be56e4da23b4757eb16656a6c9dac4a4174e2a403cda7a49016fa2dbbe",
        "description": "Font by Haley Halcyon ",
        "extract": False
    },
    "https://2gd4.me/tidbit/shavian-fonts/bantershaviantriline.otf": {
        "path": "./fonts/bantershaviantriline.otf",
        "sha256": "e8aef26f4518db8b0f958ef0e4ddd9dacd058d8efa102a8572b906a211d34865",
        "description": "Font by Haley Halcyon ",
        "extract": False
    },
    # format if more dependencies necessary (like hopefully a better ocr)
    # "url": {
    #     "path": "destination/as/path",
    #     "description": "figure it out"
    # }
}

def extract_file(file_path, extract_to=None):
    """Extract compressed files based on their extension."""
    if extract_to is None:
        extract_to = os.path.dirname(file_path)

    create_directory(extract_to)
    file_name = os.path.basename(file_path)

    try:
        print(f"▸ Extracting {file_name}...")

        if file_path.endswith('.xz'):
            # Handle .xz files
            with lzma.LZMAFile(file_path, 'rb') as compressed_file:
                extracted_name = file_name[:-3]  # Remove .xz extension
                extracted_path = os.path.join(extract_to, extracted_name)

                with open(extracted_path, 'wb') as output_file:
                    # Read in chunks for large files
                    while chunk := compressed_file.read(8192):
                        output_file.write(chunk)

                print(f"✓ Extracted to: {extracted_path}")
                return extracted_path

        elif file_path.endswith('.tar.xz') or file_path.endswith('.txz'):
            # Handle tar.xz files
            with tarfile.open(file_path, 'r:xz') as tar:
                tar.extractall(path=extract_to)
                members = tar.getnames()
                print(f"✓ Extracted {len(members)} files to: {extract_to}")
                return extract_to

        elif file_path.endswith('.zip'):
            # Handle zip files
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                print(f"✓ Extracted zip to: {extract_to}")
                return extract_to

        elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            # Handle tar.gz files
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=extract_to)
                members = tar.getnames()
                print(f"✓ Extracted {len(members)} files to: {extract_to}")
                return extract_to

        elif file_path.endswith('.tar.bz2') or file_path.endswith('.tbz2'):
            # Handle tar.bz2 files
            with tarfile.open(file_path, 'r:bz2') as tar:
                tar.extractall(path=extract_to)
                members = tar.getnames()
                print(f"✓ Extracted {len(members)} files to: {extract_to}")
                return extract_to

        elif file_path.endswith('.tar'):
            # Handle tar files
            with tarfile.open(file_path, 'r:') as tar:
                tar.extractall(path=extract_to)
                members = tar.getnames()
                print(f"✓ Extracted {len(members)} files to: {extract_to}")
                return extract_to

        elif file_path.endswith('.gz'):
            # Handle gzip files (single file compression)
            with gzip.open(file_path, 'rb') as compressed_file:
                extracted_name = file_name[:-3]  # Remove .gz extension
                extracted_path = os.path.join(extract_to, extracted_name)

                with open(extracted_path, 'wb') as output_file:
                    while chunk := compressed_file.read(8192):
                        output_file.write(chunk)

                print(f"✓ Extracted to: {extracted_path}")
                return extracted_path

        elif file_path.endswith('.bz2'):
            # Handle bzip2 files
            with bz2.BZ2File(file_path, 'rb') as compressed_file:
                extracted_name = file_name[:-4]  # Remove .bz2 extension
                extracted_path = os.path.join(extract_to, extracted_name)

                with open(extracted_path, 'wb') as output_file:
                    while chunk := compressed_file.read(8192):
                        output_file.write(chunk)

                print(f"✓ Extracted to: {extracted_path}")
                return extracted_path

        else:
            print(f"▲ Unsupported archive format: {file_name}")
            return None

    except Exception as e:
        print(f"✗ Error extracting {file_name}: {e}")
        return None
    """Create directory if it doesn't exist."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def sha256_of(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def create_directory(file_path):
    """Create directory if it doesn't exist."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def download_file(url, file_path, description, should_extract=False, extract_to=None, sha256=None):
    """Download a file with progress bar."""
    try:
        print(f"\n▶ {description}")
        print(f"▸ URL: {url}")
        print(f"▸ Path: {file_path}")

        # Create directory if needed
        create_directory(file_path)

        # Start the download
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get file size
        total_size = int(response.headers.get('content-length', 0))

        # Setup progress bar
        progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {os.path.basename(file_path)}",
            ncols=80
        )

        # Download and write file
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()

        if sha256 and sha256_of(file_path) != sha256:
            os.remove(file_path)
            print(f"✗ Checksum mismatch for {file_path}: the file changed upstream. Removed it;")
            print("  check the new version and update its sha256 in init.py if it's fine.")
            return False

        print(f"✓ Successfully downloaded: {file_path}")

        # Extract if requested
        if should_extract:
            extracted_path = extract_file(file_path, extract_to)
            if extracted_path:
                print("▸ Archive extracted successfully")
                # Optionally remove the archive after extraction
                os.remove(file_path)
                print(f"▸ Removed archive: {file_path}")

        return True

    except requests.RequestException as e:
        print(f"✗ Error downloading {url}: {e}")
        return False
    except IOError as e:
        print(f"✗ Error writing file {file_path}: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Main function to download all files."""
    # All paths in DOWNLOADS are relative to this script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("▸ File Downloader Started")
    print("=" * 50)

    if not DOWNLOADS:
        print("✗ No downloads configured!")
        return

    successful = 0
    failed = 0

    for url, config in DOWNLOADS.items():
        file_path = config["path"]
        description = config["description"]
        should_extract = config.get("extract", False)
        extract_to = config.get("extract_to", None)
        sha256 = config.get("sha256")

        # Already present and verified: nothing to do
        if sha256 and os.path.exists(file_path) and sha256_of(file_path) == sha256:
            print(f"✓ {file_path} already present (checksum OK)")
            successful += 1
            continue

        # Check if file already exists
        if os.path.exists(file_path):
            try:
                response = input(f"\n▲ File '{file_path}' already exists. Overwrite? (y/N): ").strip().lower()
            except EOFError:  # non-interactive: keep existing files
                response = ""
            if response not in ['y', 'yes']:
                print(f"▷ Skipping {file_path}")
                continue

        # Download the file
        if download_file(url, file_path, description, should_extract, extract_to, sha256):
            successful += 1
        else:
            failed += 1

        # Small delay between downloads to be respectful
        if len(DOWNLOADS) > 1:
            time.sleep(0.5)

    # Summary
    print("\n" + "=" * 50)
    print("▸ Download Summary:")
    print(f"   ✓ Successful: {successful}")
    print(f"   ✗ Failed: {failed}")
    print(f"   ▸ Total files: {len(DOWNLOADS)}")

    if failed == 0:
        print("◆ All downloads completed successfully!")
    else:
        print("▲ Some downloads failed. Check the errors above.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n◾ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)
