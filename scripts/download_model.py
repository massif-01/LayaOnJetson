"""Download and verify the exact multilingual checkpoint used in the Thor run."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


MANIFEST = Path(__file__).resolve().parents[1] / "requirements/model.json"


def verify_model(directory):
    manifest = json.loads(MANIFEST.read_text())
    for filename, expected in manifest["sha256"].items():
        path = directory / filename
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"Checkpoint checksum mismatch for {filename}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if not args.verify_only:
        for filename in manifest["sha256"]:
            hf_hub_download(
                manifest["repository"], filename, revision=manifest["revision"],
                local_dir=args.directory,
            )
    verify_model(args.directory)
    print(f"Verified {manifest['repository']} at {manifest['revision']}")


if __name__ == "__main__":
    main()
