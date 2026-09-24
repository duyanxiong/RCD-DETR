"""Fail fast if the public subtree contains private or inconsistent artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PUBLIC_DIR = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".pyc"}
FORBIDDEN_NAMES = {"train.py", "trainrdd.py", "ultralytics"}
REQUIRED_FILES = {
    "README.md",
    "requirements.txt",
    "val.py",
    "configs/rdd2022.yaml",
    "tools/export_for_release.py",
    "tools/check_release.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the GitHub release subtree.")
    parser.add_argument("--require-model", action="store_true", help="fail if the TorchScript artifact is absent")
    args = parser.parse_args()

    errors: list[str] = []
    present = {path.relative_to(PUBLIC_DIR).as_posix() for path in PUBLIC_DIR.rglob("*") if path.is_file()}
    for required in sorted(REQUIRED_FILES - present):
        errors.append(f"missing required file: {required}")

    for path in PUBLIC_DIR.rglob("*"):
        relative = path.relative_to(PUBLIC_DIR)
        lowered_parts = {part.lower() for part in relative.parts}
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"private checkpoint/cache must not be published: {relative}")
        if lowered_parts & FORBIDDEN_NAMES:
            errors.append(f"private source path must not be published: {relative}")

    model_path = PUBLIC_DIR / "models" / "best.torchscript"
    metadata_path = PUBLIC_DIR / "models" / "model_metadata.json"
    if args.require_model and not model_path.is_file():
        errors.append("missing release model: models/best.torchscript")
    if model_path.is_file():
        if not metadata_path.is_file():
            errors.append("release model exists but models/model_metadata.json is missing")
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            actual = sha256(model_path)
            if metadata.get("sha256") != actual:
                errors.append("models/best.torchscript SHA-256 does not match model_metadata.json")

    if errors:
        print("Release check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Release check passed ({len(present)} files inspected).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
