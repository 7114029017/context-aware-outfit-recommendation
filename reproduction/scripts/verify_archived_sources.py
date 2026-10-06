#!/usr/bin/env python3
"""Check that the original research folders 01-03 are unchanged.

The full-run launcher refuses to start training unless the files in
01_資料建構_data_construction, 02_模型訓練和驗證_model_training_validation and
03_實驗與結果_experiments_results match
reproduction/environment/archived_sources_manifest.json: the path, size and
SHA-256 of every file. The manifest records these folders as published in this
repository: the frozen handoff baseline (commit 7a5cc9c of the original
development repository, the content the official run used) without the
third-party publications removed for the public release, and with the embedded
Polyvore images removed from three notebooks. The removed files and images are
listed in reproduction/docs/public_release_cleanup.md; the reproduction
pipeline reads none of them.

In a git work tree the tracked files are checked; elsewhere every file
under the folders. --write regenerates the manifest from the working tree.
Exit status: 0 when every file matches, 1 when files differ, 2 on errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "reproduction" / "environment" / "archived_sources_manifest.json"
FOLDERS = (
    "01_資料建構_data_construction",
    "02_模型訓練和驗證_model_training_validation",
    "03_實驗與結果_experiments_results",
)
BASELINE = "7a5cc9cd8f884865e1e27a234c18dd132f949598"
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
CHUNK = 8 * 1024 * 1024


def fail(message: str) -> None:
    print("[ARCHIVED SOURCES BLOCKED] " + message, file=sys.stderr)
    raise SystemExit(2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(len(LFS_PREFIX)) == LFS_PREFIX


def listed_files(root: Path, folders: list[str]) -> list[str]:
    """Tracked files under the folders, or every file outside a git work tree."""
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", *folders],
                             check=True, capture_output=True).stdout
        return sorted(p for p in out.decode("utf-8").split("\0") if p)
    except (OSError, subprocess.CalledProcessError):
        return sorted(p.relative_to(root).as_posix() for f in folders
                      for p in (root / f).rglob("*") if p.is_file())


def write(root: Path, manifest_path: Path) -> None:
    records = []
    for path in listed_files(root, list(FOLDERS)):
        file = root / path
        if is_lfs_pointer(file):
            fail(f"Git LFS file not downloaded (run git lfs pull first): {path}")
        records.append({"path": path, "bytes": file.stat().st_size, "sha256": sha256(file)})
    manifest = {
        "description": "Files of the original research folders 01-03 as published in this repository: the "
                       "frozen handoff baseline, which the official run used, without the third-party "
                       "publications removed for the public release and with the embedded Polyvore images "
                       "removed from three notebooks. Checked by reproduction/scripts/verify_archived_sources.py "
                       "before a full run starts.",
        "baseline_commit": BASELINE,
        "baseline_repository": "original development repository (not this public repository)",
        "public_release_changes": "reproduction/docs/public_release_cleanup.md",
        "folders": list(FOLDERS),
        "file_count": len(records),
        "files": records,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[ARCHIVED SOURCES] wrote {len(records)} files to {manifest_path}")


def check(root: Path, manifest_path: Path) -> int:
    if not manifest_path.is_file():
        fail(f"manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {f["path"]: f for f in manifest["files"]}
    present = set(listed_files(root, manifest["folders"]))
    missing = set(expected) - present
    added = sorted(present - set(expected))
    changed, not_downloaded = [], []
    for path in sorted(set(expected) & present):
        file = root / path
        if not file.is_file():
            missing.add(path)
        elif is_lfs_pointer(file) and expected[path]["bytes"] > 1024:
            not_downloaded.append(path)
        elif file.stat().st_size != expected[path]["bytes"] or sha256(file) != expected[path]["sha256"]:
            changed.append(path)
    problems = [("missing", sorted(missing)), ("added", added), ("changed", changed),
                ("Git LFS file not downloaded (run git lfs pull)", not_downloaded)]
    if not any(items for _, items in problems):
        print(f"[ARCHIVED SOURCES] {len(expected)} files in folders 01-03 match {manifest_path.name}")
        return 0
    for label, items in problems:
        for path in items[:20]:
            print(f"[ARCHIVED SOURCES] {label}: {path}", file=sys.stderr)
        if len(items) > 20:
            print(f"[ARCHIVED SOURCES] ... and {len(items) - 20} more {label}", file=sys.stderr)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root (default: this repository).")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--write", action="store_true", help="Regenerate the manifest from the working tree.")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.write:
        write(root, args.manifest)
        return
    raise SystemExit(check(root, args.manifest))


if __name__ == "__main__":
    main()
