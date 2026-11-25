"""Download and verify JFM-2026-1283 release data. All coordinates and numbers are preserved byte for byte."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 4 * 1024 * 1024


def filehash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def obtain(asset, cache):
    path = cache / asset["name"]
    if path.exists() and path.stat().st_size == asset["bytes"] and filehash(path) == asset["sha256"]:
        return path
    cache.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(asset["url"], headers={"User-Agent": "JFM-1283-research-data"})
    with urllib.request.urlopen(request, timeout=180) as source, temporary.open("wb") as sink:
        shutil.copyfileobj(source, sink, length=BLOCK)
    if temporary.stat().st_size != asset["bytes"] or filehash(temporary) != asset["sha256"]:
        raise ValueError(f"Downloaded bytes differ from SHA-256 manifest for {path.name}")
    temporary.replace(path)
    return path


def target_path(dest, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or relative.parts[0] not in ("data", "checksums") or ".." in relative.parts:
        raise ValueError(f"Invalid scientific archive path {name}")
    target = (dest / relative).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise ValueError(f"Escaping scientific archive path {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def data_file(source, dest, name, bytes_expected, digest_expected):
    path = target_path(dest, name)
    if path.exists():
        if path.stat().st_size == bytes_expected and filehash(path) == digest_expected:
            return
        raise FileExistsError(f"Existing file differs, choose an empty destination or move it: {path}")
    temp = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    with temp.open("wb") as sink:
        while chunk := source.read(BLOCK):
            sink.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != digest_expected or temp.stat().st_size != bytes_expected:
        raise ValueError(f"Checksum mismatch for archive member {name}")
    temp.replace(path)


def unpack(file, destination):
    if file.name.endswith(".zip"):
        archive = zipfile.ZipFile(file)
        names = archive.namelist()
        reader = archive.open
    else:
        archive = tarfile.open(file, "r:gz")
        if any(not member.isfile() for member in archive):
            raise ValueError(f"Unexpected archive member type in {file}")
        names = archive.getnames()
        reader = archive.extractfile
    with archive:
        sums = [name for name in names if name.startswith("checksums/")]
        if len(sums) != 1 or len(set(names)) != len(names):
            raise ValueError(f"Invalid member list in {file}")
        with reader(sums[0]) as frame:
            checks = json.load(frame)
        expected = {entry["path"] for entry in checks}
        if set(names) != expected | set(sums):
            raise ValueError(f"Unexpected member in {file}")
        for entry in checks:
            with reader(entry["path"]) as data:
                data_file(data, destination, entry["path"], entry["bytes"], entry["sha256"])
        print(f"Verified {len(checks)} members in {file.name}", flush=True)


def restore_asset(asset, cache, destination):
    encoded = obtain(asset, cache)
    if "relative_path" in asset:
        with encoded.open("rb") as source:
            data_file(source, destination, asset["relative_path"], asset["bytes"], asset["sha256"])
        print(f"Verified movie {asset['name']}", flush=True)
    else:
        unpack(encoded, destination)


def decompressor():
    try:
        from compression import zstd
        return zstd.open
    except ImportError:
        try:
            import zstandard
        except ImportError as exc:
            raise SystemExit("Use Python 3.14 or install zstandard for particle arrays.") from exc
        return zstandard.open


def restore_particle(item, assets, cache, dest):
    path = target_path(dest, item["relative_path"])
    if path.exists():
        if path.stat().st_size == item["bytes"] and filehash(path) == item["sha256"]:
            print(f"Verified existing {path.name}", flush=True)
            return
        raise FileExistsError(f"Existing file differs, choose another destination: {path}")
    temporary = path.with_suffix(".npz.partial")
    total = hashlib.sha256()
    zopen = decompressor()
    with temporary.open("wb") as out:
        for part in item["parts"]:
            encoded = obtain(assets[part["name"]], cache)
            count = 0
            digest = hashlib.sha256()
            with zopen(encoded, "rb") as stream:
                while chunk := stream.read(BLOCK):
                    out.write(chunk)
                    digest.update(chunk)
                    total.update(chunk)
                    count += len(chunk)
            if count != part["decoded_bytes"] or digest.hexdigest() != part["decoded_sha256"]:
                raise ValueError(f"Uncompressed range failed SHA-256 for {part['name']}")
    if temporary.stat().st_size != item["bytes"] or total.hexdigest() != item["sha256"]:
        raise ValueError(f"Reassembled NPZ differs from original {path.name}")
    temporary.replace(path)
    print(f"Verified {item['frames']} frames and the full NPZ in {path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="*", help="Dataset group(s), particle basename(s), particles, or all")
    parser.add_argument("--list", action="store_true", help="Show scientific dataset groups and sizes")
    parser.add_argument("--cache", type=Path, default=ROOT / ".release-cache")
    parser.add_argument("--dest", type=Path, default=ROOT)
    params = parser.parse_args()
    recorded = json.loads((ROOT / "release_assets.json").read_text(encoding="utf-8"))
    particles = json.loads((ROOT / "particle_arrays.json").read_text(encoding="utf-8"))
    assets = {asset["name"]: asset for asset in recorded}
    groups = {}
    for asset in recorded:
        if asset["group"] != "particles":
            groups.setdefault(asset["group"], []).append(asset)
    particle_groups = {item["file"].removesuffix(".npz"): item for item in particles}
    if params.list or not params.groups:
        for name, group_assets in groups.items():
            for asset in group_assets:
                print(f"{name:30} {asset['bytes'] / 1e6:9.1f} MB   {asset['name']}")
        for key, item in particle_groups.items():
            print(f"{key:30} {sum(p['bytes'] for p in item['parts']) / 1e6:9.1f} MB   {item['frames']} frames")
        return
    requested = list(dict.fromkeys(params.groups))
    unknown = set(requested) - set(groups) - set(particle_groups) - {"all", "particles"}
    if unknown:
        parser.error("Unknown dataset group: " + ", ".join(sorted(unknown)))
    for key, group_assets in groups.items():
        if "all" in requested or key in requested:
            for asset in group_assets:
                restore_asset(asset, params.cache, params.dest)
    for key, item in particle_groups.items():
        if "all" in requested or "particles" in requested or key in requested:
            restore_particle(item, assets, params.cache, params.dest)


if __name__ == "__main__":
    main()
