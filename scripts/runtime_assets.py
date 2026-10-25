"""Build, download and verify the versioned SatQuery inference asset bundle.

Uses only the Python standard library. Large artifacts live in GitHub Releases;
the checked-in runtime-assets.json pins every file and the archive by SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runtime-assets.json"
REPO = "Godavenger/SatQuery-AI-SIH26167"
MISSION_SITES = ("assam", "kaziranga", "delhi", "sundarbans", "punjab",
                 "gangotri", "rondonia", "creekfire", "nile", "dubai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(name: str) -> Path:
    p = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or p.is_absolute()
            or ".." in p.parts or p.as_posix() != name
            or not (name.startswith("models/") or name.startswith("data/raw/"))):
        raise ValueError(f"Unsafe asset path: {name}")
    return Path(*p.parts)


def read_manifest(path: Path = MANIFEST) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("Unsupported or empty asset manifest")
    seen = set()
    for item in manifest["files"]:
        safe_path(item["path"])
        key = item["path"].casefold()
        if key in seen:
            raise ValueError(f"Duplicate asset: {item['path']}")
        seen.add(key)
        if not isinstance(item["size"], int) or item["size"] < 0:
            raise ValueError("Invalid asset size")
        if len(item["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in item["sha256"]):
            raise ValueError("Invalid SHA-256")
    return manifest


def asset_target(root: Path, name: str) -> Path:
    target = root / safe_path(name)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes destination: {name}")
    return target


def verify(manifest: dict, destination: Path) -> list[str]:
    issues = []
    for item in manifest["files"]:
        path = asset_target(destination, item["path"])
        if not path.is_file():
            issues.append(f"missing: {item['path']}")
        elif path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            issues.append(f"changed: {item['path']}")
    return issues


def install_archive(archive: Path, manifest: dict, destination: Path,
                    overwrite: bool = False) -> None:
    if archive.stat().st_size != manifest["size"] or sha256(archive) != manifest["sha256"]:
        raise ValueError("Archive checksum mismatch; nothing was installed")
    expected = {entry["path"]: entry for entry in manifest["files"]}
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory(
            prefix="satquery-assets-", dir=destination) as temporary:
        members = bundle.infolist()
        if len(members) != len(expected) or {m.filename for m in members} != set(expected):
            raise ValueError("Archive files do not match the asset manifest")
        staging = Path(temporary)
        # Validate every entry and existing destination before replacing any file.
        for member in members:
            item = expected[member.filename]
            safe_path(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16) or member.file_size != item["size"]:
                raise ValueError(f"Invalid archive entry: {member.filename}")
            target = asset_target(destination, member.filename)
            if target.exists() and (not target.is_file() or
                    (sha256(target) != item["sha256"] and not overwrite)):
                raise ValueError(f"Existing asset differs: {member.filename}; use --overwrite to replace it")
            staged = asset_target(staging, member.filename)
            staged.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, staged.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            if sha256(staged) != item["sha256"]:
                raise ValueError(f"Asset checksum mismatch: {member.filename}")
        for item in expected.values():
            target = asset_target(destination, item["path"])
            if target.is_file() and sha256(target) == item["sha256"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            asset_target(staging, item["path"]).replace(target)


def download(manifest: dict, output: Path) -> None:
    url = manifest["url"]
    if not url.startswith(f"https://github.com/{REPO}/releases/download/"):
        raise ValueError("Asset URL must refer to this project's GitHub Releases")
    request = Request(url, headers={"User-Agent": "SatQuery-Assets/1.1"})
    count = 0
    with urlopen(request, timeout=60) as response, output.open("wb") as stream:
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > manifest["size"]:
                raise ValueError("Download exceeds expected archive size")
            stream.write(block)
    if count != manifest["size"]:
        raise ValueError("Incomplete download; run install again")


def build(version: str, minilm: Path, output_dir: Path) -> dict:
    if not version or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for c in version):
        raise ValueError("Invalid release version")
    assets = {
        name: ROOT / name for name in (
            "models/backbone_ft.pth", "models/ben_txt_heads.pth", "models/router_v2.pkl",
            "models/onnx/opt_feat.int8.onnx", "models/onnx/sar_feat.int8.onnx",
            "models/onnx/caption_head.int8.onnx", "models/onnx/s1_head.int8.onnx")
    }
    for path in minilm.rglob("*"):
        if path.is_file() and ".cache" not in path.relative_to(minilm).parts:
            assets[f"models/minilm/{path.relative_to(minilm).as_posix()}"] = path
    if "models/minilm/model.safetensors" not in assets or "models/minilm/LICENSE" not in assets:
        raise ValueError("MiniLM directory must contain weights, tokenizer, configs and LICENSE")
    local_sites = json.loads((ROOT / "data/raw/manifest.json").read_text(encoding="utf-8"))
    # Publish the reproducible mission corpus, excluding personal ad-hoc pins.
    sites = {name: local_sites[name] for name in MISSION_SITES}
    for site in sites.values():
        for scene in site.get("s2", []) + site.get("s1", []):
            for key in ("path", "scl"):
                if scene.get(key):
                    name = scene[key].replace("\\", "/")
                    safe_path(name)
                    scene[key] = name
                    assets[name] = ROOT / safe_path(name)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"satquery-runtime-{version}.zip"
    files = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as bundle:
        for name, path in sorted(assets.items()):
            safe_path(name)
            print(f"Packing {name}", flush=True)
            files.append({"path": name, "size": path.stat().st_size, "sha256": sha256(path)})
            bundle.write(path, name)
        portable_manifest = (json.dumps(sites, indent=2) + "\n").encode("utf-8")
        name = "data/raw/manifest.json"
        bundle.writestr(name, portable_manifest)
        files.append({"path": name, "size": len(portable_manifest),
                      "sha256": hashlib.sha256(portable_manifest).hexdigest()})
    manifest = {"schema_version": 1, "version": version, "archive": archive.name,
                "url": f"https://github.com/{REPO}/releases/download/{version}/{archive.name}",
                "size": archive.stat().st_size, "sha256": sha256(archive), "files": files}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "SHA256SUMS").write_text(f"{manifest['sha256']}  {archive.name}\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("build", help="Maintainer: pack existing trained runtime artifacts")
    pack.add_argument("--version", required=True)
    pack.add_argument("--minilm-dir", type=Path, required=True)
    pack.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    for name in ("install", "verify"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--destination", type=Path, default=ROOT)
        cmd.add_argument("--manifest", type=Path, default=MANIFEST)
        if name == "install":
            cmd.add_argument("--archive", type=Path, help="Use a previously downloaded archive")
            cmd.add_argument("--overwrite", action="store_true", help="Replace changed runtime artifacts")
    args = parser.parse_args()
    try:
        if args.command == "build":
            manifest = build(args.version, args.minilm_dir, args.output_dir)
            print(f"Built {manifest['archive']} ({manifest['size'] / 1024**2:.1f} MiB)")
            return 0
        manifest = read_manifest(args.manifest)
        if args.command == "install":
            with tempfile.TemporaryDirectory(prefix="satquery-download-") as temporary:
                archive = args.archive
                if archive is None:
                    archive = Path(temporary) / manifest["archive"]
                    print(f"Downloading {manifest['size'] / 1024**2:.1f} MiB from GitHub Releases...", flush=True)
                    download(manifest, archive)
                install_archive(archive, manifest, args.destination, args.overwrite)
        issues = verify(manifest, args.destination)
        if issues:
            print("\n".join(issues))
            return 1
        print(f"Verified {len(manifest['files'])} runtime assets ({manifest['version']}).")
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Asset {args.command} failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
