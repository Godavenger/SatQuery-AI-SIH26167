"""Integrity and destination safety of downloadable runtime artifacts."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.runtime_assets import install_archive, read_manifest, safe_path, sha256, verify


def make_bundle(tmp_path, files=None):
    files = files or {"models/example.pth": b"weights", "data/raw/manifest.json": b"{}"}
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
    manifest = {"schema_version": 1, "size": archive.stat().st_size,
                "sha256": sha256(archive), "files": [
                    {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in files.items()]}
    return archive, manifest


def test_install_verify_and_repeat(tmp_path):
    archive, manifest = make_bundle(tmp_path)
    destination = tmp_path / "installed"
    install_archive(archive, manifest, destination)
    assert verify(manifest, destination) == []
    install_archive(archive, manifest, destination)
    assert verify(manifest, destination) == []


def test_checksum_failure_does_not_write(tmp_path):
    archive, manifest = make_bundle(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"corrupted")
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="checksum"):
        install_archive(archive, manifest, destination)
    assert not destination.exists()


def test_changed_files_preserved_unless_overwrite(tmp_path):
    archive, manifest = make_bundle(tmp_path)
    destination = tmp_path / "installed"
    target = destination / "models/example.pth"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user model")
    with pytest.raises(ValueError, match="Existing asset differs"):
        install_archive(archive, manifest, destination)
    assert target.read_bytes() == b"user model"
    assert not (destination / "data/raw/manifest.json").exists()
    install_archive(archive, manifest, destination, overwrite=True)
    assert verify(manifest, destination) == []


@pytest.mark.parametrize("name", ["../escape", "/models/a", "models/../../escape", "C:/models/a",
                                      "models\\a", "satquery/api.py", "models//a", "models/./a"])
def test_unsafe_archive_paths_rejected(name):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_path(name)


def test_extra_archive_member_rejected(tmp_path):
    archive, manifest = make_bundle(tmp_path)
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("models/unlisted.pth", b"extra")
    manifest.update(size=archive.stat().st_size, sha256=sha256(archive))
    with pytest.raises(ValueError, match="do not match"):
        install_archive(archive, manifest, tmp_path / "installed")


def test_duplicate_manifest_rejected(tmp_path):
    _, manifest = make_bundle(tmp_path)
    manifest["files"].append(manifest["files"][0])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(path)


def test_bad_file_digest_leaves_existing_files_untouched(tmp_path):
    archive, manifest = make_bundle(tmp_path)
    manifest["files"][-1]["sha256"] = "0" * 64
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="Asset checksum"):
        install_archive(archive, manifest, destination)
    assert not (destination / "models").exists()
