#!/usr/bin/env python3
"""Build a macOS app bundle from a verified runtime release archive."""

import argparse
import hashlib
import json
import os
import plistlib
import posixpath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

if __package__:
    from .package import digest, macos_major
else:
    from package import digest, macos_major

ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "Splash M1.app"
APP_ID = "io.github.paulouski.splash-m1"
APP_EXECUTABLE = "SplashM1"
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
APP_VERSION_PATTERN = re.compile(
    r"v?([0-9]+)\.([0-9]+)\.([0-9]+)(?:[-+][A-Za-z0-9.-]+)?"
)


def release_name(version, macos_major_version):
    return f"splash-m1-{version}-arm64-macos{macos_major_version}"


def app_version_number(version):
    """Use a numeric Apple bundle version; local non-semantic tags become 0.0.0."""
    match = APP_VERSION_PATTERN.fullmatch(version)
    return ".".join(match.groups()) if match else "0.0.0"


def read_checksum(path):
    try:
        fields = path.read_text().split()
    except OSError as error:
        raise ValueError(f"missing runtime checksum: {path}") from error
    if not fields or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        raise ValueError(f"invalid runtime checksum: {path}")
    return fields[0]


def member_path(member, root_name):
    name = member.name
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe runtime archive path: {name!r}")
    parts = name.split("/")
    if member.isdir() and parts[-1] == "":
        parts.pop()
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe runtime archive path: {name!r}")
    if parts[0] != root_name:
        raise ValueError(f"runtime archive contains unexpected path: {name!r}")
    return PurePosixPath(*parts)


def validate_link(member, relative_path, root_name):
    target = member.linkname
    if not target or target.startswith("/") or "\\" in target or "\x00" in target:
        raise ValueError(f"unsafe runtime archive link: {member.name!r}")
    if member.issym():
        resolved = posixpath.normpath(posixpath.join(str(relative_path.parent), target))
    else:
        resolved = posixpath.normpath(target)
    if resolved == "." or not (
        resolved == root_name or resolved.startswith(root_name + "/")
    ):
        raise ValueError(f"runtime archive link escapes its release: {member.name!r}")


def validate_members(archive, root_name):
    members = archive.getmembers()
    indexed = {}
    for member in members:
        path = member_path(member, root_name)
        key = path.as_posix()
        if key in indexed:
            raise ValueError(f"duplicate runtime archive path: {key}")
        if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
            raise ValueError(f"unsupported runtime archive entry: {member.name!r}")
        if member.issym() or member.islnk():
            validate_link(member, path, root_name)
        indexed[key] = member
    root = indexed.get(root_name)
    if root is None or not root.isdir():
        raise ValueError("runtime archive is missing its release directory")
    return members, indexed


def read_member_json(archive, member):
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError("runtime release manifest is not a regular file")
    try:
        with stream:
            return json.load(stream)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("runtime release manifest is invalid") from error


def digest_member(archive, member):
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"runtime release entry is not a regular file: {member.name}")
    with stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_runtime_archive(archive_path, version, macos_min, major, destination):
    expected_root = release_name(version, major)
    expected_filename = expected_root + ".tar.gz"
    if archive_path.name != expected_filename:
        raise ValueError(f"runtime archive must be named {expected_filename}")
    if not archive_path.is_file():
        raise ValueError(f"missing runtime archive: {archive_path}")
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    expected_checksum = read_checksum(checksum_path)
    if digest(archive_path) != expected_checksum:
        raise ValueError("runtime archive checksum mismatch")

    try:
        archive = tarfile.open(archive_path, "r:gz")
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"cannot read runtime archive: {archive_path}") from error
    with archive:
        members, indexed = validate_members(archive, expected_root)
        manifest_member = indexed.get(f"{expected_root}/release.json")
        if manifest_member is None or not manifest_member.isfile():
            raise ValueError("runtime archive is missing release.json")
        metadata = read_member_json(archive, manifest_member)
        if not isinstance(metadata, dict):
            raise ValueError("runtime release manifest must contain a JSON object")
        expected_metadata = {
            "product": "splash-m1",
            "version": version,
            "architecture": "arm64",
            "minimum_macos": macos_min,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"runtime release {key} mismatch: expected {expected!r}"
                )

        required_files = (
            "engine/splash",
            "engine/splash.metallib",
            "install/desktop.py",
            "install/launcher.py",
            "python/bin/python3",
        )
        for relative in required_files:
            member = indexed.get(f"{expected_root}/{relative}")
            if member is None or not (member.isfile() or member.issym()):
                raise ValueError(f"runtime archive is missing {relative}")

        for relative, key in (
            ("engine/splash", "binary_sha256"),
            ("engine/splash.metallib", "metallib_sha256"),
        ):
            member = indexed[f"{expected_root}/{relative}"]
            expected = metadata.get(key)
            if not isinstance(expected, str) or not re.fullmatch(
                r"[0-9a-f]{64}", expected
            ):
                raise ValueError(f"runtime release has invalid {key}")
            if digest_member(archive, member) != expected:
                raise ValueError(f"runtime release {relative} checksum mismatch")

        archive.extractall(destination, members=members, filter="data")

    return destination / expected_root


def write_info_plist(template, destination, version, macos_min):
    try:
        with template.open("rb") as source:
            info = plistlib.load(source)
    except (OSError, plistlib.InvalidFileException) as error:
        raise ValueError(f"cannot read app Info.plist template: {template}") from error
    if not isinstance(info, dict):
        raise ValueError("app Info.plist template must contain a dictionary")
    bundle_version = app_version_number(version)
    info.update(
        {
            "CFBundleDevelopmentRegion": "en",
            "CFBundleExecutable": APP_EXECUTABLE,
            "CFBundleIdentifier": APP_ID,
            "CFBundleName": "Splash M1",
            "CFBundleDisplayName": "Splash M1",
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": bundle_version,
            "CFBundleVersion": bundle_version,
            "LSMinimumSystemVersion": macos_min,
        }
    )
    with destination.open("wb") as output:
        plistlib.dump(info, output, sort_keys=True)


def compile_app(source, executable, build_dir, macos_min):
    build_dir.mkdir(parents=True, exist_ok=True)
    module_cache = build_dir / "swift-module-cache"
    module_cache.mkdir(parents=True, exist_ok=True)
    sdk = subprocess.run(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()
    if not sdk:
        raise ValueError("xcrun did not return a macOS SDK path")
    subprocess.run(
        [
            "xcrun",
            "swiftc",
            "-O",
            "-target",
            f"arm64-apple-macos{macos_min}",
            "-sdk",
            sdk,
            "-framework",
            "AppKit",
            "-module-cache-path",
            str(module_cache),
            "-file-prefix-map",
            f"{ROOT}=.",
            "-debug-prefix-map",
            f"{ROOT}=.",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        cwd=ROOT,
    )
    executable.chmod(0o755)
    subprocess.run([str(executable), "--check"], check=True, cwd=ROOT)


def write_zip_tree(bundle, archive_path):
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as output:
        root_info = zipfile.ZipInfo(APP_NAME + "/")
        root_info.create_system = 3
        root_info.external_attr = (stat.S_IFDIR | 0o755) << 16 | 0x10
        output.writestr(root_info, b"")
        for directory, directories, files in os.walk(bundle, followlinks=False):
            directories.sort()
            files.sort()
            parent = Path(directory)
            entries = [parent / name for name in directories + files]
            for entry in entries:
                relative = entry.relative_to(bundle).as_posix()
                name = f"{APP_NAME}/{relative}"
                mode = entry.lstat().st_mode
                if stat.S_ISLNK(mode):
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    info.compress_type = zipfile.ZIP_STORED
                    output.writestr(info, os.readlink(entry).encode())
                elif stat.S_ISDIR(mode):
                    info = zipfile.ZipInfo(name + "/")
                    info.create_system = 3
                    info.external_attr = (stat.S_IFDIR | (mode & 0o777)) << 16 | 0x10
                    output.writestr(info, b"")
                elif stat.S_ISREG(mode):
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | (mode & 0o777)) << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    with (
                        entry.open("rb") as source,
                        output.open(info, "w", force_zip64=True) as target,
                    ):
                        shutil.copyfileobj(source, target)
                else:
                    raise ValueError(f"unsupported app bundle entry: {entry}")


def publish_artifacts(temporary_archive, temporary_checksum, archive, checksum):
    linked_archive = False
    try:
        os.link(temporary_archive, archive)
        linked_archive = True
        os.link(temporary_checksum, checksum)
    except FileExistsError as error:
        if linked_archive:
            archive.unlink()
        raise ValueError(f"release already exists: {error.filename}") from error
    except OSError:
        if linked_archive:
            archive.unlink()
        raise


def build_app(version, macos_min, runtime_archive, build_dir):
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError("invalid release version")
    try:
        major = macos_major(macos_min)
    except ValueError as error:
        raise ValueError(str(error)) from error

    root_name = release_name(version, major)
    runtime_archive = Path(runtime_archive).resolve()
    if runtime_archive.name != root_name + ".tar.gz":
        raise ValueError(f"runtime archive must be named {root_name}.tar.gz")
    if not runtime_archive.is_file():
        raise ValueError(f"missing runtime archive: {runtime_archive}")

    source = ROOT / "desktop/main.swift"
    plist_template = ROOT / "desktop/Info.plist"
    if not source.is_file():
        raise ValueError(f"missing app source: {source}")
    if not plist_template.is_file():
        raise ValueError(f"missing app Info.plist template: {plist_template}")
    with plist_template.open("rb") as template:
        try:
            plistlib.load(template)
        except plistlib.InvalidFileException as error:
            raise ValueError(
                f"cannot read app Info.plist template: {plist_template}"
            ) from error

    dist = ROOT / "dist"
    archive = dist / f"{root_name}-app.zip"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    dist.mkdir(exist_ok=True)
    if archive.exists() or checksum.exists():
        raise ValueError(f"release already exists: {archive}")

    build_dir = Path(build_dir).resolve()
    with tempfile.TemporaryDirectory(prefix=".package-app-", dir=dist) as temporary:
        stage = Path(temporary)
        extraction = stage / "extracted"
        extraction.mkdir()
        runtime = validate_runtime_archive(
            runtime_archive, version, macos_min, major, extraction
        )
        bundle = stage / APP_NAME
        contents = bundle / "Contents"
        macos_dir = contents / "MacOS"
        resources = contents / "Resources"
        macos_dir.mkdir(parents=True)
        resources.mkdir()
        runtime.rename(resources / "runtime")
        write_info_plist(plist_template, contents / "Info.plist", version, macos_min)
        executable = macos_dir / APP_EXECUTABLE
        compile_app(source, executable, build_dir, macos_min)

        temporary_archive = stage / archive.name
        write_zip_tree(bundle, temporary_archive)
        temporary_checksum = stage / checksum.name
        temporary_checksum.write_text(digest(temporary_archive) + "\n")
        publish_artifacts(temporary_archive, temporary_checksum, archive, checksum)
    print(f"Built {archive} and {checksum}.")
    return archive, checksum


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--macos-min", default="15.0")
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "build",
        help="Swift module cache location",
    )
    args = parser.parse_args(argv)
    try:
        build_app(args.version, args.macos_min, args.runtime_archive, args.build_dir)
    except (OSError, tarfile.TarError, zipfile.BadZipFile, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
