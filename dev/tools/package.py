#!/usr/bin/env python3
"""Build the runtime-only macOS archive and its Homebrew formula."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20260825/"
    "cpython-3.13.15%2B20260825-aarch64-apple-darwin-install_only_stripped.tar.gz"
)
PYTHON_SHA256 = "149038dd0c194c25d4616d7e42a35f67f2edee96412788f74115819b6a4c8548"
INSTALL_FILES = (
    "__init__.py",
    "launcher.py",
    "clients.py",
    "paths.py",
    "models.py",
    "desktop.py",
    "desktop_models.py",
    "hub.py",
    "families.py",
    "assembly.py",
    "legacy.py",
    "upstream.py",
    "uninstall.py",
    "gguf.py",
    "catalog.py",
    "requirements.txt",
)
COMPLETION_FILES = (
    "models",
    "_splash",
    "splash.bash",
    "official-models.txt",
    "suggested-models.txt",
)
SERVER_FILES = (
    "__init__.py",
    "server.py",
    "backend.py",
    "constraints.py",
    "output.py",
    "frontend.py",
    "chat_templates.py",
    "judgments.py",
    "diagnostics.py",
    "api_shapes.py",
    "tool_schema.py",
    "tokenization.py",
    "json_codec.py",
    "latency.py",
    "metrics.py",
    "errors.py",
    "runtime.py",
    "protocol.py",
    "images.py",
    "documents.py",
    "document_worker.py",
    "http_security.py",
    "thinking.py",
    "schema_validation.py",
    "crash_trace.py",
    "web_tools.py",
    "user_settings.py",
    "chat.html",
)
# Splash's license and the notices of the third-party code it ships.
LICENSE_FILES = ("LICENSE", "THIRD_PARTY_NOTICES")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def remove_staged_entrypoints(python, staging):
    binary_dir = python.parent
    if not binary_dir.is_dir():
        return
    staging_prefix = str(staging).encode()
    for entry in binary_dir.iterdir():
        if entry.name.startswith("python") or entry.is_symlink() or not entry.is_file():
            continue
        if staging_prefix in entry.read_bytes():
            entry.unlink()


def filter_archive_member(item):
    if "__pycache__" in Path(item.name).parts:
        return None
    item.uid = item.gid = 0
    item.uname = item.gname = ""
    return item


def macos_major(macos_min):
    match = re.fullmatch(
        r"(?P<major>0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))?",
        macos_min,
    )
    if not match or int(match.group("major")) < 15:
        raise ValueError("minimum macOS must be 15.0 or newer, such as 15.0")
    return int(match.group("major"))


def stage_runtime(destination, version, build_dir=None, macos_min="15.0"):
    build_dir = Path(build_dir or ROOT / "build")
    for folder, names in (
        ("install", INSTALL_FILES),
        ("install/completions", COMPLETION_FILES),
        ("server", SERVER_FILES),
        ("engine", ("splash", "splash.metallib")),
    ):
        (destination / folder).mkdir()
        source = build_dir if folder == "engine" else ROOT / folder
        for name in names:
            shutil.copy2(source / name, destination / folder / name)
    shutil.copytree(ROOT / "server/static", destination / "server/static")
    for name in LICENSE_FILES:
        shutil.copy2(ROOT / name, destination / name)
    launcher = destination / "splash-m1"
    launcher.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        'HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
        "export PYTHONDONTWRITEBYTECODE=1\n"
        'exec "$HERE/python/bin/python3" -u "$HERE/install/launcher.py" "$@"\n'
    )
    launcher.chmod(0o755)
    (destination / "release.json").write_text(
        json.dumps(
            {
                "product": "splash-m1",
                "version": version,
                "architecture": "arm64",
                "minimum_macos": macos_min,
                "features": ["affine Qwen text checkpoints", "int8 KV cache"],
                "binary_sha256": digest(destination / "engine/splash"),
                "metallib_sha256": digest(destination / "engine/splash.metallib"),
            },
            indent=2,
        )
        + "\n"
    )


def formula(version, url, checksum, macos_min):
    # Inputs are encoded as Ruby double-quoted literals with the backslash,
    # the quote and the interpolation opener escaped, so no path or URL can
    # become executable Ruby; double quotes are what `brew audit` expects.
    def quote(text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("#{", "\\#{")
        return '"' + escaped + '"'

    return f"""class SplashM1MacOSRequirement < Requirement
  fatal true
  satisfy(build_env: false) {{ OS.mac? && MacOS.full_version >= {quote(macos_min)} }}

  def message
    {quote(f"Splash M1 requires macOS {macos_min} or newer.")}
  end
end

class SplashM1 < Formula
  desc "Local text inference engine for Apple silicon"
  homepage "https://github.com/paulouski/splash-m1"
  url {quote(url)}
  version {quote(version)}
  sha256 {quote(checksum)}
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on SplashM1MacOSRequirement

  def install
    libexec.install Dir["*"]
    (bin/"splash-m1").write <<~SH
      #!/bin/sh
      exec "#{{opt_libexec}}/splash-m1" "$@"
    SH
    chmod 0755, bin/"splash-m1"
    zsh_completion.install_symlink libexec/"install/completions/_splash" => "_splash-m1"
    bash_completion.install_symlink libexec/"install/completions/splash.bash" => "splash-m1"
  end

  def caveats
    <<~CAVEAT
      Serve a model:
        splash-m1 serve --model mlx-community/Qwen3.8-27B-4bit --language-only --max-context 32K
      Macs below 32 GB:
        splash-m1 serve --model prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --language-only
    CAVEAT
  end

  test do
    assert_match version.to_s, shell_output("#{{bin}}/splash-m1 --version")
    assert_match "serve", shell_output("#{{bin}}/splash-m1 --help")
  end
end
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--macos-min", default="15.0", help="minimum macOS, e.g. 15.0")
    parser.add_argument(
        "--build-dir", type=Path, default=ROOT / "build", help="runtime build directory"
    )
    parser.add_argument(
        "--url", help="published tarball URL; defaults to GitHub Releases"
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        parser.error("invalid release version")
    try:
        platform_major = macos_major(args.macos_min)
    except ValueError as error:
        parser.error(str(error))
    build_dir = args.build_dir.resolve()
    for name in ("splash", "splash.metallib"):
        if not (build_dir / name).is_file():
            parser.error(f"missing {build_dir / name}")
    dist = ROOT / "dist"
    name = f"splash-m1-{args.version}-arm64-macos{platform_major}"
    archive = dist / f"{name}.tar.gz"
    if archive.exists():
        parser.error(f"release already exists: {archive}")
    dist.mkdir(exist_ok=True)
    cached = ROOT / "build/release/python-runtime.tar.gz"
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.exists() or digest(cached) != PYTHON_SHA256:
        with tempfile.NamedTemporaryFile(dir=cached.parent) as download:
            with urllib.request.urlopen(PYTHON_URL, timeout=60) as response:
                shutil.copyfileobj(response, download)
            download.flush()
            if digest(Path(download.name)) != PYTHON_SHA256:
                raise ValueError("Python distribution checksum mismatch")
            shutil.copyfile(download.name, cached)
    with tempfile.TemporaryDirectory(prefix=".package-", dir=dist) as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        stage_runtime(stage, args.version, build_dir, args.macos_min)
        with tarfile.open(cached) as python:
            python.extractall(stage, filter="data")
        python = stage / "python/bin/python3"
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "--no-compile",
                "-r",
                str(stage / "install/requirements.txt"),
            ],
            check=True,
        )
        remove_staged_entrypoints(python, stage)
        # Exercise imports and the public entry point from the actual staged
        # runtime, not the developer's virtualenv or source PYTHONPATH.
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import sys; sys.path.insert(0, '.'); "
                "import tokenizers, llguidance, numpy, PIL, transformers, huggingface_hub, pypdfium2; "
                "import server.server, install.launcher",
            ],
            cwd=stage,
            check=True,
        )
        subprocess.run(
            [str(python), "-B", str(stage / "install/launcher.py"), "--help"],
            cwd=stage,
            check=True,
        )
        packed = Path(temporary) / archive.name
        with tarfile.open(packed, "w:gz") as release:
            release.add(
                stage,
                arcname=name,
                filter=filter_archive_member,
            )
        packed.replace(archive)
    checksum = digest(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(checksum + "\n")
    url = (
        args.url
        or f"https://github.com/paulouski/splash-m1/releases/download/{args.version}/{archive.name}"
    )
    (dist / "splash-m1.rb").write_text(
        formula(args.version, url, checksum, args.macos_min)
    )
    shutil.copy2(ROOT / "dev/tools/install.sh", dist / "install.sh")
    (dist / "latest").write_text(args.version + "\n")
    print(
        f"Built {archive}; run make package-bottle RELEASE_VERSION={args.version} "
        "before publishing the Homebrew formula."
    )


if __name__ == "__main__":
    main()
