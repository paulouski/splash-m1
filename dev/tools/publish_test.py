#!/usr/bin/env python3
"""Upload a built Splash M1 package to a private Hugging Face test repo."""

import argparse
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--macos-min", default="15.0")
    parser.add_argument(
        "--repo",
        required=True,
        help="private Hugging Face repo, e.g. owner/splash-test",
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="upload without moving the `latest` pointer",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        parser.error("invalid release version")
    macos = re.fullmatch(
        r"(0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))?",
        args.macos_min,
    )
    if not macos or int(macos.group(1)) < 15:
        parser.error("minimum macOS must be 15.0 or newer, such as 15.0")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo
    ):
        parser.error("invalid Hugging Face repository; expected owner/repository")
    dist = ROOT / "dist"
    archive = dist / f"splash-m1-{args.version}-arm64-macos{macos.group(1)}.tar.gz"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    installer = dist / "install.sh"
    required = (archive, checksum, installer)
    for path in required:
        if not path.is_file():
            sys.exit(
                f"missing {path}; run `make package RELEASE_VERSION={args.version}` first"
            )

    api = HfApi()
    api.create_repo(args.repo, private=True, exist_ok=True)
    uploads = [
        (archive, archive.name),
        (checksum, checksum.name),
    ]
    uploads.append((installer, "install.sh"))
    for path, name in uploads:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=args.repo)
        print(f"uploaded {name}")
    if not args.no_latest:
        api.upload_file(
            path_or_fileobj=(args.version + "\n").encode(),
            path_in_repo="latest",
            repo_id=args.repo,
        )
        print(f"latest -> {args.version}")
    print(
        "Private-HF testers: export SPLASH_TOKEN to the supplied read token, then run:\n"
        "curl -qfsSL --config - "
        f"https://huggingface.co/{args.repo}/resolve/main/install.sh <<EOF"
        f" | SPLASH_BASE_URL=https://huggingface.co/{args.repo}/resolve/main "
        'SPLASH_TOKEN="$SPLASH_TOKEN" sh\n'
        'header = "Authorization: Bearer $SPLASH_TOKEN"\n'
        "EOF"
    )


if __name__ == "__main__":
    main()
