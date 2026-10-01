import hashlib
import json
import plistlib
import stat
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from dev.tools import package_app


class PackageAppTests(unittest.TestCase):
    def make_release(
        self,
        root,
        version="1.2.3",
        macos_min="15.0",
        manifest_overrides=None,
        unsafe_link=False,
    ):
        major = package_app.macos_major(macos_min)
        name = package_app.release_name(version, major)
        dist = root / "dist"
        dist.mkdir(exist_ok=True)
        source = root / "archive-source" / name
        for folder in (
            "engine",
            "install",
            "python/bin",
            "server",
        ):
            (source / folder).mkdir(parents=True, exist_ok=True)
        engine = source / "engine/splash"
        engine.write_bytes(b"fixture native engine")
        engine.chmod(0o755)
        metallib = source / "engine/splash.metallib"
        metallib.write_bytes(b"fixture metallib")
        (source / "install/desktop.py").write_text("# desktop entry point\n")
        (source / "install/launcher.py").write_text("# launcher\n")
        python = source / "python/bin/python3"
        python.write_bytes(b"fixture bundled python")
        python.chmod(0o755)
        (source / "python/bin/python").symlink_to("python3")
        (source / "server/server.py").write_text("# server\n")
        (source / "splash-m1").write_text("#!/bin/sh\n")
        (source / "splash-m1").chmod(0o755)
        if unsafe_link:
            (source / "python/bin/escape").symlink_to("../../../../outside")

        metadata = {
            "product": "splash-m1",
            "version": version,
            "architecture": "arm64",
            "minimum_macos": macos_min,
            "binary_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "metallib_sha256": hashlib.sha256(metallib.read_bytes()).hexdigest(),
        }
        metadata.update(manifest_overrides or {})
        (source / "release.json").write_text(json.dumps(metadata) + "\n")

        archive = dist / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            output.add(source, arcname=name)
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.with_suffix(archive.suffix + ".sha256").write_text(checksum + "\n")
        return archive

    def setup_app_sources(self, root):
        (root / "desktop").mkdir(parents=True)
        (root / "desktop/main.swift").write_text("// app source fixture\n")
        with (root / "desktop/Info.plist").open("wb") as output:
            plistlib.dump(
                {
                    "CFBundleIdentifier": "io.github.paulouski.splash-m1",
                    "CFBundleExecutable": "SplashM1",
                    "CFBundleName": "Splash M1",
                },
                output,
            )
        (root / "build").mkdir()

    def mock_toolchain(self):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            if command[:3] == ["xcrun", "--sdk", "macosx"]:
                return subprocess.CompletedProcess(command, 0, stdout="/sdk\n")
            if command[:2] == ["xcrun", "swiftc"]:
                executable = Path(command[command.index("-o") + 1])
                executable.write_bytes(b"fixture SplashM1 binary")
                executable.chmod(0o755)
                return subprocess.CompletedProcess(command, 0)
            if command[-1:] == ["--check"]:
                return subprocess.CompletedProcess(command, 0)
            raise AssertionError(f"unexpected command: {command}")

        return calls, run

    def test_bundle_uses_verified_runtime_and_preserves_modes_and_symlinks(self):
        with tempfile.TemporaryDirectory(prefix="splash package app ") as temporary:
            root = Path(temporary)
            archive = self.make_release(root, version="0.1.0-local")
            self.setup_app_sources(root)
            calls, run = self.mock_toolchain()
            with (
                mock.patch.object(package_app, "ROOT", root),
                mock.patch.object(package_app.subprocess, "run", side_effect=run),
            ):
                output, checksum = package_app.build_app(
                    "0.1.0-local", "15.0", archive, root / "build"
                )

            self.assertEqual(output.name, "splash-m1-0.1.0-local-arm64-macos15-app.zip")
            self.assertEqual(
                checksum.read_text(),
                hashlib.sha256(output.read_bytes()).hexdigest() + "\n",
            )
            app_root = "Splash M1.app/Contents/"
            with zipfile.ZipFile(output) as bundle:
                names = set(bundle.namelist())
                self.assertIn(app_root + "MacOS/SplashM1", names)
                self.assertIn(app_root + "Resources/runtime/release.json", names)
                release = json.loads(
                    bundle.read(app_root + "Resources/runtime/release.json")
                )
                self.assertEqual(release["version"], "0.1.0-local")
                self.assertEqual(
                    bundle.read(app_root + "Resources/runtime/engine/splash"),
                    b"fixture native engine",
                )
                executable_mode = (
                    bundle.getinfo(
                        app_root + "Resources/runtime/engine/splash"
                    ).external_attr
                    >> 16
                )
                self.assertTrue(executable_mode & 0o111)
                symlink_info = bundle.getinfo(
                    app_root + "Resources/runtime/python/bin/python"
                )
                self.assertTrue(stat.S_ISLNK(symlink_info.external_attr >> 16))
                self.assertEqual(bundle.read(symlink_info.filename), b"python3")
                info = plistlib.loads(bundle.read(app_root + "Info.plist"))
                self.assertEqual(
                    info["CFBundleIdentifier"], "io.github.paulouski.splash-m1"
                )
                self.assertEqual(info["CFBundleVersion"], "0.1.0")
                self.assertEqual(info["CFBundleShortVersionString"], "0.1.0")
                self.assertEqual(info["LSMinimumSystemVersion"], "15.0")
                self.assertEqual(info["CFBundleDevelopmentRegion"], "en")
                self.assertTrue(
                    all(
                        not name.startswith("/") and ".." not in Path(name).parts
                        for name in names
                    )
                )

            compiler = next(
                call[0] for call in calls if call[0][:2] == ["xcrun", "swiftc"]
            )
            self.assertIn("arm64-apple-macos15.0", compiler)
            self.assertEqual(package_app.app_version_number("v0.1.0"), "0.1.0")
            self.assertEqual(package_app.app_version_number("local-build"), "0.0.0")
            self.assertIn("-module-cache-path", compiler)
            self.assertTrue(
                Path(compiler[compiler.index("-module-cache-path") + 1]).is_relative_to(
                    (root / "build").resolve()
                )
            )
            self.assertIn(f"{root}=.", compiler)
            self.assertTrue(any(call[0][-1:] == ["--check"] for call in calls))

    def test_invalid_inputs_and_mismatched_release_fail_before_compile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.make_release(root, manifest_overrides={"version": "9.9.9"})
            self.setup_app_sources(root)
            calls, run = self.mock_toolchain()
            with (
                mock.patch.object(package_app, "ROOT", root),
                mock.patch.object(package_app.subprocess, "run", side_effect=run),
            ):
                for version, macos_min, expected in (
                    ("../latest", "15.0", "invalid release version"),
                    ("1.2.3", "14.0", "minimum macOS"),
                    ("1.2.3", "15.0", "release version mismatch"),
                ):
                    with self.subTest(version=version, macos_min=macos_min):
                        with self.assertRaisesRegex(ValueError, expected):
                            package_app.build_app(
                                version, macos_min, archive, root / "build"
                            )
                self.assertEqual(calls, [])
                self.assertFalse(
                    (root / "dist/splash-m1-1.2.3-arm64-macos15-app.zip").exists()
                )

    def test_unsafe_archive_link_and_existing_output_are_never_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.make_release(root, unsafe_link=True)
            self.setup_app_sources(root)
            calls, run = self.mock_toolchain()
            with (
                mock.patch.object(package_app, "ROOT", root),
                mock.patch.object(package_app.subprocess, "run", side_effect=run),
            ):
                with self.assertRaisesRegex(ValueError, "link escapes"):
                    package_app.build_app("1.2.3", "15.0", archive, root / "build")
                self.assertEqual(calls, [])
                self.assertFalse((root / "outside").exists())

                output = root / "dist/splash-m1-1.2.3-arm64-macos15-app.zip"
                output.write_bytes(b"existing release")
                with self.assertRaisesRegex(ValueError, "release already exists"):
                    package_app.build_app("1.2.3", "15.0", archive, root / "build")
                self.assertEqual(output.read_bytes(), b"existing release")
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
