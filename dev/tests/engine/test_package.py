import contextlib
import fcntl
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tools import package, publish_test
from install import paths


class PackageTests(unittest.TestCase):
    def test_release_inputs_are_rejected_before_network_or_distribution_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / "selected-build"
            build.mkdir()
            for name in ("splash", "splash.metallib"):
                (build / name).write_bytes(b"fixture")
            for arguments, expected in (
                (("../latest", "15.0"), "invalid release version"),
                (("1.0", "15;bad"), "minimum macOS must be 15.0"),
            ):
                error = io.StringIO()
                with (
                    self.subTest(version=arguments[0], macos_min=arguments[1]),
                    mock.patch.object(package, "ROOT", root),
                    mock.patch.object(
                        package.urllib.request,
                        "urlopen",
                        side_effect=AssertionError("unexpected network access"),
                    ) as urlopen,
                    contextlib.redirect_stderr(error),
                    self.assertRaises(SystemExit),
                ):
                    package.main(
                        [
                            "--version",
                            arguments[0],
                            "--macos-min",
                            arguments[1],
                            "--build-dir",
                            str(build),
                        ]
                    )
                self.assertIn(expected, error.getvalue())
                urlopen.assert_not_called()
                self.assertFalse((root / "dist").exists())

    def test_staging_absolute_console_launchers_are_removed_but_python_is_kept(self):
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "package/.package-stage"
            binary_dir = staging / "python/bin"
            binary_dir.mkdir(parents=True)
            python = binary_dir / "python3"
            python.write_bytes(b"bundled python executable")
            (binary_dir / "pip3").write_text(f"#!{python}\n# pip launcher\n")
            (binary_dir / "safe-tool").write_text("#!/usr/bin/env python3\n")
            package.remove_staged_entrypoints(python, staging)
            self.assertTrue(python.is_file())
            self.assertFalse((binary_dir / "pip3").exists())
            self.assertTrue((binary_dir / "safe-tool").is_file())

    def test_build_needs_no_hub_access_and_never_ships_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder, names in (
                ("install", package.INSTALL_FILES),
                ("install/completions", package.COMPLETION_FILES),
                ("server", package.SERVER_FILES),
                ("build-macos15", ("splash", "splash.metallib")),
            ):
                (root / folder).mkdir(parents=True)
                for name in names:
                    (root / folder / name).write_text("fixture")
            for name in package.LICENSE_FILES:
                (root / name).write_text("fixture")
            (root / "dev/tools").mkdir(parents=True)
            shutil.copy2(
                package.ROOT / "dev/tools/install.sh", root / "dev/tools/install.sh"
            )
            static = root / "server/static"
            static.mkdir()
            (static / "fixture.js").write_text("static fixture")
            (root / "build").mkdir()
            for name in ("splash", "splash.metallib"):
                (root / "build" / name).write_text("wrong default build")
            (root / "install/completions/official-models.txt").write_text(
                "company/Published\n"
            )
            (root / "install/download-token").write_text("hf_legacycredential")
            cached = root / "build/release/python-runtime.tar.gz"
            cached.parent.mkdir()
            with tarfile.open(cached, "w:gz"):
                pass
            for version, environment in (
                ("public", {}),
                ("private", {"HF_TOKEN": "hf_testcredential"}),
            ):
                with (
                    self.subTest(version=version),
                    mock.patch.object(package, "ROOT", root),
                    mock.patch.object(package, "PYTHON_SHA256", package.digest(cached)),
                    mock.patch.object(
                        package.subprocess,
                        "run",
                        return_value=subprocess.CompletedProcess([], 0),
                    ) as run,
                    mock.patch.object(
                        package.urllib.request,
                        "urlopen",
                        side_effect=AssertionError("unexpected network access"),
                    ),
                    mock.patch.dict(os.environ, environment, clear=True),
                    mock.patch("sys.stdout"),
                ):
                    package.main(
                        [
                            "--version",
                            version,
                            "--macos-min",
                            "15.0",
                            "--build-dir",
                            str(root / "build-macos15"),
                        ]
                    )
                self.assertEqual(run.call_count, 4)
                name = f"splash-m1-{version}-arm64-macos15"
                self.assertEqual((root / "dist/latest").read_text(), version + "\n")
                self.assertEqual(
                    (root / "dist/install.sh").read_bytes(),
                    (root / "dev/tools/install.sh").read_bytes(),
                )
                self.assertTrue((root / "dist/splash-m1.rb").is_file())
                with tarfile.open(root / f"dist/{name}.tar.gz") as archive:
                    self.assertEqual(
                        archive.extractfile(f"{name}/engine/splash").read(),
                        b"fixture",
                    )
                    self.assertEqual(
                        archive.extractfile(f"{name}/engine/splash.metallib").read(),
                        b"fixture",
                    )
                    launcher = archive.getmember(f"{name}/splash-m1")
                    self.assertTrue(launcher.mode & 0o111)
                    with archive.extractfile(f"{name}/release.json") as manifest:
                        metadata = json.load(manifest)
                    self.assertEqual(metadata["product"], "splash-m1")
                    self.assertEqual(metadata["architecture"], "arm64")
                    self.assertEqual(metadata["minimum_macos"], "15.0")
                    self.assertEqual(
                        metadata["features"],
                        ["affine Qwen text checkpoints", "int8 KV cache"],
                    )
                    for member in archive.getmembers():
                        self.assertEqual(member.uid, 0)
                        self.assertEqual(member.gid, 0)
                        self.assertEqual(member.uname, "")
                        self.assertEqual(member.gname, "")
                        self.assertNotIn("download-token", member.name)
                        if member.isfile():
                            content = archive.extractfile(member).read()
                            self.assertNotIn(b"hf_legacycredential", content)
                            self.assertNotIn(b"hf_testcredential", content)
                    with archive.extractfile(
                        f"{name}/install/completions/official-models.txt"
                    ) as catalog:
                        self.assertEqual(catalog.read(), b"company/Published\n")

    def test_release_has_only_runtime_files_and_excludes_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            stage = Path(temporary) / "stage"
            stage.mkdir()
            for folder, names in (
                ("install", package.INSTALL_FILES),
                ("server", package.SERVER_FILES),
                ("build", ("splash", "splash.metallib")),
            ):
                (root / folder).mkdir(parents=True)
                for name in names:
                    (root / folder / name).write_text("fixture")
            for name in package.LICENSE_FILES:
                (root / name).write_text("license")
            (root / "server/static").mkdir()
            (root / "server/static/fixture.js").write_text("static fixture")
            completions = root / "install/completions"
            completions.mkdir()
            completion_names = {
                "models",
                "_splash",
                "splash.bash",
                "official-models.txt",
                "suggested-models.txt",
            }
            for name in completion_names:
                (completions / name).write_text(f"fixture {name}\n")
            (completions / "models").chmod(0o755)
            (completions / "private-junk").write_text("must not ship")
            (root / "install/private-junk").write_text("must not ship")
            (root / "server/local.log").write_text("must not ship")
            token = root / "install/download-token"
            token.write_text("hf_testdistributiontoken")
            with mock.patch.object(package, "ROOT", root):
                package.stage_runtime(stage, "test")
            self.assertEqual(
                {p.name for p in stage.iterdir()},
                {
                    "install",
                    "server",
                    "engine",
                    "LICENSE",
                    "THIRD_PARTY_NOTICES",
                    "release.json",
                    "splash-m1",
                },
            )
            self.assertEqual(
                {p.name for p in (stage / "install").iterdir()},
                {*package.INSTALL_FILES, "completions"},
            )
            shipped = stage / "install/completions"
            self.assertEqual({p.name for p in shipped.iterdir()}, completion_names)
            self.assertEqual((shipped / "models").stat().st_mode & 0o777, 0o755)
            for name in completion_names:
                self.assertEqual(
                    (shipped / name).read_bytes(), (completions / name).read_bytes()
                )
            self.assertEqual(
                {p.name for p in (stage / "server").iterdir()},
                {*package.SERVER_FILES, "static"},
            )
            self.assertNotIn(
                "hf_testdistributiontoken", (stage / "release.json").read_text()
            )
            self.assertFalse((stage / "install/model-catalog.json").exists())
            self.assertNotIn(
                "catalog_sha256", json.loads((stage / "release.json").read_text())
            )
            metadata = json.loads((stage / "release.json").read_text())
            self.assertEqual(metadata["product"], "splash-m1")
            self.assertEqual(metadata["minimum_macos"], "15.0")
            self.assertTrue(os.access(stage / "splash-m1", os.X_OK))
            bundled_python = stage / "python/bin/python3"
            bundled_python.parent.mkdir(parents=True)
            bundled_python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            bundled_python.chmod(0o755)
            result = subprocess.run(
                [str(stage / "splash-m1"), "--help"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.stdout.splitlines(),
                ["-u", str(stage / "install/launcher.py"), "--help"],
            )

    def test_every_installer_module_ships(self):
        # The packaged launcher and installer import these by module name.
        modules = {path.name for path in (package.ROOT / "install").glob("*.py")}
        self.assertLessEqual(modules, set(package.INSTALL_FILES))

    def test_packaged_paths_keep_user_data_outside_versioned_prefix(self):
        source = Path(paths.__file__).read_text()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(Path, "home", return_value=root / "home"):
                results = []
                for version in ("1", "2"):
                    prefix = (root / version).resolve()
                    prefix.mkdir()
                    (prefix / "release.json").write_text("{}")
                    namespace = {"__file__": str(prefix / "install/paths.py")}
                    exec(compile(source, "paths.py", "exec"), namespace)
                    self.assertTrue(namespace["PACKAGED"])
                    self.assertEqual(namespace["BINARY"], prefix / "engine/splash")
                    self.assertEqual(namespace["PYTHON"], prefix / "python/bin/python3")
                    results.append((namespace["MODELS"], namespace["RUNTIME"]))
                self.assertEqual(results[0], results[1])
                self.assertTrue(results[0][0].is_relative_to(root / "home"))

    def test_formula_invokes_bundled_runtime_without_build_or_dependency_install(self):
        text = package.formula(
            "1.0", "https://example.org/release.tar.gz", "a" * 64, "15.0"
        )
        self.assertNotIn("depends_on macos:", text)
        self.assertIn("class SplashM1 < Formula", text)
        self.assertIn("depends_on SplashM1MacOSRequirement", text)
        self.assertIn('version "1.0"', text)
        self.assertIn(
            'satisfy(build_env: false) { OS.mac? && MacOS.full_version >= "15.0" }',
            text,
        )
        self.assertLess(
            text.index("satisfy(build_env: false)"), text.index("def install")
        )
        self.assertIn('libexec.install Dir["*"]', text)
        self.assertIn('exec "#{opt_libexec}/splash-m1" "$@"', text)
        self.assertNotIn("make", text)
        self.assertNotIn("pip install", text)
        self.assertNotIn("download-token", text)
        self.assertIn('"$@"', text)
        self.assertIn('chmod 0755, bin/"splash-m1"', text)
        self.assertIn('=> "_splash-m1"', text)
        self.assertIn('=> "splash-m1"', text)
        self.assertIn("--language-only --max-context 32K", text)

    @unittest.skipUnless(shutil.which("ruby"), "Ruby is needed to exercise the formula")
    def test_formula_checks_minimum_os_without_running_install(self):
        driver = r"""
require "rubygems/version"
class Requirement
  def self.fatal(*args); end
  def self.satisfy(build_env:, &block)
    raise "must not require build tools" if build_env
    define_singleton_method(:check, &block)
  end
end
class Formula
  [:desc, :homepage, :url, :version, :sha256, :license, :depends_on, :test].each do |name|
    define_singleton_method(name) { |*args, &block| }
  end
end
module OS
  def self.mac?; ENV.fetch("TEST_MAC") == "1"; end
end
module MacOS
  class Version < Gem::Version
    def >=(other); super(Gem::Version.new(other)); end
  end
  def self.full_version; Version.new(ENV.fetch("TEST_VERSION")); end
end
eval STDIN.read
            puts SplashM1MacOSRequirement.check
"""
        formula = package.formula(
            "1.0", "https://example.org/a.tar.gz", "a" * 64, "15.0"
        )
        for mac, version, allowed in (
            (True, "14.9", False),
            (True, "15.0", True),
            (True, "15.7", True),
            (True, "26.0", True),
            (False, "27.0", False),
        ):
            with self.subTest(mac=mac, version=version):
                result = subprocess.run(
                    ["ruby", "-e", driver],
                    input=formula,
                    text=True,
                    capture_output=True,
                    check=True,
                    env={
                        **os.environ,
                        "TEST_MAC": str(int(mac)),
                        "TEST_VERSION": version,
                    },
                )
                self.assertEqual(result.stdout.strip(), str(allowed).lower())

    @unittest.skipUnless(shutil.which("ruby"), "Ruby is needed to exercise the formula")
    def test_formula_completion_links_survive_upgrading_an_existing_keg(self):
        with tempfile.TemporaryDirectory(prefix="splash formula ") as temporary:
            root = Path(temporary)
            brew = root / "homebrew prefix"
            previous = brew / "Cellar/splash-m1/old"
            prefix = brew / "Cellar/splash-m1/new"
            opt = brew / "opt/splash-m1"
            opt.parent.mkdir(parents=True)
            opt.symlink_to(previous)
            completion_entries = (
                ("share/zsh/site-functions/_splash-m1", "_splash"),
                ("etc/bash_completion.d/splash-m1", "splash.bash"),
            )
            old_assets = previous / "libexec/install/completions"
            old_assets.mkdir(parents=True)
            for entry, asset in completion_entries:
                (old_assets / asset).write_text(f"old {asset}\n")
                old_entry = previous / entry
                old_entry.parent.mkdir(parents=True)
                old_entry.symlink_to(
                    os.path.relpath(old_assets / asset, old_entry.parent)
                )
                linked = brew / entry
                linked.parent.mkdir(parents=True)
                linked.symlink_to(os.path.relpath(old_entry, linked.parent))
            (old_assets / "models").write_text("#!/bin/sh\nprintf '%s\\n' old/model\n")
            (old_assets / "models").chmod(0o755)
            source = root / "source"
            assets = source / "install/completions"
            assets.mkdir(parents=True)
            for name in ("_splash", "splash.bash", "official-models.txt"):
                (assets / name).write_text(f"fixture {name}\n")
            (assets / "models").write_text("#!/bin/sh\nprintf '%s\\n' new/model\n")
            (assets / "models").chmod(0o755)
            prefix.mkdir()
            formula = root / "splash-m1.rb"
            formula.write_text(
                package.formula(
                    "test", "https://example.org/splash-m1.tar.gz", "a" * 64, "15.0"
                )
            )
            # Match Homebrew's parent realpath and relative install_symlink
            # behavior while opt still points at the previously installed keg.
            driver = r"""
require "fileutils"
require "pathname"
class Pathname
  def install(files)
    mkpath
    FileUtils.cp_r(files, to_s)
  end
  def install_symlink(source)
    mkpath
    dstdir = realpath
    links = source.is_a?(Hash) ? source : {source => source.basename}
    links.each do |from, to|
      from = Pathname(from).expand_path(dstdir)
      from = from.dirname.realpath/from.basename if from.dirname.exist?
      FileUtils.ln_sf(from.relative_path_from(dstdir), dstdir/to)
    end
  end
end
module MacOS
  def self.full_version; "26.4"; end
end
class Requirement
  def self.fatal(*args); end
  def self.satisfy(*args, &block); end
end
class Formula
  [:desc, :homepage, :url, :version, :sha256, :license, :depends_on, :test].each do |name|
    define_singleton_method(name) { |*args, &block| }
  end
  attr_reader :libexec, :opt_libexec, :bin, :zsh_completion, :bash_completion
  def initialize(prefix, opt)
    @libexec = prefix/"libexec"
    @opt_libexec = opt/"libexec"
    @bin = prefix/"bin"
    @bin.mkpath
    @zsh_completion = prefix/"share/zsh/site-functions"
    @bash_completion = prefix/"etc/bash_completion.d"
  end
  def chmod(mode, path); File.chmod(mode, path); end
  def odie(message); raise message; end
end
load ARGV[0]
            SplashM1.new(Pathname(ARGV[1]), Pathname(ARGV[2])).install
"""
            subprocess.run(
                ["ruby", "-e", driver, str(formula), str(prefix), str(opt)],
                cwd=source,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(opt.resolve(), previous.resolve())
            for entry, asset in completion_entries:
                link = prefix / entry
                self.assertTrue(link.is_symlink())
                self.assertEqual(
                    link.resolve(strict=True),
                    (prefix / "libexec/install/completions" / asset).resolve(),
                )
                self.assertEqual(link.read_text(), f"fixture {asset}\n")
                linked = brew / entry
                linked.unlink()
                linked.symlink_to(os.path.relpath(link, linked.parent))
            opt.unlink()
            opt.symlink_to(prefix)
            shutil.rmtree(previous)
            for entry, asset in completion_entries:
                linked = brew / entry
                self.assertEqual(linked.read_text(), f"fixture {asset}\n")
                helper = linked.resolve(strict=True).parent / "models"
                result = subprocess.run(
                    [str(helper)], check=True, capture_output=True, text=True
                )
                self.assertEqual(result.stdout, "new/model\n")


class InstallerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.releases = self.root / "releases"
        self.bin = self.root / "bin"
        for folder in (self.root / "home", self.releases, self.bin):
            folder.mkdir()
        self.app = self.root / "home/Library/Application Support/Splash M1/app"
        self.command = self.bin / "splash-m1"
        self.apps = self.root / "apps"
        self.apps.mkdir()

    def publish(self, version, help_status=0, completions=True):
        name = f"splash-m1-{version}-arm64-macos15"
        staging = self.root / "staging"
        release = staging / name
        (release / "python/bin").mkdir(parents=True)
        (release / "install").mkdir()
        (release / "release.json").write_text(f'{{"version": "{version}"}}\n')
        interpreter = release / "python/bin/python3"
        interpreter.write_text(
            "#!/bin/sh\n"
            f'if [ "$1" = - ]; then exec {shlex.quote(sys.executable)} "$@"; fi\n'
            f"exit {help_status}\n"
        )
        interpreter.chmod(0o755)
        (release / "install/launcher.py").write_text("# stub\n")
        if completions:
            assets = release / "install/completions"
            assets.mkdir()
            for asset in ("splash.bash", "_splash"):
                (assets / asset).write_text(
                    f"SPLASH_COMPLETION_TEST_VERSION='{version}'\n"
                )
            helper = assets / "models"
            helper.write_text("#!/bin/sh\nprintf '%s\\n' fixture/model\n")
            helper.chmod(0o755)
            (assets / "official-models.txt").write_text("fixture/model\n")
        archive = self.releases / f"{name}.tar.gz"
        subprocess.run(
            ["tar", "-czf", str(archive), "-C", str(staging), name], check=True
        )
        shutil.rmtree(staging)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (self.releases / f"{name}.tar.gz.sha256").write_text(
            f"{digest}  {name}.tar.gz\n"
        )
        (self.releases / "latest").write_text(f"{version}\n")
        bundle = self.root / "bundle/Splash M1.app/Contents"
        bundle.mkdir(parents=True)
        (bundle / "Info.plist").write_text(f"<plist>{version}</plist>\n")
        app_zip = self.releases / f"{name}-app.zip"
        subprocess.run(
            ["ditto", "-c", "-k", "--keepParent", str(bundle.parent), str(app_zip)],
            check=True,
        )
        shutil.rmtree(self.root / "bundle")
        (self.releases / f"{app_zip.name}.sha256").write_text(
            f"{hashlib.sha256(app_zip.read_bytes()).hexdigest()}  {app_zip.name}\n"
        )

    def install(self):
        return self.run_installer()

    def run_installer(self, public=False, extra_environment=None, arguments=()):
        environment = {
            "PATH": str(self.root / "commands") + os.pathsep + os.environ["PATH"],
            "HOME": str(self.root / "home"),
            "SPLASH_BIN_DIR": str(self.bin),
            "SPLASH_APP_DIR": str(self.apps),
        }
        if not public:
            environment["SPLASH_BASE_URL"] = self.releases.as_uri()
        if extra_environment:
            environment.update(extra_environment)
        return subprocess.run(
            ["/bin/sh", str(package.ROOT / "dev/tools/install.sh"), *arguments],
            env=environment,
            capture_output=True,
            text=True,
        )

    def test_unsafe_version_is_rejected_before_creating_the_install_prefix(self):
        result = self.run_installer(extra_environment={"SPLASH_VERSION": "../1.0"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("invalid release version", result.stderr)
        self.assertFalse(self.app.exists())
        self.assertFalse(self.command.exists())

    def test_public_github_default_installs_without_a_token(self):
        self.publish("1.0")
        commands = self.root / "commands"
        commands.mkdir()
        curl = commands / "curl"
        curl.write_text(
            f"#!{sys.executable}\n"
            "import shutil, sys\n"
            "arguments = sys.argv[1:]\n"
            "if '--config' in arguments: sys.exit('public download used auth config')\n"
            "url = arguments[-1]\n"
            "prefix = 'https://github.com/paulouski/splash-m1/releases/'\n"
            "if not url.startswith(prefix): sys.exit('unexpected URL: ' + url)\n"
            "asset = url[len(prefix):]\n"
            "if asset.startswith('latest/download/'):\n"
            "    asset = asset.removeprefix('latest/download/')\n"
            "else:\n"
            "    asset = asset.split('download/', 1)[1].split('/', 1)[1]\n"
            f"source = open({str(self.releases)!r} + '/' + asset, 'rb')\n"
            "target = open(arguments[arguments.index('-o') + 1], 'wb')\n"
            "shutil.copyfileobj(source, target)\n"
        )
        curl.chmod(0o755)
        result = self.run_installer(public=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.current(), "splash-m1-1.0-arm64-macos15")
        self.assertIn("Downloading Splash M1 1.0", result.stdout)

    def installed(self):
        return sorted(
            entry.name for entry in self.app.iterdir() if not entry.is_symlink()
        )

    def current(self):
        return Path(os.readlink(self.app / "current")).name

    def completion_version(self):
        script = self.app / "current/install/completions/splash.bash"
        result = subprocess.run(
            [
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                'source "$1"; printf "%s" "$SPLASH_COMPLETION_TEST_VERSION"',
                "bash",
                str(script),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def assertKeepsFirstVersion(self, result, command):
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.current(), "splash-m1-1.0-arm64-macos15")
        self.assertEqual(self.command.read_text(), command)
        self.assertEqual(self.completion_version(), "1.0")

    def test_app_is_installed_without_quarantine(self):
        self.publish("1.0")
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        plist = self.apps / "Splash M1.app/Contents/Info.plist"
        self.assertIn("1.0", plist.read_text())
        attributes = subprocess.run(
            ["xattr", "-r", str(self.apps / "Splash M1.app")],
            capture_output=True,
            text=True,
        ).stdout
        self.assertNotIn("com.apple.quarantine", attributes)
        self.assertEqual(
            [entry.name for entry in self.apps.iterdir()], ["Splash M1.app"]
        )

    def test_app_checksum_mismatch_changes_nothing(self):
        self.publish("1.0")
        (self.releases / "splash-m1-1.0-arm64-macos15-app.zip.sha256").write_text(
            "0" * 64 + "  app.zip\n"
        )
        result = self.install()
        self.assertEqual(result.returncode, 1)
        self.assertIn("checksum mismatch", result.stderr)
        self.assertFalse(self.command.exists())
        self.assertFalse((self.app / "current").exists())
        self.assertEqual(list(self.apps.iterdir()), [])

    def test_cli_only_skips_the_app(self):
        self.publish("1.0")
        (self.releases / "splash-m1-1.0-arm64-macos15-app.zip").unlink()
        result = self.run_installer(arguments=["--cli-only"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.current(), "splash-m1-1.0-arm64-macos15")
        self.assertEqual(list(self.apps.iterdir()), [])
        self.assertNotIn("Desktop app", result.stdout)

    def test_running_app_blocks_the_install(self):
        self.publish("1.0")
        commands = self.root / "commands"
        commands.mkdir()
        pgrep = commands / "pgrep"
        pgrep.write_text("#!/bin/sh\nexit 0\n")
        pgrep.chmod(0o755)
        result = self.install()
        self.assertEqual(result.returncode, 1)
        self.assertIn("quit Splash M1", result.stderr)
        self.assertFalse(self.command.exists())
        self.assertEqual(list(self.apps.iterdir()), [])

    def test_install_points_the_command_at_the_new_version_and_prunes_the_old(self):
        self.publish("1.0")
        first = self.install()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(self.installed(), ["splash-m1-1.0-arm64-macos15"])
        self.assertEqual(self.current(), "splash-m1-1.0-arm64-macos15")
        self.assertIn(
            "Splash M1/app/current/install/launcher.py", self.command.read_text()
        )
        self.assertTrue(os.access(self.command, os.X_OK))
        self.assertEqual(self.completion_version(), "1.0")
        self.assertIn(
            'source "$HOME/Library/Application Support/Splash M1/app/current/install/completions/splash.bash"',
            first.stdout,
        )
        self.assertIn("Zsh needs compinit initialized", first.stdout)
        self.publish("2.0")
        upgrade = self.install()
        self.assertEqual(upgrade.returncode, 0, upgrade.stderr)
        self.assertEqual(self.installed(), ["splash-m1-2.0-arm64-macos15"])
        self.assertEqual(self.current(), "splash-m1-2.0-arm64-macos15")
        self.assertEqual(self.completion_version(), "2.0")
        self.assertTrue(
            os.access(self.app / "current/install/completions/models", os.X_OK)
        )

    def test_live_serve_lock_blocks_upgrade_but_an_unlocked_file_does_not(self):
        self.publish("1.0")
        self.assertEqual(self.install().returncode, 0)
        command = self.command.read_text()
        self.publish("2.0")
        lock_path = self.app.parent / "runtime/serve.lock"
        with lock_path.open("a+") as lock:
            lock.write('{"pid":123,"model":"test/model","port":8000}')
            lock.flush()
            for mode in (fcntl.LOCK_EX, fcntl.LOCK_SH):
                with self.subTest(lock_mode=mode):
                    fcntl.flock(lock, mode | fcntl.LOCK_NB)
                    blocked = self.install()
                    self.assertKeepsFirstVersion(blocked, command)
                    self.assertIn("stop the running Splash M1 server", blocked.stderr)
                    self.assertEqual(self.installed(), ["splash-m1-1.0-arm64-macos15"])
                    fcntl.flock(lock, fcntl.LOCK_UN)
        self.assertEqual(self.install().returncode, 0)
        self.assertEqual(self.current(), "splash-m1-2.0-arm64-macos15")

    def test_running_incomplete_same_version_is_not_replaced(self):
        self.publish("1.0")
        self.assertEqual(self.install().returncode, 0)
        release = self.app / "splash-m1-1.0-arm64-macos15"
        (release / "release.json").unlink()
        marker = release / "in-use"
        marker.write_text("preserve")
        with (self.app.parent / "runtime/serve.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            blocked = self.install()
        self.assertEqual(blocked.returncode, 1, blocked.stderr)
        self.assertIn("stop the running Splash M1 server", blocked.stderr)
        self.assertEqual(marker.read_text(), "preserve")
        self.assertFalse((release / "release.json").exists())
        self.assertEqual(self.current(), release.name)

    def test_upgrade_holds_the_lifecycle_lock_while_removing_the_old_version(self):
        self.publish("1.0")
        self.assertEqual(self.install().returncode, 0)
        self.publish("2.0")
        commands = self.root / "commands"
        commands.mkdir()
        proof = self.root / "lock-proof"
        old = self.app / "splash-m1-1.0-arm64-macos15"
        lock = self.app.parent / "runtime/serve.lock"
        shim = commands / "rm"
        shim.write_text(
            f"#!{sys.executable}\n"
            "import fcntl, subprocess, sys\n"
            "from pathlib import Path\n"
            f"if {str(old)!r} in sys.argv:\n"
            f"    with open({str(lock)!r}, 'a+') as lock:\n"
            "        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "        except BlockingIOError:\n"
            f"            Path({str(proof)!r}).write_text('held')\n"
            "        else: sys.exit('upgrade released the lifecycle lock early')\n"
            "sys.exit(subprocess.call(['/bin/rm', *sys.argv[1:]]))\n"
        )
        shim.chmod(0o755)
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(proof.read_text(), "held")
        with lock.open("a+") as descriptor:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertFalse(old.exists())

    def test_older_release_without_completions_does_not_print_a_broken_source_command(
        self,
    ):
        self.publish("1.0", completions=False)
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("source ", result.stdout)
        self.assertEqual(self.current(), "splash-m1-1.0-arm64-macos15")

    def test_failed_same_version_check_preserves_absolute_and_relative_current(self):
        self.publish("1.0")
        first = self.install()
        self.assertEqual(first.returncode, 0, first.stderr)
        command = self.command.read_text()
        release = self.app / "splash-m1-1.0-arm64-macos15"
        (release / "python/bin/python3").write_text("#!/bin/sh\nexit 1\n")
        current = self.app / "current"

        for target in (str(release), release.name):
            with self.subTest(target=target):
                current.unlink()
                current.symlink_to(target)
                failed = self.install()
                self.assertKeepsFirstVersion(failed, command)
                self.assertIn("fails 'splash-m1 --help'", failed.stderr)
                self.assertEqual(os.readlink(current), target)
                self.assertTrue(release.is_dir())
                self.assertEqual(current.resolve(strict=True), release.resolve())

    def test_failed_first_install_removes_the_unusable_version(self):
        self.publish("1.0", help_status=1)
        failed = self.install()
        self.assertEqual(failed.returncode, 1, failed.stderr)
        self.assertIn("fails 'splash-m1 --help'", failed.stderr)
        self.assertEqual(self.installed(), [])
        self.assertFalse((self.app / "current").exists())
        self.assertFalse(self.command.exists())

    def test_a_failed_install_keeps_the_installed_version_link_and_command(self):
        self.publish("1.0")
        first = self.install()
        self.assertEqual(first.returncode, 0, first.stderr)
        command = self.command.read_text()

        self.publish("2.0", help_status=1)
        smoke = self.install()
        self.assertKeepsFirstVersion(smoke, command)
        self.assertIn("fails 'splash-m1 --help'", smoke.stderr)
        self.assertEqual(self.installed(), ["splash-m1-1.0-arm64-macos15"])

        self.publish("3.0")
        foreign = "#!/bin/sh\necho not the installer's\n"
        self.command.write_text(foreign)
        conflict = self.install()
        self.assertKeepsFirstVersion(conflict, foreign)
        self.assertIn("was not created by this installer", conflict.stderr)
        self.assertEqual(self.installed(), ["splash-m1-1.0-arm64-macos15"])
        self.command.write_text(command)

        (self.bin / "splash-m1.tmp").mkdir()
        blocked = self.install()
        self.assertKeepsFirstVersion(blocked, command)
        self.assertIn("could not write", blocked.stderr)
        self.assertEqual(
            self.installed(),
            ["splash-m1-1.0-arm64-macos15", "splash-m1-3.0-arm64-macos15"],
        )

    def test_printed_tester_instructions_install_from_the_published_repo(self):
        self.publish("1.0")
        (self.root / "dist").symlink_to(self.releases)
        shutil.copy2(
            package.ROOT / "dev/tools/install.sh", self.releases / "install.sh"
        )
        hub = self.root / "hub"
        hub.mkdir()

        def upload(path_or_fileobj, path_in_repo, repo_id):
            self.assertEqual(repo_id, "owner/splash-releases")
            data = path_or_fileobj
            if not isinstance(data, bytes):
                data = Path(data).read_bytes()
            (hub / path_in_repo).write_bytes(data)

        printed = io.StringIO()
        with (
            mock.patch.object(publish_test, "ROOT", self.root),
            mock.patch.object(publish_test, "HfApi") as api,
            contextlib.redirect_stdout(printed),
        ):
            api.return_value.upload_file.side_effect = upload
            publish_test.main(["--version", "1.0", "--repo", "owner/splash-releases"])
        # This curl serves the uploaded files only to requests that carry the
        # token, as the private repo does.
        commands = self.root / "commands"
        commands.mkdir()
        curl = commands / "curl"
        curl.write_text(
            f"#!{sys.executable}\n"
            "import shutil, sys\n"
            "arguments = sys.argv[1:]\n"
            "config = arguments[arguments.index('--config') + 1]\n"
            "header = (sys.stdin if config == '-' else open(config)).read()\n"
            "if 'Authorization: Bearer test-token' not in header: sys.exit(22)\n"
            "prefix = 'https://huggingface.co/owner/splash-releases/resolve/main/'\n"
            "if not arguments[-1].startswith(prefix): sys.exit(22)\n"
            f"source = open({str(hub)!r} + '/' + arguments[-1][len(prefix):], 'rb')\n"
            "target = sys.stdout.buffer\n"
            "if '-o' in arguments:\n"
            "    target = open(arguments[arguments.index('-o') + 1], 'wb')\n"
            "shutil.copyfileobj(source, target)\n"
        )
        curl.chmod(0o755)
        result = subprocess.run(
            ["/bin/sh", "-c", printed.getvalue().split("run:\n", 1)[1]],
            env={
                "PATH": str(commands) + os.pathsep + os.environ["PATH"],
                "HOME": str(self.root / "home"),
                "SPLASH_TOKEN": "test-token",
                "SPLASH_BIN_DIR": str(self.bin),
                "SPLASH_APP_DIR": str(self.apps),
            },
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.current(), "splash-m1-1.0-arm64-macos15")
        self.assertIn(
            "Splash M1/app/current/install/launcher.py", self.command.read_text()
        )


if __name__ == "__main__":
    unittest.main()
