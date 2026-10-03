import os
import plistlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import families, hub, paths, uninstall
from dev.tests.engine.test_desktop_models import DRAFT_REVISION, MODEL, REVISION, install


class UninstallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name).resolve()
        self.data = self.home / "Library/Application Support/Splash M1"
        self.weights = self.home / "Library/Caches/Splash/weights"
        self.weights.mkdir(parents=True)
        (self.weights / "entry").write_bytes(b"w")
        self.app = self.home / "Applications/Splash M1.app"
        (self.app / "Contents").mkdir(parents=True)
        with (self.app / "Contents/Info.plist").open("wb") as info:
            plistlib.dump({"CFBundleIdentifier": uninstall.BUNDLE_ID}, info)
        self.wrapper = self.home / ".local/bin/splash-m1"
        self.wrapper.parent.mkdir(parents=True)
        self.wrapper.write_text(f'exec "{self.data}/{uninstall.WRAPPER_MARKER}"\n')
        self.prefs = self.home / f"Library/Preferences/{uninstall.BUNDLE_ID}.plist"
        self.prefs.parent.mkdir(parents=True)
        self.prefs.write_bytes(b"x")
        (self.data / "app/current").mkdir(parents=True)
        # Hub cache entries: one Splash installed, one it never touched.
        self.link = self.data / "models/team/m"
        install(self.home, self.link, MODEL, REVISION)
        self.cache = self.home / "hub"
        unrelated = self.cache / hub.folder_name("someone/else")
        (unrelated / "snapshots" / ("d" * 40)).mkdir(parents=True)
        self.unrelated = unrelated
        self.checkpoint = self.home / "Models/mine"
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "w.safetensors").write_bytes(b"c")
        for patcher in (
            mock.patch.dict(os.environ, {"HOME": str(self.home)}),
            mock.patch.object(paths, "PACKAGED", True),
            mock.patch.object(paths, "DATA", self.data),
            mock.patch.object(paths, "RUNTIME", self.data / "runtime"),
            mock.patch.object(uninstall, "APP_DIRS", ()),
            mock.patch.object(uninstall, "BIN_DIRS", ()),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("SPLASH_WEIGHT_CACHE", None)
        os.environ.pop("SPLASH_APP_DIR", None)
        os.environ.pop("SPLASH_BIN_DIR", None)

    def untouched(self):
        self.assertTrue((self.checkpoint / "w.safetensors").is_file())
        self.assertTrue(self.unrelated.is_dir())

    def test_removes_everything_managed(self):
        self.assertTrue(uninstall.run(from_app=True, echo=lambda *_: None))
        for gone in (self.app, self.wrapper, self.data, self.weights.parent, self.prefs):
            self.assertFalse(gone.exists(), gone)
        self.assertFalse((self.cache / hub.folder_name(MODEL) / "snapshots" / REVISION).exists())
        draft = families.FAMILIES[0].draft.repo
        self.assertFalse((self.cache / hub.folder_name(draft) / "snapshots" / DRAFT_REVISION).exists())
        self.untouched()

    def test_keep_models_and_declined_confirm(self):
        self.assertFalse(uninstall.run(from_app=True, confirm=lambda: False, echo=lambda *_: None))
        self.assertTrue(self.app.exists() and self.link.is_symlink())
        self.assertTrue(uninstall.run(True, from_app=True, echo=lambda *_: None))
        self.assertFalse(self.app.exists() or self.wrapper.exists() or self.prefs.exists())
        self.assertFalse((self.data / "app").exists())
        self.assertTrue(self.link.is_symlink() and self.weights.is_dir())
        self.assertTrue((self.cache / hub.folder_name(MODEL) / "snapshots" / REVISION).is_dir())
        self.untouched()


if __name__ == "__main__":
    unittest.main()
