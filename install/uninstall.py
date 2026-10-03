"""Remove Splash M1 and everything it created: app, CLI, data, models, settings."""

from __future__ import annotations

import fcntl
import json
import os
import plistlib
import shutil
import signal
import subprocess
import time
from pathlib import Path

if __package__:
    from . import assembly, desktop_models, hub, launcher, models, paths, upstream
else:
    import assembly
    import desktop_models
    import hub
    import launcher
    import models
    import paths
    import upstream

BUNDLE_ID = "io.github.paulouski.splash-m1"
WRAPPER_MARKER = "Splash M1/app/current"
APP_DIRS = (Path("/Applications"),)
BIN_DIRS = (Path("/opt/homebrew/bin"), Path("/usr/local/bin"))


def _stop_server():
    try:
        lock = (paths.RUNTIME / f"serve-{launcher.PORT}.lock").open("r")
    except FileNotFoundError:
        return
    with lock:
        deadline = time.monotonic() + 30
        signalled = False
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise models.ModelError("could not stop the running Splash server")
                if not signalled:
                    lock.seek(0)
                    try:
                        pid = json.load(lock)["pid"]
                    except (OSError, ValueError, KeyError, TypeError):
                        raise models.ModelError("Splash is serving; stop it first") from None
                    if type(pid) is not int or pid <= 0:
                        raise models.ModelError("Splash is serving; stop it first")
                    os.kill(pid, signal.SIGTERM)
                    signalled = True
                time.sleep(0.2)
                continue
            fcntl.flock(lock, fcntl.LOCK_UN)
            return


def _size(path, skip=None):
    total = 0
    for file in path.rglob("*"):
        if skip is not None and skip in file.parents:
            continue
        if file.is_file() and not file.is_symlink():
            total += file.stat().st_size
    return total


def _is_splash_bundle(path):
    try:
        with (path / "Contents/Info.plist").open("rb") as info:
            return plistlib.load(info).get("CFBundleIdentifier") == BUNDLE_ID
    except (OSError, plistlib.InvalidFileException):
        return False


def _is_splash_wrapper(path):
    try:
        return path.is_file() and WRAPPER_MARKER in path.read_text()
    except (OSError, UnicodeDecodeError):
        return False


def _plan(keep_models):
    if not paths.PACKAGED:
        raise models.ModelError(
            "uninstall applies to the installed app; remove a source checkout by hand"
        )
    home = Path.home()
    data = home / "Library/Application Support/Splash M1"
    if paths.DATA != data:
        raise models.ModelError(f"unexpected Splash data folder {paths.DATA}")
    models_root = data / "models"
    items = []

    def add(label, path, *, size=None, skip=None):
        if path.is_symlink() or path.exists():
            items.append((label, path, _size(path, skip) if size is None else size))

    bundles = [Path(d) for d in (os.environ.get("SPLASH_APP_DIR"),) if d]
    bundles = [d / "Splash M1.app" for d in [*bundles, *APP_DIRS, home / "Applications"]]
    running = paths.ROOT.parents[2] if len(paths.ROOT.parents) > 2 else None
    if running is not None and running.suffix == ".app":
        bundles.append(running)
    for bundle in dict.fromkeys(bundles):
        if _is_splash_bundle(bundle):
            add("App", bundle)
    bins = [Path(d) for d in (os.environ.get("SPLASH_BIN_DIR"),) if d]
    for directory in [*bins, home / ".local/bin", *BIN_DIRS]:
        if _is_splash_wrapper(directory / "splash-m1"):
            add("Command", directory / "splash-m1")
    add("Install home", data, skip=models_root)

    drafts, model_ids = [], []
    if not keep_models:
        links = models.selection_links(models_root) if models_root.is_dir() else []
        owners = {hub.pin_owner(link) for link in links}
        draft_repos = desktop_models.draft_repos()
        for link in links:
            try:
                model_ids.append(models.read_json(link / "model.json")["model"])
            except (models.ModelError, KeyError):
                continue
            for snapshot, repo in assembly.recorded_pins(link):
                if (
                    repo in draft_repos
                    and snapshot.is_dir()
                    and snapshot not in drafts
                    and desktop_models._only_splash_pins(snapshot, owners)
                ):
                    drafts.append(snapshot)
        model_ids = sorted(set(model_ids))
        model_bytes = sum(
            desktop_models.delete_plan(model, models_root)["bytes"] for model in model_ids
        )
        draft_bytes = sum(
            desktop_models._revision_strategy(s).expected_freed_size for s in drafts
        )
        if model_ids or drafts:
            items.append(("Models and draft", models_root, model_bytes + draft_bytes))
        default_cache = home / "Library/Caches/Splash"
        if default_cache.exists() and upstream._weight_cache().parent == default_cache:
            add("Weight cache", default_cache)
    for label, path in (
        ("App settings", home / f"Library/Preferences/{BUNDLE_ID}.plist"),
        ("App cache", home / f"Library/Caches/{BUNDLE_ID}"),
        ("Saved window state", home / f"Library/Saved Application State/{BUNDLE_ID}.savedState"),
    ):
        add(label, path)
    return items, model_ids, drafts, models_root


def _remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _gb(size):
    return f"{size / 1e9:.1f} GB"


def run(keep_models=False, *, confirm=None, echo=print, from_app=False):
    """Remove Splash. Returns False when confirm declines."""
    if not from_app and subprocess.run(
        ["pgrep", "-x", "SplashM1"], capture_output=True
    ).returncode == 0:
        raise models.ModelError("quit Splash M1 before uninstalling.")
    _stop_server()
    items, model_ids, drafts, models_root = _plan(keep_models)
    echo("Splash M1 uninstall will remove:")
    for label, path, size in items:
        echo(f"  {label}: {path} ({_gb(size)})")
    if keep_models:
        echo(f"  Kept: downloaded models and weight cache ({models_root})")
    echo("Local checkpoints you passed by path are never touched.")
    if confirm is not None and not confirm():
        return False
    if not keep_models:
        for model in model_ids:
            desktop_models.delete(model, models_root)
        for snapshot in drafts:
            if snapshot.is_dir() and desktop_models._only_splash_pins(snapshot, set()):
                desktop_models._revision_strategy(snapshot).execute()
    data = paths.DATA
    for _label, path, _size_bytes in items:
        if path == models_root:
            continue
        if path == data and keep_models:
            for child in data.iterdir():
                if child != models_root:
                    _remove(child)
        else:
            _remove(path)
    return True
