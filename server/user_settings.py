"""Settings the web UI can change at runtime and that survive restarts."""

import json
import os
import tempfile
from pathlib import Path

if __package__:
    from install import paths

    from .errors import APIError

    DEFAULT_PATH = (
        paths.DATA / "settings.json"
        if paths.PACKAGED
        else Path.home() / "Library/Application Support/Splash/settings.json"
    )
else:
    from errors import APIError

    packaged = (Path(__file__).resolve().parents[1] / "release.json").is_file()
    data_name = "Splash M1" if packaged else "Splash"
    DEFAULT_PATH = (
        Path.home() / "Library/Application Support" / data_name / "settings.json"
    )
MIN_IDLE_UNLOAD_SECONDS = 30
MAX_IDLE_UNLOAD_SECONDS = 86400


def validate_idle_unload(value):
    """0 (never) or 30..86400 seconds."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not (
            value == 0 or MIN_IDLE_UNLOAD_SECONDS <= value <= MAX_IDLE_UNLOAD_SECONDS
        )
    ):
        raise APIError(
            400,
            "idle_unload_seconds must be 0 (never) or between "
            f"{MIN_IDLE_UNLOAD_SECONDS} and {MAX_IDLE_UNLOAD_SECONDS}",
        )
    return float(value)


def load_idle_unload(path):
    """The saved idle-unload seconds, or None when absent or unusable."""
    try:
        return validate_idle_unload(
            json.loads(Path(path).read_text())["idle_unload_seconds"]
        )
    except (OSError, ValueError, KeyError, TypeError, APIError):
        return None


def save_idle_unload(path, seconds):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".settings-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as target:
            json.dump({"idle_unload_seconds": seconds}, target)
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise
