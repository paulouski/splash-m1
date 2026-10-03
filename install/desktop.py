"""Small process supervisor for the packaged Splash M1 desktop app."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import select
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if __name__ == "__main__" and not __package__:
    # Run as a script by the app: import siblings as the install package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "install"

from . import desktop_models, launcher, paths, uninstall

ROOT = paths.ROOT
MODEL = "mlx-community/Qwen3.8-27B-4bit"
PORT = 8000
MIN_MEMORY_BYTES = 16 * 1024**3
RUNTIME_DIR = paths.RUNTIME
DATA_DIR = paths.DATA
PYTHON = paths.PYTHON
BINARY = paths.BINARY
STOP_GRACE_SECONDS = 20

_signal_received = False


class DesktopError(RuntimeError):
    pass


class DesktopCancelled(Exception):
    pass


def _event(output, kind, **fields):
    output.write(json.dumps({"type": kind, **fields}, ensure_ascii=True) + "\n")
    output.flush()


def _status(output, stage, message):
    _event(output, "status", stage=stage, message=message)


def _memory_bytes():
    result = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "hw.memsize"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode:
        raise DesktopError("Could not check this Mac's installed memory.")
    try:
        return int(result.stdout.strip())
    except ValueError:
        raise DesktopError("Could not read this Mac's installed memory.") from None


def _version_tuple(version):
    try:
        return tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        return ()


def _check_cancel(control):
    if _signal_received:
        raise DesktopCancelled
    control.poll()
    while control.pending:
        action, _selected_model = control.pending.pop(0)
        if action in ("stop", "quit"):
            control.cancelled = action
            raise DesktopCancelled
    if control.eof:
        control.cancelled = "quit"
        raise DesktopCancelled


def _nearest_existing(path):
    path = Path(path)
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def _ensure_available(output):
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = RUNTIME_DIR / f"serve-{PORT}.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DesktopError(
                f"Splash is already serving{launcher._serve_lock_owner(lock)}."
            ) from None
        finally:
            try:
                fcntl.flock(lock, fcntl.LOCK_UN)
            except OSError:
                pass
    try:
        launcher._check_port("127.0.0.1", PORT)
    except OSError as error:
        raise DesktopError(
            f"Port {PORT} is already in use. Stop the service using it, then retry."
        ) from error


def _device_check():
    result = subprocess.run(
        [str(BINARY), "device-check"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        report = result.stderr.strip() or result.stdout.strip()
        detail = report.splitlines()[-1].removeprefix("error: ") if report else ""
        raise DesktopError(detail or "Splash's native device check failed.")


def _preflight(output, control):
    _status(output, "preflight", "Checking this Mac…")
    if platform.system() != "Darwin":
        raise DesktopError("Splash M1 requires macOS 15 or newer on Apple silicon.")
    if platform.machine() != "arm64":
        raise DesktopError("Splash M1 requires an arm64 Apple silicon Mac.")
    if _version_tuple(platform.mac_ver()[0]) < (15, 0):
        raise DesktopError("Splash M1 requires macOS 15 or newer.")
    if _memory_bytes() < MIN_MEMORY_BYTES:
        raise DesktopError(
            "The current Splash M1 app configuration requires 16 GB of memory."
        )
    _check_cancel(control)

    _status(output, "preflight", "Checking the Splash engine…")
    _device_check()
    _check_cancel(control)

    _status(output, "preflight", f"Checking port {PORT}…")
    _ensure_available(output)
    _check_cancel(control)

    try:
        available = shutil.disk_usage(_nearest_existing(DATA_DIR)).free
    except OSError:
        available = None
    if available is not None:
        _event(output, "disk", available_bytes=available)
        _event(
            output,
            "log",
            message=(
                "Available disk space is shown for reference; the installer will "
                "report actual download sizes and reuse cached weights."
            ),
        )


def _launcher_command(model=MODEL):
    return [
        str(PYTHON),
        "-u",
        str(ROOT / "install/desktop_models.py"),
        "serve",
        "--model",
        model,
    ]


def _model_check_command(model):
    return [
        str(PYTHON),
        "-u",
        str(ROOT / "install/desktop_models.py"),
        "check",
        "--model",
        model,
    ]


def _emit_models(output):
    try:
        installed = desktop_models.installed_models()
    except (OSError, desktop_models.models.ModelError):
        installed = []
    _event(output, "models", models=installed)


def _delete_action(output, action, model, *, busy=False):
    try:
        if busy:
            raise desktop_models.models.ModelError(
                "Stop Splash before deleting a model."
            )
        model = desktop_models.normalize_model_id(model)
        if action == "delete_plan":
            _event(
                output,
                "delete_plan",
                model=model,
                bytes=desktop_models.delete_plan(model)["bytes"],
            )
            return
        desktop_models.delete(model)
        _event(output, "deleted", model=model)
        _emit_models(output)
    except (desktop_models.models.ModelError, OSError) as error:
        _event(output, "error", message=str(error))


def _uninstall_action(output, action, *, busy=False):
    try:
        if busy:
            raise desktop_models.models.ModelError("Stop Splash before uninstalling.")
        uninstall.run(action == "uninstall_keep", echo=lambda *_: None, from_app=True)
        _event(output, "uninstalled")
    except (desktop_models.models.ModelError, OSError) as error:
        _event(output, "error", message=str(error))


class _Control:
    def __init__(self, fd):
        self.fd = fd
        self.buffer = bytearray()
        self.pending = []
        self.eof = False
        self.cancelled = None
        os.set_blocking(fd, False)

    def poll(self):
        if self.eof:
            return
        while select.select([self.fd], [], [], 0)[0]:
            try:
                chunk = os.read(self.fd, 4096)
            except OSError:
                self.eof = True
                return
            if not chunk:
                self.eof = True
                return
            self.buffer.extend(chunk)
            while b"\n" in self.buffer:
                line, _, rest = self.buffer.partition(b"\n")
                self.buffer[:] = rest
                try:
                    command = json.loads(line)
                except (UnicodeDecodeError, ValueError):
                    continue
                if isinstance(command, dict) and command.get("action") in (
                    "models",
                    "delete_plan",
                    "delete",
                    "uninstall",
                    "uninstall_keep",
                    "check",
                    "start",
                    "stop",
                    "quit",
                ):
                    self.pending.append((command["action"], command.get("model")))


def _owns_serve_lock(pid, model=MODEL):
    lock_path = RUNTIME_DIR / f"serve-{PORT}.lock"
    try:
        with lock_path.open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.seek(0)
                try:
                    owner = json.load(lock)
                except (OSError, UnicodeDecodeError, ValueError):
                    return False
                return (
                    isinstance(owner, dict)
                    and owner.get("pid") == pid
                    and owner.get("model") == model
                    and owner.get("port") == PORT
                )
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)
                return False
    except OSError:
        return False


def _ready_endpoint():
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/ready", timeout=0.5
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return False


def _group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + STOP_GRACE_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        if process.returncode is not None and not _group_exists(process.pid):
            break
        time.sleep(0.1)
    if _group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if process.stdout is not None:
        process.stdout.close()


def _lines(buffer):
    while True:
        positions = [
            position
            for marker in (b"\n", b"\r")
            if (position := buffer.find(marker)) >= 0
        ]
        if not positions:
            return
        position = min(positions)
        line = bytes(buffer[:position])
        del buffer[: position + 1]
        if buffer[:1] == b"\n" and line.endswith(b"\r"):
            del buffer[:1]
        yield line.decode("utf-8", errors="replace")[:4096]


def _terminate_child(process, output):
    _status(output, "stopping", "Stopping Splash…")
    _stop_group(process)
    _event(output, "stopped")


def _monitor(process, control, output, model=MODEL, *, checking=False):
    fd = process.stdout.fileno()
    os.set_blocking(fd, False)
    buffer = bytearray()
    saw_ready_line = False
    ready_sent = False
    stop_requested = None
    error_message = None
    while True:
        if _signal_received:
            stop_requested = "quit"
        control.poll()
        while control.pending:
            action, _selected_model = control.pending.pop(0)
            if action == "models":
                _emit_models(output)
            elif action in ("delete_plan", "delete"):
                _delete_action(output, action, _selected_model, busy=True)
            elif action in ("uninstall", "uninstall_keep"):
                _uninstall_action(output, action, busy=True)
            elif action in ("stop", "quit"):
                stop_requested = action
        if control.eof:
            stop_requested = "quit"
        if stop_requested:
            _terminate_child(process, output)
            return stop_requested

        readable, _, _ = select.select([control.fd, fd], [], [], 0.25)
        for ready_fd in readable:
            if ready_fd == control.fd:
                control.poll()
                continue
            try:
                chunk = os.read(fd, 8192)
            except BlockingIOError:
                continue
            if not chunk:
                continue
            buffer.extend(chunk)
            for line in _lines(buffer):
                _event(output, "log", message=line)
                if line.startswith("error: ") and not error_message:
                    error_message = line.removeprefix("error: ")
                if "Ready · " in line:
                    saw_ready_line = True

        if (
            not ready_sent
            and saw_ready_line
            and _owns_serve_lock(process.pid, model)
            and _ready_endpoint()
        ):
            ready_sent = True
            _event(
                output,
                "ready",
                url=f"http://127.0.0.1:{PORT}",
                model=model,
            )
            _emit_models(output)

        if process.poll() is not None:
            for line in _lines(buffer):
                _event(output, "log", message=line)
                if line.startswith("error: ") and not error_message:
                    error_message = line.removeprefix("error: ")
            _stop_group(process)
            if checking and process.returncode == 0:
                _event(
                    output,
                    "model_checked",
                    model=model,
                    message=desktop_models.COMPATIBILITY_MESSAGE,
                )
            elif checking:
                _event(
                    output,
                    "error",
                    message=error_message
                    or f"Model check failed (code {process.returncode}).",
                )
            elif ready_sent:
                _event(output, "stopped")
            else:
                _event(
                    output,
                    "error",
                    message=error_message
                    or f"Splash exited during setup (code {process.returncode}).",
                )
            return None


def supervise(
    input_fd,
    output,
    *,
    preflight=_preflight,
    command_factory=_launcher_command,
    process_factory=subprocess.Popen,
):
    control = _Control(input_fd)
    while True:
        if _signal_received:
            return
        control.poll()
        if control.eof:
            return
        if not control.pending:
            select.select([input_fd], [], [], 0.25)
            continue
        action, selected_model = control.pending.pop(0)
        if action == "quit":
            return
        if action == "models":
            try:
                _emit_models(output)
            except BrokenPipeError:
                return
            continue
        if action in ("delete_plan", "delete"):
            try:
                _delete_action(output, action, selected_model)
            except BrokenPipeError:
                return
            continue
        if action in ("uninstall", "uninstall_keep"):
            try:
                _uninstall_action(output, action)
            except BrokenPipeError:
                return
            continue
        if action == "check":
            process = None
            try:
                model = desktop_models.normalize_model_id(
                    MODEL if selected_model is None else selected_model
                )
                _status(output, "checking", "Checking model metadata…")
                environment = dict(
                    os.environ,
                    PYTHONUNBUFFERED="1",
                    TRANSFORMERS_VERBOSITY="error",
                )
                process = process_factory(
                    _model_check_command(model),
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    env=environment,
                    start_new_session=True,
                )
                if _monitor(process, control, output, model, checking=True) == "quit":
                    return
            except BrokenPipeError:
                return
            except (
                DesktopError,
                desktop_models.models.ModelError,
                OSError,
                subprocess.SubprocessError,
            ) as error:
                try:
                    _event(output, "error", message=str(error))
                except BrokenPipeError:
                    return
            finally:
                if process is not None:
                    if process.poll() is None or _group_exists(process.pid):
                        _stop_group(process)
                    if process.stdout is not None and not process.stdout.closed:
                        process.stdout.close()
            continue
        if action != "start":
            continue
        process = None
        try:
            model = desktop_models.normalize_model_id(
                MODEL if selected_model is None else selected_model
            )
            _preflight_with_cancel(preflight, output, control)
            _event(
                output,
                "status",
                stage="checking",
                message="Checking model compatibility before download…",
            )
            _check_cancel(control)
            environment = dict(
                os.environ,
                PYTHONUNBUFFERED="1",
                TRANSFORMERS_VERBOSITY="error",
            )
            process = process_factory(
                command_factory(model),
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=environment,
                start_new_session=True,
            )
            _status(output, "starting", "Starting Splash…")
            if _monitor(process, control, output, model) == "quit":
                return
        except BrokenPipeError:
            return
        except DesktopCancelled:
            try:
                _event(output, "stopped")
            except BrokenPipeError:
                return
            if control.cancelled == "quit" or control.eof or _signal_received:
                return
        except (
            DesktopError,
            desktop_models.models.ModelError,
            OSError,
            subprocess.SubprocessError,
        ) as error:
            try:
                _event(output, "error", message=str(error))
            except BrokenPipeError:
                return
        finally:
            if process is not None:
                if process.poll() is None or _group_exists(process.pid):
                    _stop_group(process)
                if process.stdout is not None and not process.stdout.closed:
                    process.stdout.close()


def _preflight_with_cancel(preflight, output, control):
    _check_cancel(control)
    preflight(output, control)
    _check_cancel(control)


def _check_package():
    required = (
        ROOT / "release.json",
        PYTHON,
        BINARY,
        ROOT / "install/desktop.py",
        ROOT / "install/desktop_models.py",
        ROOT / "install/launcher.py",
        ROOT / "engine/splash.metallib",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if not paths.PACKAGED or missing:
        raise DesktopError(
            "The bundled Splash runtime is incomplete."
            + (" Missing: " + ", ".join(missing) if missing else "")
        )
    if not os.access(PYTHON, os.X_OK) or not os.access(BINARY, os.X_OK):
        raise DesktopError("The bundled Splash runtime is not executable.")
    print(json.dumps({"type": "checked", "root": str(ROOT)}))


def _handle_signal(_signum, _frame):
    global _signal_received
    _signal_received = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the relocated packaged runtime without starting Splash",
    )
    args = parser.parse_args(argv)
    if args.check:
        try:
            _check_package()
        except DesktopError as error:
            parser.error(str(error))
        return 0
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    supervise(sys.stdin.fileno(), sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
