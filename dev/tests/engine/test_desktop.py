import io
import json
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from install import desktop


class EventOutput:
    def __init__(self):
        self.events = queue.Queue()

    def write(self, line):
        self.events.put(json.loads(line))

    def flush(self):
        pass

    def wait_for(self, event_type, timeout=5):
        deadline = time.monotonic() + timeout
        seen = []
        while time.monotonic() < deadline:
            try:
                event = self.events.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            seen.append(event)
            if event.get("type") == event_type:
                return event, seen
        raise AssertionError(f"did not receive {event_type}: {seen}")

    def drain(self):
        events = []
        while True:
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                return events


class SupervisorHarness:
    def __init__(
        self,
        *,
        preflight,
        command_factory,
        process_factory=subprocess.Popen,
        output=None,
    ):
        self.read_fd, self.write_fd = os.pipe()
        self.output = output or EventOutput()
        self.thread = threading.Thread(
            target=desktop.supervise,
            args=(self.read_fd, self.output),
            kwargs={
                "preflight": preflight,
                "command_factory": command_factory,
                "process_factory": process_factory,
            },
            daemon=True,
        )
        self.thread.start()

    def send(self, action):
        os.write(self.write_fd, (json.dumps({"action": action}) + "\n").encode())

    def close(self):
        try:
            os.close(self.write_fd)
        except OSError:
            pass
        self.thread.join(timeout=5)
        try:
            os.close(self.read_fd)
        except OSError:
            pass


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.signal = desktop._signal_received
        desktop._signal_received = False

    def tearDown(self):
        desktop._signal_received = self.signal

    def test_supervisor_waits_for_explicit_start(self):
        checks = []
        harness = SupervisorHarness(
            preflight=lambda output, control: checks.append("preflight"),
            command_factory=lambda: self.fail("launcher must not start"),
        )
        try:
            time.sleep(0.15)
            self.assertEqual(checks, [])
            harness.send("quit")
            harness.thread.join(timeout=2)
            self.assertFalse(harness.thread.is_alive())
            self.assertEqual(checks, [])
        finally:
            harness.close()

    def test_packaged_check_only_reads_bundle_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / "python/bin/python3"
            binary = root / "engine/splash"
            (root / "install").mkdir()
            python.parent.mkdir(parents=True)
            binary.parent.mkdir(parents=True)
            (binary.parent / "splash.metallib").touch()
            (root / "release.json").write_text("{}")
            (root / "install/launcher.py").write_text("")
            (root / "install/desktop.py").write_text("")
            python.touch()
            binary.touch()
            python.chmod(0o755)
            binary.chmod(0o755)
            with (
                mock.patch.object(desktop, "ROOT", root),
                mock.patch.object(desktop, "PYTHON", python),
                mock.patch.object(desktop, "BINARY", binary),
                mock.patch.object(desktop.paths, "PACKAGED", True),
                mock.patch("sys.stdout", new_callable=io.StringIO) as output,
                mock.patch.object(desktop.subprocess, "run") as run,
            ):
                desktop._check_package()
            self.assertEqual(json.loads(output.getvalue())["type"], "checked")
            run.assert_not_called()

    def test_busy_port_refuses_before_launcher_process(self):
        with tempfile.TemporaryDirectory() as temporary, socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with (
                mock.patch.object(desktop.platform, "system", return_value="Darwin"),
                mock.patch.object(desktop.platform, "machine", return_value="arm64"),
                mock.patch.object(
                    desktop.platform, "mac_ver", return_value=("15.0", ())
                ),
                mock.patch.object(
                    desktop, "_memory_bytes", return_value=desktop.MIN_MEMORY_BYTES
                ),
                mock.patch.object(desktop, "_device_check"),
                mock.patch.object(desktop, "RUNTIME_DIR", Path(temporary) / "runtime"),
                mock.patch.object(desktop, "DATA_DIR", Path(temporary)),
                mock.patch.object(desktop, "PORT", listener.getsockname()[1]),
                mock.patch("install.launcher._check_port", side_effect=OSError("busy")),
            ):
                launched = mock.Mock()
                harness = SupervisorHarness(
                    preflight=desktop._preflight,
                    command_factory=lambda: [sys.executable, "-c", "pass"],
                    process_factory=launched,
                )
                try:
                    harness.send("start")
                    event, _ = harness.output.wait_for("error")
                    self.assertIn("already in use", event["message"])
                    launched.assert_not_called()
                    harness.send("quit")
                    harness.thread.join(timeout=2)
                    self.assertFalse(harness.thread.is_alive())
                finally:
                    harness.close()

    def test_ready_from_another_listener_is_not_trusted(self):
        class ReadyHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_args):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            server = ThreadingHTTPServer(("127.0.0.1", 0), ReadyHandler)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            script = Path(temporary) / "fake_launcher.py"
            script.write_text(
                "import time\nprint('Ready · fake', flush=True)\ntime.sleep(60)\n"
            )
            with (
                mock.patch.object(desktop, "RUNTIME_DIR", Path(temporary) / "runtime"),
                mock.patch.object(desktop, "PORT", server.server_port),
            ):
                harness = SupervisorHarness(
                    preflight=lambda _output, _control: None,
                    command_factory=lambda: [sys.executable, str(script)],
                )
                try:
                    harness.send("start")
                    harness.output.wait_for("status")
                    time.sleep(0.6)
                    self.assertFalse(
                        any(
                            event.get("type") == "ready"
                            for event in harness.output.drain()
                        )
                    )
                    harness.send("stop")
                    harness.output.wait_for("stopped")
                finally:
                    harness.close()
                    server.shutdown()
                    server.server_close()
                    server_thread.join(timeout=2)

    def test_stop_kills_fake_launcher_and_its_descendant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_file = root / "descendant.pid"
            script = root / "fake_launcher.py"
            script.write_text(
                "import os, subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
                "time.sleep(120)\n"
            )
            with mock.patch.object(desktop, "RUNTIME_DIR", root / "runtime"):
                harness = SupervisorHarness(
                    preflight=lambda _output, _control: None,
                    command_factory=lambda: [sys.executable, str(script)],
                )
                try:
                    harness.send("start")
                    deadline = time.monotonic() + 5
                    while not pid_file.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(pid_file.exists())
                    child_pid = int(pid_file.read_text())
                    harness.send("stop")
                    harness.output.wait_for("stopped")
                    harness.send("quit")
                    harness.thread.join(timeout=2)
                    self.assertFalse(harness.thread.is_alive())
                    self.assert_process_gone(child_pid)
                finally:
                    harness.close()

    def test_input_eof_kills_owned_launcher_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_file = root / "descendant.pid"
            script = root / "fake_launcher.py"
            script.write_text(
                "import subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
                "print('running', flush=True)\n"
                "time.sleep(120)\n"
            )
            with mock.patch.object(desktop, "RUNTIME_DIR", root / "runtime"):
                harness = SupervisorHarness(
                    preflight=lambda _output, _control: None,
                    command_factory=lambda: [sys.executable, str(script)],
                )
                try:
                    harness.send("start")
                    deadline = time.monotonic() + 5
                    while not pid_file.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(pid_file.exists())
                    child_pid = int(pid_file.read_text())
                    os.close(harness.write_fd)
                    harness.write_fd = -1
                    harness.thread.join(timeout=5)
                    self.assertFalse(harness.thread.is_alive())
                    self.assert_process_gone(child_pid)
                finally:
                    harness.close()

    def test_broken_output_pipe_kills_owned_launcher_tree(self):
        class BrokenOutput(EventOutput):
            def write(self, line):
                event = json.loads(line)
                if event.get("type") == "log":
                    raise BrokenPipeError()
                super().write(line)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_file = root / "descendant.pid"
            script = root / "fake_launcher.py"
            script.write_text(
                "import subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
                "print('running', flush=True)\n"
                "time.sleep(120)\n"
            )
            children = []

            def launch(*args, **kwargs):
                process = subprocess.Popen(*args, **kwargs)
                children.append(process)
                return process

            with mock.patch.object(desktop, "RUNTIME_DIR", root / "runtime"):
                harness = SupervisorHarness(
                    preflight=lambda _output, _control: None,
                    command_factory=lambda: [sys.executable, str(script)],
                    process_factory=launch,
                    output=BrokenOutput(),
                )
                try:
                    harness.send("start")
                    deadline = time.monotonic() + 5
                    while not pid_file.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(pid_file.exists())
                    child_pid = int(pid_file.read_text())
                    harness.thread.join(timeout=5)
                    self.assertFalse(harness.thread.is_alive())
                    self.assert_process_gone(child_pid)
                    self.assertTrue(children)
                    self.assertIsNotNone(children[0].poll())
                finally:
                    harness.close()

    def test_failed_start_can_retry(self):
        class ReadyHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_args):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = ThreadingHTTPServer(("127.0.0.1", 0), ReadyHandler)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            lock_path = root / "runtime" / f"serve-{server.server_port}.lock"
            lock_path.parent.mkdir(parents=True)
            script = root / "ready_launcher.py"
            script.write_text(
                "import fcntl, http.server, json, os, time\n"
                f"lock = open({str(lock_path)!r}, 'a+')\n"
                "fcntl.flock(lock, fcntl.LOCK_EX)\n"
                "lock.seek(0); lock.truncate()\n"
                f"json.dump({{'pid': os.getpid(), 'model': {desktop.MODEL!r}, 'port': {server.server_port}}}, lock)\n"
                "lock.flush()\n"
                "print('Ready · fake', flush=True)\n"
                "time.sleep(120)\n"
            )
            attempts = 0

            def command():
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    return [sys.executable, "-c", "raise SystemExit(7)"]
                return [sys.executable, str(script)]

            with (
                mock.patch.object(desktop, "RUNTIME_DIR", root / "runtime"),
                mock.patch.object(desktop, "PORT", server.server_port),
            ):
                harness = SupervisorHarness(
                    preflight=lambda _output, _control: None,
                    command_factory=command,
                )
                try:
                    harness.send("start")
                    first, _ = harness.output.wait_for("error")
                    self.assertIn("code 7", first["message"])
                    harness.send("start")
                    ready, _ = harness.output.wait_for("ready")
                    self.assertEqual(
                        ready["url"], f"http://127.0.0.1:{server.server_port}"
                    )
                    harness.send("stop")
                    harness.output.wait_for("stopped")
                    self.assertEqual(attempts, 2)
                finally:
                    harness.close()
                    server.shutdown()
                    server.server_close()
                    server_thread.join(timeout=2)

    @staticmethod
    def assert_process_gone(pid):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        raise AssertionError(f"process {pid} is still alive")


if __name__ == "__main__":
    unittest.main()
