import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from server import server as api
from server import user_settings
from server.backend import NativeBackend

REPO = Path(__file__).resolve().parents[3]


class IdleRuntime:
    """Just the surface the idle-unload loop and /splash/settings touch."""

    def __init__(self):
        self.ready = True
        self.unloaded = False
        self.unloads = 0

    def unload_if_idle(self):
        self.unloads += 1
        self.ready, self.unloaded = False, True
        return True

    def close(self):
        pass


def wait_for(condition, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


class IdleUnloadBackendTests(unittest.TestCase):
    def backend(self, idle_unload=0.0):
        runtime = IdleRuntime()
        backend = NativeBackend(
            runtime, None, lambda _record: None, idle_unload=idle_unload
        )
        self.addCleanup(backend.close)
        return runtime, backend

    def test_zero_never_unloads_and_enabling_at_runtime_takes_effect_promptly(self):
        runtime, backend = self.backend(0.0)
        backend.last_activity = time.monotonic() - 1000
        time.sleep(0.3)
        self.assertEqual(runtime.unloads, 0)
        self.assertIsNone(backend.idle_unload_remaining())
        backend.set_idle_unload(30)
        self.assertTrue(wait_for(lambda: runtime.unloads == 1))

    def test_shortening_wakes_a_loop_sleeping_on_a_long_timeout(self):
        runtime, backend = self.backend(86400.0)
        time.sleep(0.2)
        backend.last_activity = time.monotonic() - 60
        backend.set_idle_unload(30)
        self.assertTrue(wait_for(lambda: runtime.unloads == 1, 1.5))

    def test_remaining_counts_down_only_while_loaded_and_idle(self):
        runtime, backend = self.backend(100.0)
        self.assertTrue(90 < backend.idle_unload_remaining() <= 100)
        backend.active["job"] = object()
        self.assertIsNone(backend.idle_unload_remaining())
        backend.active.clear()
        runtime.ready = False
        self.assertIsNone(backend.idle_unload_remaining())

    def test_close_stops_the_loop(self):
        _, backend = self.backend(0.0)
        backend.close()
        loops = [t for t in threading.enumerate() if t.name == "splash-idle-unload"]
        self.assertTrue(wait_for(lambda: not any(t.is_alive() for t in loops), 1.5))


class SettingsEndpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "Splash" / "settings.json"
        self.runtime = IdleRuntime()
        self.backend = NativeBackend(
            self.runtime, None, lambda _record: None, idle_unload=300.0
        )
        self.addCleanup(self.backend.close)
        self.server = api.FrontendServer(
            ("127.0.0.1", 0),
            SimpleNamespace(backend=self.backend, request_timeout=5),
            api_key="k",
            settings_path=self.path,
        )
        thread = threading.Thread(target=self.server.serve_forever)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, body=None, headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        headers = {"Authorization": "Bearer k"} if headers is None else headers
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body)
            headers = {**headers, "Content-Type": "application/json"}
        connection.request(method, "/splash/settings", data, headers)
        response = connection.getresponse()
        payload = json.loads(response.read() or b"null")
        connection.close()
        return response.status, payload

    def test_get_reports_the_value_and_time_until_unload(self):
        status, payload = self.request("GET")
        self.assertEqual(status, 200)
        self.assertEqual(payload["idle_unload_seconds"], 300)
        self.assertTrue(250 < payload["unload_in_seconds"] <= 300)
        self.runtime.ready = False
        self.assertIsNone(self.request("GET")[1]["unload_in_seconds"])

    def test_post_applies_persists_and_survives_a_reload(self):
        status, payload = self.request("POST", {"idle_unload_seconds": 600})
        self.assertEqual((status, payload["idle_unload_seconds"]), (200, 600))
        self.assertEqual(self.backend.idle_unload, 600)
        self.assertEqual(user_settings.load_idle_unload(self.path), 600)
        self.request("POST", {"idle_unload_seconds": 0})
        self.assertEqual(self.backend.idle_unload, 0)
        self.assertEqual(user_settings.load_idle_unload(self.path), 0)

    def test_invalid_values_are_rejected_and_change_nothing(self):
        for value in (1, 29, 86401, -5, "60", None, True, float("nan")):
            with self.subTest(value=value):
                status, _ = self.request("POST", {"idle_unload_seconds": value})
                self.assertEqual(status, 400)
        self.assertEqual(self.request("POST", {"other": 1})[0], 400)
        self.assertEqual(self.request("POST", b"[1]")[0], 400)
        self.assertEqual(self.backend.idle_unload, 300)
        self.assertFalse(self.path.exists())

    def test_same_authentication_as_status(self):
        for method, body in (("GET", None), ("POST", {"idle_unload_seconds": 60})):
            with self.subTest(method=method):
                self.assertEqual(self.request(method, body, headers={})[0], 401)
        self.assertEqual(self.backend.idle_unload, 300)

    def test_foreign_host_and_origin_are_refused(self):
        for headers in (
            {"Authorization": "Bearer k", "Host": "evil.example"},
            {"Authorization": "Bearer k", "Origin": "http://evil.example"},
        ):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("GET", headers=headers)[0], 403)
                self.assertEqual(
                    self.request("POST", {"idle_unload_seconds": 60}, headers)[0], 403
                )
        self.assertEqual(self.backend.idle_unload, 300)

    def test_unwritable_location_leaves_the_running_value_unchanged(self):
        self.server.settings_path = Path(self.directory.name) / "file" / "settings.json"
        Path(self.directory.name, "file").write_text("not a directory")
        self.assertEqual(self.request("POST", {"idle_unload_seconds": 60})[0], 500)
        self.assertEqual(self.backend.idle_unload, 300)


class SavedSettingTests(unittest.TestCase):
    def test_source_and_packaged_defaults_use_their_data_namespaces(self):
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory) / "app"
            for relative in (
                "install/paths.py",
                "server/errors.py",
                "server/user_settings.py",
            ):
                source = REPO / relative
                target = staged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            home = Path(directory) / "home"
            home.mkdir()
            environment = {
                **os.environ,
                "HOME": str(home),
                "PYTHONPATH": str(staged),
            }
            for packaged, name in ((False, "Splash"), (True, "Splash M1")):
                release = staged / "release.json"
                if packaged:
                    release.write_text("{}")
                else:
                    release.unlink(missing_ok=True)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from server.user_settings import DEFAULT_PATH; print(DEFAULT_PATH)",
                    ],
                    cwd=staged,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                with self.subTest(packaged=packaged):
                    self.assertEqual(
                        result.stdout.strip(),
                        str(
                            home
                            / "Library/Application Support"
                            / name
                            / "settings.json"
                        ),
                    )

    def test_missing_or_corrupt_file_means_no_saved_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            self.assertIsNone(user_settings.load_idle_unload(path))
            for text in (
                "{",
                "[]",
                '{"idle_unload_seconds": 5}',
                '{"idle_unload_seconds": "x"}',
            ):
                path.write_text(text)
                self.assertIsNone(user_settings.load_idle_unload(path), text)
            user_settings.save_idle_unload(path, 0.0)
            self.assertEqual(user_settings.load_idle_unload(path), 0)


if __name__ == "__main__":
    unittest.main()
