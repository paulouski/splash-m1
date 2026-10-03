import http.client
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from server import server as api


class StaticTests(unittest.TestCase):
    def setUp(self):
        self.server = api.FrontendServer(
            ("127.0.0.1", 0), SimpleNamespace(request_timeout=5), api_key="k"
        )
        thread = threading.Thread(target=self.server.serve_forever)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def get(self, path):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        connection.request("GET", path)  # no Authorization: static is public
        response = connection.getresponse()
        body = response.read()
        connection.close()
        return response, body

    def test_serves_files_publicly_with_type_and_cache(self):
        response, body = self.get("/static/katex/katex.min.js")
        self.assertEqual(response.status, 200)
        self.assertTrue(
            response.getheader("Content-Type").startswith("text/javascript")
        )
        self.assertEqual(response.getheader("Cache-Control"), "max-age=86400")
        self.assertIn(b"katex", body)
        self.assertEqual(self.get("/static/katex/katex.min.css?v=1")[0].status, 200)
        response, body = self.get("/static/katex/fonts/KaTeX_Main-Regular.woff2")
        self.assertEqual(
            (response.status, response.getheader("Content-Type")), (200, "font/woff2")
        )

    def test_traversal_to_an_allowed_extension_outside_the_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "static"
            root.mkdir()
            (root / "ok.js").write_text("ok")
            (root.parent / "secret.js").write_text("secret")
            (root / "link.js").symlink_to(root.parent / "secret.js")
            with mock.patch.object(api, "STATIC_DIR", root):
                self.assertEqual(self.get("/static/ok.js")[1], b"ok")
                for path in (
                    "/static/../secret.js",
                    "/static/%2e%2e/secret.js",
                    "/static/..%2fsecret.js",
                    "/static/link.js",
                    f"/static/{root.parent}/secret.js",
                ):
                    # A path that leaves the root is routed as what it names,
                    # which needs the key; the rest are not found.
                    response, body = self.get(path)
                    self.assertIn(response.status, (401, 404), path)
                    self.assertNotIn(b"secret", body, path)

    def test_rejects_traversal_and_unknown(self):
        for path in (
            "/static/../server.py",
            "/static/katex/../../server.py",
            "/static/%2e%2e/server.py",
            "/static/..%2fserver.py",
            "/static/katex/%2e%2e/%2e%2e/chat.html",
            "/static//etc/passwd",
            "/static/%2Fetc/hosts",
            "/static/katex/LICENSE",
            "/static/katex/nope.js",
            "/static/katex",
            "/static/katex/%00.js",
        ):
            self.assertIn(self.get(path)[0].status, (401, 404), path)


if __name__ == "__main__":
    unittest.main()
