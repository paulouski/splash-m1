import http.client
import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

from dev.tests.engine.test_idle_settings import IdleRuntime
from server import server as api
from server import web_tools
from server.backend import NativeBackend
from server.errors import APIError

# Shape of html.duckduckgo.com/html/ results: organic (direct and uddg-wrapped) and an ad.
DDG = """
<div id="links" class="results">
<div class="result results_links web-result ">
 <h2 class="result__title"><a rel="nofollow" class="result__a" href="https://yandex.by/pogoda/ru/minsk">Погода в <b>Минске</b> — прогноз</a></h2>
 <a class="result__snippet" href="https://yandex.by/pogoda/ru/minsk">Сейчас в Минске <b>облачно</b>,
   +17°.</a>
</div>
<div class="result result--ad">
 <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/y.js?ad_domain=shop.example&amp;u3=x">Реклама</a></h2>
 <a class="result__snippet" href="//duckduckgo.com/y.js?u3=x">Купите</a>
</div>
<div class="result results_links web-result ">
 <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.gismeteo.by%2Fweather-minsk-4248%2F%3Fa%3D1&amp;rut=abc">Gismeteo</a></h2>
 <div class="result__extras"><a class="result__url" href="x">gismeteo.by</a></div>
 <a class="result__snippet" href="x">Погода на неделю</a>
</div>
<div class="result"><h2><a class="result__a" href="javascript:alert(1)">Bad scheme</a></h2></div>
</div>
"""

PAGE = """<html><head><title> Тест  страница </title><style>p{}</style></head><body>
<header>Меню сайта</header><nav>Главная О нас</nav>
<h1>Заголовок</h1><p>Первый   абзац
с переносом.</p><ul><li>пункт один</li><li>пункт два</li></ul>
<table><tr><td>ячейка А</td><td>ячейка Б</td></tr></table><pre>код
  строка</pre>
<script>var secret = 1;</script><form><input value="x">Поиск по сайту</form>
<aside>Реклама сбоку</aside><footer>Подвал</footer></body></html>"""


def fake_dns(host, port, **kwargs):
    """Literal addresses and localhost resolve for real; every other name is public."""
    if (
        host == "localhost"
        or host.replace(".", "").replace(":", "").replace("[", "").isdigit()
        or ":" in host
    ):
        return real_getaddrinfo(host, port, **kwargs)
    return public()


real_getaddrinfo = socket.getaddrinfo


def public(address="93.184.216.34"):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))]


class ParseTests(unittest.TestCase):
    def test_results_unwrap_redirects_skip_ads_and_bad_schemes(self):
        results = web_tools.parse_results(DDG)
        self.assertEqual(
            results,
            [
                {
                    "title": "Погода в Минске — прогноз",
                    "url": "https://yandex.by/pogoda/ru/minsk",
                    "snippet": "Сейчас в Минске облачно, +17°.",
                },
                {
                    "title": "Gismeteo",
                    "url": "https://www.gismeteo.by/weather-minsk-4248/?a=1",
                    "snippet": "Погода на неделю",
                },
            ],
        )

    def test_results_are_capped_at_eight(self):
        page = "".join(
            f'<a class="result__a" href="https://e{i}.example/">t{i}</a>'
            for i in range(20)
        )
        self.assertEqual(len(web_tools.parse_results(page)), 8)

    def test_search_returns_results_and_reports_a_block(self):
        with mock.patch.object(web_tools, "_ddg_post", return_value=(200, DDG)) as post:
            self.assertEqual(len(web_tools.search("  погода Минск ")["results"]), 2)
            post.assert_called_once_with("погода Минск")
        with mock.patch.object(
            web_tools, "_ddg_post", return_value=(200, "<html>none</html>")
        ):
            self.assertEqual(web_tools.search("q"), {"results": []})
        blocked = (202, '<div class="anomaly-modal">bots use DuckDuckGo too</div>')
        with mock.patch.object(web_tools, "_ddg_post", return_value=blocked):
            with self.assertRaises(APIError) as caught:
                web_tools.search("q")
        self.assertEqual(
            (caught.exception.status, caught.exception.code), (502, "search_blocked")
        )
        for query in (None, "", "  ", 5, "x" * 501):
            with self.assertRaises(APIError):
                web_tools.search(query)


class TextTests(unittest.TestCase):
    def test_extraction_keeps_content_and_drops_chrome(self):
        title, text = web_tools.extract_text(PAGE + "<p>" + "слово " * 50 + "</p>")
        self.assertEqual(title, "Тест страница")
        for kept in (
            "Заголовок",
            "Первый абзац с переносом.",
            "пункт один",
            "пункт два",
            "ячейка А | ячейка Б",
            "код",
            "строка",
        ):
            self.assertIn(kept, text)
        for dropped in (
            "Меню сайта",
            "Главная",
            "secret",
            "Поиск по сайту",
            "Реклама сбоку",
            "Подвал",
        ):
            self.assertNotIn(dropped, text)
        self.assertNotIn("  ", text)

    def test_unclosed_form_does_not_swallow_a_short_page(self):
        _, text = web_tools.extract_text(
            "<body><form><p>Только это и есть на странице</p></body>"
        )
        self.assertIn("Только это", text)

    def test_short_text_is_returned_whole(self):
        self.assertEqual(web_tools.select_text("abc", "q"), ("abc", False))

    def test_window_with_the_most_query_hits_is_chosen(self):
        filler = [f"строка {i} " + "заполнитель " * 12 for i in range(200)]
        filler[150] = "Температура воздуха в Минске завтра плюс двадцать"
        text = "\n".join(filler)
        chosen, truncated = web_tools.select_text(text, "температура Минске")
        self.assertTrue(truncated)
        self.assertLessEqual(len(chosen), web_tools.TEXT_BUDGET)
        self.assertIn("Температура воздуха в Минске", chosen)
        start, _ = web_tools.select_text(text, None)
        self.assertTrue(start.startswith("строка 0"))
        self.assertLessEqual(len(start), web_tools.TEXT_BUDGET)
        self.assertNotIn("Температура", start)
        # No hits falls back to the start.
        self.assertTrue(web_tools.select_text(text, "квазар")[0].startswith("строка 0"))


class GuardTests(unittest.TestCase):
    def test_non_public_addresses_and_schemes_are_rejected(self):
        for url in (
            "http://127.0.0.1/",
            "http://10.1.2.3/x",
            "http://[::1]/",
            "http://localhost:8080/",
            "http://169.254.169.254/latest",
            "http://192.168.0.1/",
            "http://[::ffff:127.0.0.1]/",
            "http://2130706433/",
            "http://0.0.0.0/",
            "http://224.0.0.1/",
            "http://100.64.0.1/",
        ):
            with self.subTest(url=url), self.assertRaises(APIError) as caught:
                web_tools.check_url(url)
            self.assertEqual(caught.exception.status, 403, url)
        for url in (
            "file:///etc/passwd",
            "ftp://example.com/",
            "http://user:pw@example.com/",
            "http:///nohost",
            "",
            None,
            "http://example.com:99999/",
        ):
            with self.subTest(url=url), self.assertRaises(APIError) as caught:
                web_tools.check_url(url)
            self.assertEqual(caught.exception.status, 400, url)

    def test_a_name_resolving_to_a_private_address_is_rejected_even_when_mixed(self):
        answers = public() + [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 80))
        ]
        with (
            mock.patch("socket.getaddrinfo", return_value=answers),
            self.assertRaises(APIError),
        ):
            web_tools.check_url("http://rebind.example/")
        with mock.patch("socket.getaddrinfo", return_value=public()):
            self.assertEqual(
                web_tools.check_url("https://ok.example/a?b=1"),
                "https://ok.example/a?b=1",
            )

    def test_connection_is_pinned_to_validated_addresses(self):
        private = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
        with (
            mock.patch("socket.getaddrinfo", return_value=private),
            self.assertRaises(APIError),
        ):
            web_tools._connect("rebound.example", 80, 1)


def page_response(status=200, kind="text/html; charset=utf-8", body=b"", location=None):
    headers = {"Content-Type": kind}
    if location:
        headers["Location"] = location
    return status, headers, body


class FetchTests(unittest.TestCase):
    def fetch(self, responses, url="http://site.example/a", query=None):
        seen = []

        def request(target, timeout):
            seen.append(target)
            return responses.pop(0)

        with (
            mock.patch("socket.getaddrinfo", side_effect=fake_dns),
            mock.patch.object(web_tools, "_request", side_effect=request),
        ):
            try:
                return web_tools.fetch(url, query), seen
            except APIError as error:
                return error, seen

    def test_html_page_is_fetched_and_extracted(self):
        result, seen = self.fetch(
            [page_response(body=(PAGE + "<p>" + "слово " * 50 + "</p>").encode())]
        )
        self.assertEqual(seen, ["http://site.example/a"])
        self.assertEqual(result["title"], "Тест страница")
        self.assertFalse(result["truncated"])
        self.assertIn("Заголовок", result["text"])

    def test_redirects_are_followed_and_revalidated(self):
        ok = page_response(body=b"<title>t</title><p>done</p>")
        result, seen = self.fetch(
            [
                page_response(302, location="/b"),
                page_response(301, location="https://other.example/c"),
                ok,
            ]
        )
        self.assertEqual(seen[-1], "https://other.example/c")
        self.assertEqual(result["url"], "https://other.example/c")
        for target in (
            "http://10.0.0.5/admin",
            "http://127.0.0.1:18081/status",
            "http://[::1]/",
            "http://localhost/",
        ):
            error, seen = self.fetch([page_response(302, location=target), ok])
            self.assertIsInstance(error, APIError, target)
            self.assertEqual(error.status, 403, target)
            self.assertEqual(len(seen), 1)

    def test_more_than_three_redirects_fail(self):
        error, seen = self.fetch([page_response(302, location="/n")] * 5)
        self.assertEqual((error.status, error.message), (502, "too many redirects"))
        self.assertEqual(len(seen), 4)

    def test_type_and_status_limits(self):
        for kind in ("application/pdf", "image/png", "application/octet-stream", ""):
            error, _ = self.fetch([page_response(kind=kind, body=b"x")])
            self.assertEqual(error.status, 415, kind)
        error, _ = self.fetch([page_response(404, body=b"no")])
        self.assertEqual(error.status, 502)
        plain, _ = self.fetch([page_response(kind="text/plain", body=b"a  b\n\n c")])
        self.assertEqual(plain["text"], "a b\nc")

    def test_charset_from_header_or_meta(self):
        body = "<title>Привет</title><p>Мир</p>".encode("cp1251")
        header, _ = self.fetch(
            [page_response(kind="text/html; charset=windows-1251", body=body)]
        )
        meta, _ = self.fetch(
            [
                page_response(
                    kind="text/html", body=b'<meta charset="windows-1251">' + body
                )
            ]
        )
        self.assertEqual((header["title"], meta["title"]), ("Привет", "Привет"))

    def test_long_page_is_trimmed_to_the_query_window(self):
        lines = [f"<p>абзац {i} " + "вода " * 20 + "</p>" for i in range(300)]
        lines[200] = "<p>курс доллара сегодня составляет 3,2 рубля</p>"
        result, _ = self.fetch(
            [page_response(body="".join(lines).encode())], query="курс доллара"
        )
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["text"]), web_tools.TEXT_BUDGET)
        self.assertIn("курс доллара сегодня", result["text"])


class Body(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        try:
            self.wfile.write(b"<p>x</p>" * (web_tools.MAX_BYTES // 4))
        except OSError:
            pass

    def log_message(self, *args):
        pass


class SizeCapTest(unittest.TestCase):
    def test_body_is_cut_at_the_size_cap(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Body)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_address[1]

        def loopback(host, _port, timeout):
            return socket.create_connection(("127.0.0.1", port), timeout)

        with mock.patch.object(web_tools, "_connect", loopback):
            status, _, body = web_tools._request(f"http://site.example:{port}/", 5)
        self.assertEqual((status, len(body)), (200, web_tools.MAX_BYTES))


class EndpointTests(unittest.TestCase):
    def setUp(self):
        backend = NativeBackend(
            IdleRuntime(), None, lambda _record: None, idle_unload=0.0
        )
        self.addCleanup(backend.close)
        self.server = api.FrontendServer(
            ("127.0.0.1", 0),
            SimpleNamespace(backend=backend, request_timeout=5),
            api_key="k",
        )
        thread = threading.Thread(target=self.server.serve_forever)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def post(self, path, body, key="k"):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        headers = {
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {key}"} if key else {}),
        }
        connection.request(
            "POST", path, body if isinstance(body, bytes) else json.dumps(body), headers
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        return response.status, payload

    def test_search_and_fetch_routes_use_the_tools_and_need_the_key(self):
        with mock.patch.object(web_tools, "_ddg_post", return_value=(200, DDG)):
            self.assertEqual(
                self.post("/splash/tools/search", {"query": "q"})[1]["results"][1][
                    "title"
                ],
                "Gismeteo",
            )
            self.assertEqual(
                self.post("/splash/tools/search", {"query": "q"}, key=None)[0], 401
            )
            self.assertEqual(self.post("/splash/tools/search", {"query": ""})[0], 400)
            self.assertEqual(self.post("/splash/tools/search", b"[1]")[0], 400)
        page = page_response(body=b"<title>T</title><p>hello</p>")
        with (
            mock.patch.object(web_tools, "resolve"),
            mock.patch.object(web_tools, "_request", return_value=page),
        ):
            status, payload = self.post(
                "/splash/tools/fetch", {"url": "http://a.example/", "query": "hello"}
            )
        self.assertEqual(
            (status, payload),
            (
                200,
                {
                    "url": "http://a.example/",
                    "title": "T",
                    "text": "hello",
                    "truncated": False,
                },
            ),
        )
        status, payload = self.post(
            "/splash/tools/fetch", {"url": "http://127.0.0.1:18081/status"}
        )
        self.assertEqual(status, 403)
        self.assertEqual(
            self.post("/splash/tools/fetch", {"url": "http://a.example/", "query": 5})[
                0
            ],
            400,
        )


if __name__ == "__main__":
    unittest.main()
