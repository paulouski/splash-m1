"""Web search and single-page reading for the chat UI's tool calling (stdlib only)."""

import http.client
import ipaddress
import re
import socket
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlsplit

if __package__:
    from .errors import APIError
else:
    from errors import APIError

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.5 Safari/605.1.15"
)
ACCEPT_LANGUAGE = "ru,en;q=0.8"
SEARCH_URL = "https://html.duckduckgo.com/html/"
SEARCH_TIMEOUT = 10
FETCH_TIMEOUT = 15
MAX_RESULTS = 8
MAX_REDIRECTS = 3
MAX_BYTES = 2 * 1024 * 1024
TEXT_BUDGET = 6000
CHUNK = 1000
TEXT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")
MAX_QUERY = 500
MAX_URL = 2048


def _unwrap(href):
    """The target of a DuckDuckGo redirect link, or the link itself."""
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    parts = urlsplit(href)
    host = (parts.hostname or "").lower()
    if host.endswith("duckduckgo.com") and parts.path.startswith("/l/"):
        target = parse_qs(parts.query).get("uddg")
        return target[0] if target else ""
    return href


class _ResultParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.field = None
        self.tag = None
        self.depth = 0
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = (attrs.get("class") or "").split()
        if self.field:
            if tag == self.tag:
                self.depth += 1
            return
        if tag == "a" and "result__a" in classes:
            self.current = {
                "title": "",
                "url": _unwrap(attrs.get("href") or ""),
                "snippet": "",
            }
            self.results.append(self.current)
            self.field, self.tag, self.depth = "title", tag, 1
        elif "result__snippet" in classes and self.current is not None:
            self.field, self.tag, self.depth = "snippet", tag, 1

    def handle_endtag(self, tag):
        if self.field and tag == self.tag:
            self.depth -= 1
            if self.depth == 0:
                self.field = None

    def handle_data(self, data):
        if self.field:
            self.current[self.field] += data


def parse_results(page):
    """Up to MAX_RESULTS organic results from DuckDuckGo's HTML page."""
    parser = _ResultParser()
    parser.feed(page)
    parser.close()
    found = []
    for item in parser.results:
        parts = urlsplit(item["url"])
        if parts.scheme not in ("http", "https") or not parts.hostname:
            continue
        # Ads redirect through duckduckgo.com itself.
        if parts.hostname.lower().endswith("duckduckgo.com"):
            continue
        found.append(
            {
                "title": " ".join(item["title"].split()),
                "url": item["url"],
                "snippet": " ".join(item["snippet"].split()),
            }
        )
        if len(found) == MAX_RESULTS:
            break
    return found


def _ddg_post(query):
    """The DuckDuckGo HTML page for a query; the network seam for tests."""
    data = urllib.parse.urlencode({"q": query}).encode()
    request = urllib.request.Request(
        SEARCH_URL,
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": ACCEPT_LANGUAGE,
            "Accept": "text/html",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=SEARCH_TIMEOUT) as response:
            status = response.status
            page = response.read(MAX_BYTES).decode("utf-8", "replace")
    except (OSError, http.client.HTTPException) as error:
        raise APIError(
            502, f"search request failed: {error}", "search_failed"
        ) from error
    return status, page


def search(query):
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY:
        raise APIError(
            400, "query must be a non-empty string of at most 500 characters"
        )
    status, page = _ddg_post(query.strip())
    results = parse_results(page)
    if not results and (
        status != 200 or "anomaly" in page or "captcha" in page.lower()
    ):
        raise APIError(
            502,
            "DuckDuckGo blocked or refused the search (anti-bot); retry later",
            "search_blocked",
        )
    return {"results": results}


# Fetching one page with an SSRF guard.


def check_address(address):
    """Reject anything that is not a globally routable unicast address."""
    ip = ipaddress.ip_address(address.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if not ip.is_global or ip.is_multicast:
        raise APIError(403, f"address {ip} is not allowed", "forbidden_address")


def resolve(host, port):
    """Public addresses of a host; any non-public answer rejects the host."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise APIError(502, f"cannot resolve {host}", "fetch_failed") from error
    addresses = [info[4][0] for info in infos]
    if not addresses:
        raise APIError(502, f"cannot resolve {host}", "fetch_failed")
    for address in addresses:
        check_address(address)
    return addresses


def _connect(host, port, timeout):
    """Connect to a validated address, so a rebinding DNS answer cannot slip in."""
    error = OSError("no address")
    for address in resolve(host, port):
        try:
            return socket.create_connection((address, port), timeout)
        except OSError as failure:
            error = failure
    raise error


class _Http(http.client.HTTPConnection):
    def connect(self):
        self.sock = _connect(self.host, self.port, self.timeout)


class _Https(http.client.HTTPSConnection):
    def connect(self):
        sock = _connect(self.host, self.port, self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _request(url, timeout):
    """One GET without redirects: (status, headers, body up to MAX_BYTES). Network seam."""
    parts = urlsplit(url)
    connection = (_Https if parts.scheme == "https" else _Http)(
        parts.hostname, parts.port, timeout=timeout
    )
    try:
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        connection.request(
            "GET",
            path,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": ACCEPT_LANGUAGE,
                "Accept": "text/html,text/plain;q=0.9",
                "Accept-Encoding": "identity",
            },
        )
        response = connection.getresponse()
        body = response.read(MAX_BYTES)
        return response.status, response.headers, body
    finally:
        connection.close()


def check_url(url):
    if not isinstance(url, str) or not url.strip() or len(url) > MAX_URL:
        raise APIError(400, "url must be a non-empty string of at most 2048 characters")
    parts = urlsplit(url.strip())
    try:
        parts.port
    except ValueError:
        raise APIError(400, "invalid url port") from None
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
    ):
        raise APIError(400, "only plain http and https URLs are allowed")
    resolve(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    return parts.geturl()


_DROP = {
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "noscript",
    "svg",
    "template",
    "iframe",
}
_BLOCK = {
    "p",
    "div",
    "br",
    "li",
    "ul",
    "ol",
    "tr",
    "table",
    "pre",
    "section",
    "article",
    "main",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "dd",
    "dt",
    "figcaption",
    "hr",
}
_CELL = {"td", "th"}


class _TextParser(HTMLParser):
    def __init__(self, drop):
        super().__init__(convert_charrefs=True)
        self.drop = drop
        self.skip = 0
        self.in_title = False
        self.pre = 0
        self.title = ""
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.drop:
            self.skip += 1
        elif tag == "title":
            self.in_title = True
        elif tag in _BLOCK:
            self.pre += tag == "pre"
            self.parts.append("\n")
        elif tag in _CELL:
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.drop:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self.in_title = False
        elif tag in _BLOCK:
            self.pre = max(0, self.pre - (tag == "pre"))
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip:
            # Source line breaks inside a paragraph are plain whitespace.
            self.parts.append(data if self.pre else re.sub(r"\s+", " ", data))


def extract_text(page):
    """(title, readable text) of an HTML page with whitespace collapsed."""
    for drop in (_DROP, _DROP - {"form", "header", "aside", "nav"}):
        parser = _TextParser(drop)
        parser.feed(page)
        parser.close()
        lines = (" ".join(line.split()) for line in "".join(parser.parts).split("\n"))
        text = "\n".join(line.strip(" |") for line in lines if line.strip(" |"))
        # An unclosed <form> or <header> can swallow the page; retry keeping them.
        if len(text) >= 200:
            break
    return " ".join(parser.title.split()), text


def _chunks(text):
    chunks, current = [], ""
    for line in text.split("\n"):
        while len(line) > CHUNK:
            cut = line.rfind(" ", 0, CHUNK)
            cut = cut if cut > 0 else CHUNK
            chunks.append(line[:cut])
            line = line[cut:].lstrip()
        if current and len(current) + len(line) + 1 > CHUNK:
            chunks.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks


def _stems(query):
    words = re.findall(r"\w{3,}", (query or "").lower())
    # ponytail: crude prefix stemming for Russian endings; upgrade to a real stemmer if recall matters.
    return {word[:5] if len(word) > 5 else word for word in words}


def select_text(text, query, budget=TEXT_BUDGET):
    """(text within budget, truncated): the best-matching windows for the query, else the start."""
    if len(text) <= budget:
        return text, False
    chunks = _chunks(text)
    stems = _stems(query)
    scores = [
        sum(min(chunk.lower().count(stem), 3) for stem in stems) for chunk in chunks
    ]
    ranked = [
        index
        for index in sorted(range(len(chunks)), key=lambda i: -scores[i])
        if scores[index] > 0
    ]
    picked, used = [], 0
    for index in ranked or range(len(chunks)):
        if used + len(chunks[index]) > budget:
            if ranked:
                continue
            break
        picked.append(index)
        used += len(chunks[index]) + 3
    return "\n…\n".join(chunks[i] for i in sorted(picked)), True


def _charset(headers, body):
    match = re.search(r"charset=([\w-]+)", headers.get("Content-Type", ""), re.I)
    if not match:
        match = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", body[:2048], re.I)
        name = match.group(1).decode("ascii") if match else "utf-8"
    else:
        name = match.group(1)
    try:
        "".encode(name)
    except LookupError:
        return "utf-8"
    return name


def fetch(url, query=None):
    deadline = time.monotonic() + FETCH_TIMEOUT
    current = check_url(url)
    for _ in range(MAX_REDIRECTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise APIError(504, "page fetch timed out", "fetch_timeout")
        try:
            status, headers, body = _request(current, remaining)
        except APIError:
            raise
        except (OSError, http.client.HTTPException) as error:
            raise APIError(
                502, f"page request failed: {error}", "fetch_failed"
            ) from error
        if status in (301, 302, 303, 307, 308) and headers.get("Location"):
            current = check_url(urljoin(current, headers["Location"]))
            continue
        break
    else:
        raise APIError(502, "too many redirects", "fetch_failed")
    if status >= 400:
        raise APIError(502, f"page answered HTTP {status}", "fetch_failed")
    kind = headers.get("Content-Type", "").split(";")[0].strip().lower()
    if kind not in TEXT_TYPES:
        raise APIError(
            415, f"unsupported content type {kind or 'unknown'}", "unsupported_type"
        )
    page = body.decode(_charset(headers, body), "replace")
    if kind == "text/plain":
        title, text = (
            "",
            "\n".join(
                " ".join(line.split()) for line in page.split("\n") if line.strip()
            ),
        )
    else:
        title, text = extract_text(page)
    text, truncated = select_text(text, query)
    return {"url": current, "title": title, "text": text, "truncated": truncated}
