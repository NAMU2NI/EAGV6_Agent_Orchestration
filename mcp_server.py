"""
MCP server for EAGV3 Session 6.

Nine tools, stdio transport:
    web_search, fetch_url, get_time, currency_convert,
    read_file, list_dir, create_file, update_file, edit_file

web_search:  Tavily primary, DuckDuckGo fallback. Hard-capped at 5 results.
fetch_url:   crawl4ai only — clean markdown via headless Chromium.
Usage for tavily and duckduckgo is logged to ./usage.json with monthly
rollover and a soft cap of 950/1000 on Tavily.

File tools are sandboxed under ./sandbox/. Run:  python mcp_server.py
"""

from __future__ import annotations

import json
import os
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from mcp.server.fastmcp import FastMCP

from env_loader import load_env

ROOT = Path(__file__).parent
MAX_SEARCH_RESULTS = 5  # hard cap — Tavily prices per result

load_env(ROOT.parent / ".env")
load_env(ROOT / ".env")

mcp = FastMCP("eagv3-s6-server")

SANDBOX = Path(__file__).parent / "sandbox"
SANDBOX.mkdir(exist_ok=True)

USAGE_PATH = Path(__file__).parent / "usage.json"
MONTHLY_CAP = 950  # leave 50/mo headroom on Tavily
_usage_lock = threading.Lock()


def _safe(path: str) -> Path:
    p = (SANDBOX / path).resolve()
    base = SANDBOX.resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"Path '{path}' escapes the sandbox")
    return p


def _empty_usage(month: str) -> dict:
    return {
        "month": month,
        "tavily": {"count": 0, "errors": 0},
        "duckduckgo": {"count": 0, "errors": 0},
    }


def _load_usage() -> dict:
    month = datetime.now().strftime("%Y-%m")
    if not USAGE_PATH.exists():
        return _empty_usage(month)
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_usage(month)
    if data.get("month") != month:
        return _empty_usage(month)
    for k in ("tavily", "duckduckgo"):
        data.setdefault(k, {"count": 0, "errors": 0})
    return data


def _save_usage(data: dict) -> None:
    USAGE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _bump(provider: str, field: str = "count") -> None:
    with _usage_lock:
        data = _load_usage()
        data[provider][field] = data[provider].get(field, 0) + 1
        _save_usage(data)


def _under_cap(provider: str) -> bool:
    return _load_usage()[provider]["count"] < MONTHLY_CAP


def _tavily_search(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(os.environ["TAVILY_API_KEY"])
    resp = client.search(query=query, max_results=max_results, search_depth="advanced")
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in resp.get("results", [])
    ]


def _ddg_search(query: str, max_results: int) -> list[dict]:
    hits: list[dict] = []
    with DDGS() as ddgs:
        for backend in ("auto", "html", "lite"):
            try:
                hits = list(ddgs.text(query, max_results=max_results, backend=backend))
            except Exception:
                hits = []
            if hits:
                break
    return [
        {
            "title": h.get("title", ""),
            "url": h.get("href", ""),
            "snippet": h.get("body", ""),
        }
        for h in hits
    ]


async def _crawl4ai_fetch(url: str, timeout: int = 20) -> dict:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

    # crawl4ai uses Rich which writes via its own captured stdout reference, so
    # contextlib.redirect_stdout doesn't catch it. Redirect at the file-descriptor
    # level — crawl4ai's banner / [FETCH] / [SCRAPE] markers would otherwise
    # corrupt the MCP stdio JSON-RPC stream.
    saved_fd = os.dup(1)
    os.dup2(2, 1)
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            config = CrawlerRunConfig(
                page_timeout=max(1, timeout) * 1000,
                wait_until="domcontentloaded",
                max_retries=0,
            )
            r = await crawler.arun(url=url, config=config)
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
    # r.markdown is a str subclass (StringCompatibleMarkdown) that Pydantic
    # serializes as {} because its real field is private. Pull the raw string
    # out and force a plain str so FastMCP serializes correctly.
    md = r.markdown
    raw = (
        getattr(md, "raw_markdown", None)
        or getattr(md, "fit_markdown", None)
        or md
        or r.cleaned_html
        or r.html
        or ""
    )
    text = str(raw)
    return {
        "status": int(getattr(r, "status_code", None) or 200),
        "content_type": "text/markdown",
        "length_bytes": len(text.encode("utf-8")),
        "text": text,
    }


def _wikipedia_api_url(url: str) -> str | None:
    """Convert a Wikipedia article URL to the Action API (plain-text extract).
    en.wikipedia.org/wiki/X → en.wikipedia.org/w/api.php?action=query&...
    The Action API is designed for bots and does not block programmatic access."""
    import re
    m = re.match(r"https?://([\w.]+)/wiki/(.+)", url)
    if m and "wikipedia.org" in m.group(1):
        title = m.group(2)
        return (
            f"https://{m.group(1)}/w/api.php"
            f"?action=query&titles={title}&prop=extracts"
            f"&explaintext=1&format=json&redirects=1"
        )
    return None


def _ddg_instant_answer(title: str) -> dict | None:
    """Retrieve a Wikipedia article via DuckDuckGo when direct Wikipedia access is blocked.

    Combines the DDG Instant Answer API (abstract + infobox) with up to 5 DDG
    text-search results for the same topic. The combined document is typically
    > 4 KB, which causes action.py to store it as a named artifact — preserving
    the same agent behaviour as a successful full-page fetch would have.
    """
    import re
    import urllib.parse
    clean_title = re.sub(r"#.*$", "", title).replace("_", " ").strip()
    query = urllib.parse.quote_plus(clean_title)
    ia_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=10, follow_redirects=True, headers=headers) as client:
            r = client.get(ia_url)
            r.raise_for_status()
        data = r.json()
        abstract = data.get("AbstractText", "")
        if not abstract:
            return None
        source = data.get("AbstractSource") or "Wikipedia"
        parts = [f"# {clean_title}", f"*Source: {source} (via DuckDuckGo — direct Wikipedia access blocked)*", "", abstract]

        # Infobox structured data — skip raw Wikidata objects (non-string values)
        infobox = data.get("Infobox") or {}
        items = infobox.get("content") or []
        fact_lines = []
        for item in items:
            label = item.get("label", "")
            value = item.get("value", "")
            if label and isinstance(value, str) and value.strip():
                fact_lines.append(f"- **{label}**: {value}")
        if fact_lines:
            parts += ["", "## Key Facts"] + fact_lines

        # Related topics from DDG (skip if empty)
        related = [t for t in (data.get("RelatedTopics") or []) if t.get("Text")]
        if related:
            parts += ["", "## Related Topics"]
            for topic in related[:8]:
                text_val = topic.get("Text", "") or ""
                first_url = topic.get("FirstURL", "") or ""
                parts.append(f"- {text_val}" + (f" ({first_url})" if first_url else ""))

        # Full web-search results to ensure the document crosses the artifact
        # threshold (4096 bytes). A real page fetch would return tens of KB;
        # these snippets provide equivalent navigable content for the agent.
        search_results = _ddg_search(clean_title, 5)
        _bump("duckduckgo")
        if search_results:
            parts += ["", "## Web Search Results"]
            for idx, res in enumerate(search_results, 1):
                parts += [
                    f"### {idx}. {res['title']}",
                    f"URL: {res['url']}",
                    res["snippet"],
                    "",
                ]

        text = "\n".join(parts)
        return {
            "status": 200,
            "content_type": "text/markdown",
            "length_bytes": len(text.encode("utf-8")),
            "text": text,
        }
    except Exception:
        return None


def _httpx_fetch(url: str, timeout: int = 20) -> dict:
    import re
    is_wiki = bool(re.match(r"https?://[\w.]*wikipedia\.org/", url))
    fetch_target = _wikipedia_api_url(url) or url
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "application/json, text/html, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        r = client.get(fetch_target)

    # Wikimedia IPs-blocks certain networks with a hard 403. When that happens
    # fall back to DuckDuckGo Instant Answer, which returns the same Wikipedia
    # abstract + infobox without requiring direct *.wikipedia.org connectivity.
    if r.status_code == 403 and is_wiki:
        m = re.match(r"https?://[\w.]+/wiki/(.+)", url)
        if m:
            result = _ddg_instant_answer(m.group(1))
            if result:
                return result

    r.raise_for_status()
    content_type = r.headers.get("content-type", "application/octet-stream").split(";")[0]
    text = r.text
    if "json" in content_type:
        import json as _json
        try:
            data = _json.loads(text)
            pages = (data.get("query") or {}).get("pages") or {}
            page = next(iter(pages.values()), {})
            title = page.get("title", url)
            extract = page.get("extract", text)
            text = f"# {title}\n\n{extract}"
        except Exception:
            pass
        content_type = "text/markdown"
    elif "html" in content_type:
        soup = BeautifulSoup(text, "lxml")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else url
        body = soup.get_text("\n", strip=True)
        text = f"# {title}\n\n{body}"
        content_type = "text/markdown"
    return {
        "status": r.status_code,
        "content_type": content_type,
        "length_bytes": len(text.encode("utf-8")),
        "text": text,
    }


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web (Tavily primary, DDG fallback). Hard-capped at 5 results. Example: web_search("python asyncio tutorial", 3)."""
    max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
    if os.environ.get("TAVILY_API_KEY") and _under_cap("tavily"):
        try:
            results = _tavily_search(query, max_results)
            if results:
                _bump("tavily")
                return results
        except Exception:
            _bump("tavily", "errors")
    results = _ddg_search(query, max_results)
    _bump("duckduckgo")
    return results


@mcp.tool()
async def fetch_url(url: str, timeout: int = 20) -> dict:
    """Fetch clean markdown from a URL. Example: fetch_url("https://example.com")."""
    backend = os.getenv("FETCH_URL_BACKEND", "httpx").lower()
    if backend == "crawl4ai":
        try:
            return await asyncio.wait_for(_crawl4ai_fetch(url, timeout=timeout), timeout=timeout + 5)
        except asyncio.TimeoutError:
            text = f"ERROR: fetch_url timed out after {timeout}s for {url}"
            return {
                "status": 408,
                "content_type": "text/plain",
                "length_bytes": len(text.encode("utf-8")),
                "text": text,
            }
    return _httpx_fetch(url, timeout=timeout)


@mcp.tool()
def get_time(timezone: str = "UTC") -> dict:
    """Current time in a named IANA timezone. Example: get_time("Asia/Kolkata")."""
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    offset = now.utcoffset()
    offset_hours = offset.total_seconds() / 3600 if offset else 0.0
    return {
        "iso": now.isoformat(),
        "human": now.strftime("%A, %d %B %Y %H:%M:%S %Z"),
        "timezone": timezone,
        "offset_hours": offset_hours,
    }


@mcp.tool()
def currency_convert(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert money between ISO-3 currencies via frankfurter.dev. Example: currency_convert(100, "USD", "INR")."""
    f = from_currency.upper()
    t = to_currency.upper()
    url = f"https://api.frankfurter.dev/v1/latest?amount={amount}&base={f}&symbols={t}"
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    converted = data["rates"][t]
    return {
        "amount": amount,
        "from": f,
        "to": t,
        "rate": converted / amount if amount else 0.0,
        "converted": converted,
        "date": data["date"],
        "source": "frankfurter.dev",
    }


@mcp.tool()
def read_file(path: str) -> dict:
    """Read a UTF-8 text file from the sandbox. Example: read_file("notes.txt")."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    return {
        "path": path,
        "size_bytes": p.stat().st_size,
        "content": text,
        "encoding": "utf-8",
    }


@mcp.tool()
def list_dir(path: str = ".") -> list[dict]:
    """List a directory inside the sandbox. Example: list_dir(".")."""
    p = _safe(path)
    out = []
    for child in sorted(p.iterdir()):
        is_dir = child.is_dir()
        out.append({
            "name": child.name,
            "type": "dir" if is_dir else "file",
            "size_bytes": 0 if is_dir else child.stat().st_size,
        })
    return out


@mcp.tool()
def create_file(path: str, content: str) -> dict:
    """Create a new file in the sandbox; errors if it exists. Example: create_file("hello.txt", "hi")."""
    p = _safe(path)
    if p.exists():
        raise ValueError(f"File '{path}' already exists")
    if not p.parent.exists():
        raise ValueError(f"Parent directory of '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def update_file(path: str, content: str) -> dict:
    """Overwrite an existing sandbox file. Example: update_file("hello.txt", "new body")."""
    p = _safe(path)
    if not p.exists():
        raise ValueError(f"File '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def edit_file(path: str, find: str, replace: str, replace_all: bool = False) -> dict:
    """Find-and-replace inside a sandbox file. Example: edit_file("hello.txt", "foo", "bar")."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(find)
    if count == 0:
        raise ValueError(f"'{find}' not found in '{path}'")
    if count > 1 and not replace_all:
        raise ValueError(
            f"'{find}' occurs {count} times in '{path}'; pass replace_all=True"
        )
    new_text = text.replace(find, replace) if replace_all else text.replace(find, replace, 1)
    p.write_text(new_text, encoding="utf-8")
    replacements = count if replace_all else 1
    return {
        "ok": True,
        "path": path,
        "replacements": replacements,
        "size_bytes": p.stat().st_size,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
