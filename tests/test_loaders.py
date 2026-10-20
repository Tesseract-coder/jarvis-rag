"""Loaders: every input type, offline (demo_data writes real PDF/DOCX/CSV files)."""

import sqlite3

import httpx
import pytest

from jarvis_rag import demo_data
from jarvis_rag.loaders import (SkipFile, WebLoader, clean_web, find_symbols, iter_files, load_code,
                                load_csv, load_docx, load_file, load_pdf, table_name)

MB = 10**6


def test_pdf_one_document_per_page_with_page_numbers(tmp_path):
    path = tmp_path / "report.pdf"
    demo_data.write_pdf(path, demo_data.SAFETY_REPORT, title="Safety Report")
    docs = load_pdf(path)
    assert [d.metadata["page"] for d in docs] == [1, 2, 3]
    assert docs[0].metadata["title"] == "Safety Report" and docs[0].metadata["type"] == "pdf"
    assert "obstructed lidar sensor" in docs[1].page_content


def test_docx_keeps_heading_styles_as_markdown(tmp_path):
    path = tmp_path / "manual.docx"
    demo_data.write_docx(path, [("h", "Manual"), ("p", "Intro."), ("h", "Weekly checks"), ("p", "Wipe the lidar.")])
    doc = load_docx(path)[0]
    assert doc.page_content == "# Manual\n\nIntro.\n\n# Weekly checks\n\nWipe the lidar."
    assert doc.metadata["title"] == "Manual"


def test_dispatch_and_unsupported(tmp_path):
    (tmp_path / "notes.md").write_text("# Notes\nhello")
    (tmp_path / "pic.png").write_bytes(b"\x89PNG")
    assert load_file(tmp_path / "notes.md", MB)[0].metadata["type"] == "text"
    with pytest.raises(SkipFile, match="unsupported"):
        load_file(tmp_path / "pic.png", MB)


@pytest.mark.parametrize("name,content,reason", [
    ("blob.py", b"abc\x00def", "binary"),
    ("big.py", b"x = 1\n" * 1000, "too big"),
    ("empty.py", b"  \n", "empty"),
    ("package-lock.json", b"{}", "lockfile"),
    ("latin1.py", "café = 1".encode("latin-1"), "UTF-8"),
])
def test_code_files_we_refuse(tmp_path, name, content, reason):
    (tmp_path / name).write_bytes(content)
    with pytest.raises(SkipFile, match=reason):
        load_code(tmp_path / name, max_bytes=1000)


def test_walk_respects_gitignore_and_skip_dirs(tmp_path):
    files = {
        ".gitignore": "*.log\nbuild_out/\n",
        "app.py": "x=1", "debug.log": "noise", "build_out/gen.py": "x=1",
        "node_modules/lib/index.js": "x", ".venv/lib.py": "x", ".env": "SECRET=1",
        "pkg/.gitignore": "secret.py\n", "pkg/mod.py": "y=1", "pkg/secret.py": "z=1",
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    found = sorted(p.relative_to(tmp_path).as_posix() for p in iter_files(tmp_path))
    assert found == ["app.py", "pkg/mod.py"]


PY = '''import os


class Greeter:
    """Says hello."""

    def hello(self):
        return "hi"


async def main():
    pass
'''


def test_find_symbols_python_qualifies_methods():
    syms = find_symbols(PY, "python")
    assert [name for _, name in syms] == ["Greeter", "Greeter.hello", "main"]
    pos = dict((n, p) for p, n in syms)
    assert PY[pos["Greeter"]:].startswith("class Greeter")  # not the blank line above


def test_find_symbols_js_and_unknown():
    js = "export function load() {}\nconst save = async (x) => x\nclass Store {\n  get() {}\n}\n"
    assert [n for _, n in find_symbols(js, "js")] == ["load", "save", "Store"]
    assert find_symbols("whatever", "yaml") == []


def test_csv_goes_to_sqlite_and_schema_doc_is_embedded(tmp_path):
    csv = tmp_path / "Robot Sales 2026.csv"
    demo_data.write_sales_csv(csv)
    db = tmp_path / "x.db"
    doc = load_csv(csv, db)[0]
    assert doc.metadata["table"] == table_name(csv) == "robot_sales_2026"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM robot_sales_2026").fetchone()[0] == 24
        cols = [r[1] for r in conn.execute("PRAGMA table_info(robot_sales_2026)")]
    assert cols == ["quarter", "region", "product", "units", "revenue_eur"]  # names normalised
    assert "one of EMEA, Americas, APAC" in doc.page_content
    assert "Sample rows:" in doc.page_content
    load_csv(csv, db)  # re-load replaces, never appends
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM robot_sales_2026").fetchone()[0] == 24


def test_table_name_is_a_safe_identifier(tmp_path):
    assert table_name(tmp_path / "2026 Q1-results.csv") == "t_2026_q1_results"


# --------------------------------------------------------------------------- web

PAGES = {
    "/robots.txt": ("text/plain", "User-agent: *\nDisallow: /private\n"),
    "/": ("text/html", "<html><head><title>Home</title></head><body><nav>menu menu</nav><article>"
                       "<h1>Welcome</h1><p>" + "Halcyon builds warehouse robots for narrow aisles. " * 8 + "</p>"
                       "<a href='/a'>A</a> <a href='/private/x'>P</a> <a href='https://other.org/'>O</a>"
                       "<a href='/a#frag'>A again</a></article></body></html>"),
    "/a": ("text/html", "<html><body><article><h1>Page A</h1><p>" + "Kestrel K2 carries 45 kg. " * 10 +
                        "</p><a href='/b'>B</a></article></body></html>"),
    "/b": ("text/html", "<html><body><article><p>" + "Deep page. " * 30 + "</p></article></body></html>"),
}


def fake_site() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "site.test" or request.url.path not in PAGES:
            return httpx.Response(404)
        ctype, body = PAGES[request.url.path]
        return httpx.Response(200, headers={"content-type": ctype}, text=body)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_crawl_depth_robots_and_same_domain():
    loader = WebLoader("test-agent", client=fake_site())
    assert [d.metadata["url"] for d in loader.crawl("https://site.test/", depth=0)] == ["https://site.test/"]
    docs = WebLoader("test-agent", client=fake_site()).crawl("https://site.test/", depth=1)
    assert [d.metadata["url"] for d in docs] == ["https://site.test/", "https://site.test/a"]  # /b is depth 2
    loader = WebLoader("test-agent", client=fake_site())
    loader.crawl("https://site.test/", depth=1)
    assert ("https://site.test/private/x", "disallowed by robots.txt") in loader.skipped


def test_page_limit():
    docs = WebLoader("t", max_pages=2, client=fake_site()).crawl("https://site.test/", depth=5)
    assert len(docs) == 2


def test_web_text_is_main_content_only():
    doc = WebLoader("t", client=fake_site()).crawl("https://site.test/a")[0]
    assert "Kestrel K2 carries 45 kg." in doc.page_content
    assert doc.metadata["type"] == "web" and doc.metadata["source"] == "https://site.test/a"


def test_clean_web_drops_footnotes_and_edit_links():
    assert clean_web("Intro<sup>\\[5\\]</sup> text\n\\[edit\\]\n<b>bold</b>") == "Intro text\n\nbold"
