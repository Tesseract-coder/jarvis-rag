"""Turn files, folders, URLs and CSVs into LangChain `Document`s.

Every loader returns whole documents (one per PDF page, one per file otherwise)
with at least `source`, `type` and `title` in metadata. Chunking happens later,
in ingest.py, so the loaders stay simple and the chunker sees full structure.

    type    produced by        chunked as
    pdf     load_pdf           plain text, one Document per page (keeps `page`)
    docx    load_docx          markdown (Word heading styles kept)
    text    load_text          markdown (headings become section breadcrumbs)
    code    load_code          per-language splitter (keeps `path`, `language`)
    web     load_url / crawl   markdown from trafilatura
    table   load_csv           not chunked: schema + sample rows, data goes to SQLite
"""

import io
import os
import re
import sqlite3
import zipfile
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import pandas as pd
import pathspec
import trafilatura
from langchain_core.documents import Document
from langchain_text_splitters import Language
from lxml import etree
from lxml import html as lxml_html
from pypdf import PdfReader

TEXT_EXTS = {".md", ".markdown", ".txt", ".rst"}

CODE_LANGS: dict[str, Language | None] = {
    ".py": Language.PYTHON, ".js": Language.JS, ".jsx": Language.JS, ".mjs": Language.JS,
    ".ts": Language.TS, ".tsx": Language.TS, ".go": Language.GO, ".java": Language.JAVA,
    ".kt": Language.KOTLIN, ".rs": Language.RUST, ".rb": Language.RUBY, ".php": Language.PHP,
    ".c": Language.C, ".h": Language.C, ".cpp": Language.CPP, ".hpp": Language.CPP, ".cc": Language.CPP,
    ".cs": Language.CSHARP, ".scala": Language.SCALA, ".swift": Language.SWIFT, ".lua": Language.LUA,
    ".html": Language.HTML, ".sol": Language.SOL,
    # No language-aware separators; split on blank lines/lines like any text.
    ".sh": None, ".sql": None, ".css": None, ".toml": None, ".yaml": None, ".yml": None,
    ".json": None, ".ini": None, ".cfg": None,
}
CODE_FILENAMES = {"Makefile": None, "Dockerfile": None}

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", ".tox", "dist", "build", ".next", "target", ".idea", ".vscode"}
LOCKFILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock", "Cargo.lock",
             "Gemfile.lock", "composer.lock", "Pipfile.lock", "go.sum"}

# Definitions, so a code chunk can say which function/class it holds. Groups come
# in (indent, name) pairs. [ \t]* rather than \s*: \s also matches newlines, which
# would start the match on the blank line *above* the definition.
SYMBOL_PATTERNS = {
    "python": r"^([ \t]*)(?:async[ \t]+)?(?:def|class)[ \t]+(\w+)",
    "js": r"^([ \t]*)(?:export[ \t]+)?(?:default[ \t]+)?(?:async[ \t]+)?(?:function\*?|class)[ \t]+(\w+)"
          r"|^([ \t]*)(?:export[ \t]+)?(?:const|let)[ \t]+(\w+)[ \t]*=[ \t]*(?:async[ \t]*)?\(",
    "go": r"^()func[ \t]+(?:\([^)]*\)[ \t]*)?(\w+)|^()type[ \t]+(\w+)",
    "java": r"^([ \t]*)(?:(?:public|private|protected|static|final|abstract)[ \t]+)*"
            r"(?:class|interface|enum|record)[ \t]+(\w+)",
    "rust": r"^([ \t]*)(?:pub[ \t]+)?(?:async[ \t]+)?(?:fn|struct|enum|trait|impl)[ \t]+(\w+)",
    "ruby": r"^([ \t]*)(?:def|class|module)[ \t]+([\w.?!]+)",
}
SYMBOL_PATTERNS["ts"] = SYMBOL_PATTERNS["js"]
SYMBOL_PATTERNS["kotlin"] = SYMBOL_PATTERNS["scala"] = r"^([ \t]*)(?:\w+[ \t]+)*(?:fun|class|object|interface|def)[ \t]+(\w+)"
SYMBOL_PATTERNS["csharp"] = SYMBOL_PATTERNS["java"]


class SkipFile(Exception):
    """Raised for files we deliberately don't ingest (binary, too big, empty)."""


def source_name(path: Path) -> str:
    """Stable `source` id for a file: relative to cwd when possible, always posix."""
    path = path.resolve()
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_text(path: Path, max_bytes: int) -> str:
    size = path.stat().st_size
    if size > max_bytes:
        raise SkipFile(f"too big ({size / 1e6:.1f} MB)")
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        raise SkipFile("binary")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise SkipFile("not UTF-8 text") from None
    if not text.strip():
        raise SkipFile("empty")
    return text


# --------------------------------------------------------------------------- documents

def load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(path)
    title = (reader.metadata.title if reader.metadata else None) or path.stem
    src = source_name(path)
    docs = []
    for n, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:  # scanned pages have no text layer; OCR is out of scope
            docs.append(Document(text, metadata={"source": src, "type": "pdf", "title": title,
                                                 "path": src, "page": n}))
    if not docs:
        raise SkipFile("no extractable text (scanned PDF?)")
    return docs


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_to_markdown(path: Path) -> str:
    """Paragraphs and tables from word/document.xml, with Heading N styles turned
    into markdown '#'s. docx2txt (what Docx2txtLoader wraps) drops the styles,
    and headings are exactly what our chunk headers are built from."""
    with zipfile.ZipFile(path) as z:
        root = etree.fromstring(z.read("word/document.xml"))
    lines = []
    for el in root.iter(f"{W}p", f"{W}tbl"):
        if el.tag == f"{W}tbl":
            for row in el.iter(f"{W}tr"):
                cells = ["".join(t.text or "" for t in tc.iter(f"{W}t")).strip() for tc in row.iter(f"{W}tc")]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
            continue
        if any(a.tag == f"{W}tbl" for a in el.iterancestors()):
            continue  # table paragraphs were emitted as rows above
        text = "".join(t.text or "" for t in el.iter(f"{W}t")).strip()
        if not text:
            continue
        style = el.find(f"{W}pPr/{W}pStyle")
        level = re.fullmatch(r"(?i)heading\s?(\d)|title()", style.get(f"{W}val", "")) if style is not None else None
        if level:
            lines += ["", "#" * int(level.group(1) or 1) + " " + text, ""]
        else:
            lines += [text, ""]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def load_docx(path: Path) -> list[Document]:
    text = docx_to_markdown(path)
    if not text:
        raise SkipFile("empty")
    src = source_name(path)
    h1 = re.search(r"^#\s+(.+)$", text, re.M)
    return [Document(text, metadata={"source": src, "type": "docx", "title": h1.group(1) if h1 else path.stem,
                                     "path": src})]


def load_text(path: Path, max_bytes: int) -> list[Document]:
    text = _read_text(path, max_bytes)
    src = source_name(path)
    h1 = re.search(r"^#\s+(.+)$", text, re.M)
    return [Document(text, metadata={"source": src, "type": "text", "title": h1.group(1).strip() if h1 else path.stem,
                                     "path": src})]


def code_language(path: Path) -> tuple[bool, Language | None]:
    """(is_code, LangChain language or None for generic code)."""
    if path.name in CODE_FILENAMES:
        return True, CODE_FILENAMES[path.name]
    if path.suffix.lower() in CODE_LANGS:
        return True, CODE_LANGS[path.suffix.lower()]
    return False, None


def load_code(path: Path, max_bytes: int) -> list[Document]:
    if path.name in LOCKFILES or path.name.endswith((".min.js", ".min.css")):
        raise SkipFile("lockfile/minified")
    text = _read_text(path, max_bytes)
    _, lang = code_language(path)
    src = source_name(path)
    language = lang.value if lang else (path.suffix.lstrip(".").lower() or path.name.lower())
    return [Document(text, metadata={"source": src, "type": "code", "title": path.name, "path": src,
                                     "language": language})]


def find_symbols(text: str, language: str) -> list[tuple[int, str]]:
    """(char offset, name) of every definition the language's regex recognises.
    Indented definitions are qualified by the last unindented one: `Class.method`."""
    pattern = SYMBOL_PATTERNS.get(language)
    if not pattern:
        return []
    out, parent = [], None
    for m in re.finditer(pattern, text, re.M):
        groups = m.groups()
        i = next(i for i in range(1, len(groups), 2) if groups[i])  # the alternative that matched
        indent, name = groups[i - 1], groups[i]
        if indent:
            name = f"{parent}.{name}" if parent else name
        else:
            parent = name
        out.append((m.start(), name))
    return out


# --------------------------------------------------------------------------- folders

def iter_files(root: Path) -> Iterator[Path]:
    """Walk `root`, honouring every .gitignore on the way down plus SKIP_DIRS."""
    root = root.resolve()
    specs: list[tuple[Path, pathspec.GitIgnoreSpec]] = []

    def ignored(p: Path, is_dir: bool) -> bool:
        for base, spec in specs:
            if p.is_relative_to(base):
                rel = p.relative_to(base).as_posix() + ("/" if is_dir else "")
                if spec.match_file(rel):
                    return True
        return False

    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        gi = d / ".gitignore"
        if gi.is_file():
            specs.append((d, pathspec.GitIgnoreSpec.from_lines(gi.read_text(errors="ignore").splitlines())))
        dirnames[:] = sorted(n for n in dirnames if n not in SKIP_DIRS and not n.startswith(".")
                             and not ignored(d / n, is_dir=True))
        for name in sorted(filenames):
            p = d / name
            if not name.startswith(".") and not ignored(p, is_dir=False):
                yield p


def load_file(path: Path, max_bytes: int) -> list[Document]:
    """Dispatch on extension. CSVs are handled separately (they go to SQLite)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext in TEXT_EXTS:
        return load_text(path, max_bytes)
    if code_language(path)[0]:
        return load_code(path, max_bytes)
    raise SkipFile(f"unsupported type {ext or path.name!r}")


# --------------------------------------------------------------------------- CSV -> SQLite

def table_name(path: Path) -> str:
    name = re.sub(r"\W+", "_", path.stem.lower()).strip("_") or "table"
    return f"t_{name}" if name[0].isdigit() else name


def describe_table(conn: sqlite3.Connection, table: str, sample_rows: int = 3, max_distinct: int = 12) -> str:
    """Schema + sample rows as text. This is what gets embedded, so the agent can
    *discover* that a table exists; the numbers themselves are queried with SQL."""
    cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    n_rows = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    lines = [f"SQL table `{table}` ({n_rows} rows). Query it with SQL; columns:"]
    for _, name, typ, *_ in cols:
        line = f"- {name} ({typ or 'TEXT'})"
        if typ in ("TEXT", ""):
            distinct = [r[0] for r in conn.execute(
                f'SELECT DISTINCT "{name}" FROM "{table}" LIMIT {max_distinct + 1}')]
            if len(distinct) <= max_distinct:
                line += f": one of {', '.join(map(str, distinct))}"
        else:
            lo, hi = conn.execute(f'SELECT MIN("{name}"), MAX("{name}") FROM "{table}"').fetchone()
            line += f": {lo} to {hi}"
        lines.append(line)
    sample = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT {sample_rows}', conn)
    lines += ["", f"Sample rows:", sample.to_string(index=False)]
    return "\n".join(lines)


def load_csv(path: Path, sqlite_path: Path) -> list[Document]:
    df = pd.read_csv(path)
    if df.empty:
        raise SkipFile("empty CSV")
    df.columns = [re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") or f"col{i}"
                  for i, c in enumerate(df.columns)]
    table = table_name(path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)
        text = describe_table(conn, table)
    src = source_name(path)
    return [Document(text, metadata={"source": src, "type": "table", "title": table, "path": src,
                                     "table": table})]


# --------------------------------------------------------------------------- URLs

def clean_web(md: str) -> str:
    """Drop leftovers trafilatura keeps: footnote markers like <sup>[5]</sup> (which
    would collide with our own [n] citations) and wiki "[edit]" links."""
    md = re.sub(r"<sup>.*?</sup>", "", md)
    md = re.sub(r"\\?\[edit\\?\]", "", md)
    return re.sub(r"</?\w+[^>]*>", "", md).strip()


class WebLoader:
    """Fetch pages politely: robots.txt respected, same-domain BFS with a depth
    and page limit, boilerplate (menus, footers) stripped by trafilatura."""

    def __init__(self, user_agent: str, max_pages: int = 25, client: httpx.Client | None = None):
        self.user_agent, self.max_pages = user_agent, max_pages
        self.client = client or httpx.Client(timeout=20, follow_redirects=True)
        self.client.headers["User-Agent"] = user_agent
        self._robots: dict[str, RobotFileParser] = {}
        self.skipped: list[tuple[str, str]] = []

    def allowed(self, url: str) -> bool:
        host = "{0.scheme}://{0.netloc}".format(urlparse(url))
        if host not in self._robots:
            rp = RobotFileParser()
            try:
                r = self.client.get(f"{host}/robots.txt")
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])  # no robots.txt = allowed
            except httpx.HTTPError:
                rp.parse([])
            self._robots[host] = rp
        return self._robots[host].can_fetch(self.user_agent, url)

    def fetch(self, url: str) -> tuple[list[Document], list[str]]:
        """One URL -> (documents, same-site links found on it)."""
        r = self.client.get(url)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "pdf" in ctype:
            reader = PdfReader(io.BytesIO(r.content))
            return [Document(t, metadata={"source": url, "type": "web", "title": url, "url": url, "page": n})
                    for n, p in enumerate(reader.pages, 1) if (t := (p.extract_text() or "").strip())], []
        if "html" not in ctype:
            raise SkipFile(f"content-type {ctype!r}")
        text = clean_web(trafilatura.extract(r.text, url=url, output_format="markdown", include_tables=True,
                                             include_comments=False) or "")
        if not text:
            raise SkipFile("no main content found")
        meta = trafilatura.extract_metadata(r.text, default_url=url)
        title = (meta.title if meta and meta.title else None) or url
        return ([Document(text, metadata={"source": url, "type": "web", "title": title, "url": url})],
                self.links(r.text, str(r.url)))

    @staticmethod
    def links(html: str, base: str) -> list[str]:
        host = urlparse(base).netloc
        out = []
        for href in lxml_html.fromstring(html).xpath("//a/@href"):
            url = urldefrag(urljoin(base, href)).url
            p = urlparse(url)
            if (p.scheme in ("http", "https") and p.netloc == host
                    and not re.search(r"\.(png|jpe?g|gif|svg|zip|gz|mp4|css|js)$", p.path, re.I)):
                out.append(url)
        return list(dict.fromkeys(out))

    def crawl(self, start: str, depth: int = 0) -> list[Document]:
        """BFS from `start`; depth 0 = just that page."""
        docs, seen, queue = [], {start}, deque([(start, 0)])
        while queue and len(seen) - len(queue) < self.max_pages:
            url, d = queue.popleft()
            if not self.allowed(url):
                self.skipped.append((url, "disallowed by robots.txt"))
                continue
            try:
                page_docs, links = self.fetch(url)
            except (httpx.HTTPError, SkipFile) as e:
                self.skipped.append((url, str(e) or type(e).__name__))
                continue
            docs += page_docs
            if d < depth:
                for link in links:
                    if link not in seen:
                        seen.add(link)
                        queue.append((link, d + 1))
        return docs
