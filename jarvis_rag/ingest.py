"""load -> clean -> chunk -> embed -> upsert.

    python -m jarvis_rag.ingest --path data/docs --path data/code \\
        --url https://example.com --crawl-depth 1 --csv data/csv
    python -m jarvis_rag.ingest --list
    python -m jarvis_rag.ingest --delete https://example.com/old-page

Re-running is cheap: unchanged chunks are not re-embedded, and files deleted
from an ingested folder are removed from the index (unless --no-prune).

Every chunk starts with a one-line header naming its document and section,
e.g. "[product_specs.md > Wren W1 > Specifications]". learn/06 showed why: a
chunk saying "Battery life: 9 hours" is useless if it doesn't say which robot.
The header is embedded too, so it also helps retrieval.
"""

import argparse
import re
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (Language, MarkdownHeaderTextSplitter,
                                      RecursiveCharacterTextSplitter)

from jarvis_rag.config import Settings, get_settings
from jarvis_rag.loaders import (SkipFile, WebLoader, find_symbols, iter_files, load_csv,
                                load_file)
from jarvis_rag.vectorstore import KnowledgeBase, UpsertResult, chunk_id

HEADINGS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


class Chunker:
    def __init__(self, s: Settings):
        self.text = RecursiveCharacterTextSplitter(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap,
                                                   add_start_index=True)
        self.code_size, self.code_overlap = s.code_chunk_size, s.code_chunk_overlap
        self.markdown = MarkdownHeaderTextSplitter(HEADINGS, strip_headers=True)

    def code_splitter(self, language: str) -> RecursiveCharacterTextSplitter:
        kw = dict(chunk_size=self.code_size, chunk_overlap=self.code_overlap, add_start_index=True)
        if language in {lang.value for lang in Language}:
            return RecursiveCharacterTextSplitter.from_language(Language(language), **kw)
        return RecursiveCharacterTextSplitter(**kw)

    def split(self, doc: Document) -> list[Document]:
        m = doc.metadata
        if m["type"] == "table":  # schema docs are small; keep whole
            return [self._finish(doc.page_content, m, f"[table {m['table']}]")]
        if m["type"] == "code":
            return self._split_code(doc)
        if m["type"] in ("text", "web", "docx") and "page" not in m:
            return self._split_markdown(doc)
        where = f"{m['title']}, page {m['page']}" if "page" in m else m["title"]
        return [self._finish(c.page_content, m, f"[{where}]") for c in self.text.split_documents([doc])]

    def _split_markdown(self, doc: Document) -> list[Document]:
        m, out = doc.metadata, []
        for section in self.markdown.split_text(doc.page_content):
            trail = [section.metadata[k] for _, k in HEADINGS if k in section.metadata]
            if trail and trail[0] == m["title"]:
                trail = trail[1:]  # don't repeat the title as the first heading
            label = " > ".join([m["title"], *trail])
            for c in self.text.split_text(section.page_content):
                out.append(self._finish(c, m | ({"section": " > ".join(trail)} if trail else {}), f"[{label}]"))
        return out

    def _split_code(self, doc: Document) -> list[Document]:
        m, text = doc.metadata, doc.page_content
        symbols = find_symbols(text, m["language"])
        out = []
        for c in self.code_splitter(m["language"]).split_documents([doc]):
            start = c.metadata["start_index"]
            end = start + len(c.page_content)
            inside = [name for pos, name in symbols if start <= pos < end]
            before = [name for pos, name in symbols if pos < start]
            names = inside or before[-1:]  # a chunk mid-function belongs to that function
            meta = m | ({"symbols": ", ".join(dict.fromkeys(names))} if names else {})
            # The line number lives in metadata only: putting it in the text would change
            # every later chunk's id (and force re-embedding) whenever a line is added above.
            line = text.count("\n", 0, start) + 1
            header = f"[{m['path']} ({m['language']})" + (f": {meta['symbols']}" if names else "") + "]"
            out.append(self._finish(c.page_content, meta | {"line": line}, header))
        return out

    @staticmethod
    def _finish(body: str, meta: dict, header: str) -> Document:
        content = f"{header}\n{body.strip()}"
        meta = {k: v for k, v in meta.items() if k != "start_index" and v is not None}
        return Document(content, metadata=meta | {"chunk_id": chunk_id(meta["source"], content)})


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)  # trailing whitespace
    return re.sub(r"\n{4,}", "\n\n\n", text)


def collect(paths: list[Path], urls: list[str], csvs: list[Path], crawl_depth: int,
            s: Settings) -> tuple[list[Document], list[tuple[str, str]]]:
    """Load everything; return (documents, [(what, why_skipped)])."""
    docs, skipped = [], []
    max_bytes = int(s.max_file_mb * 1e6)
    missing = [p for p in [*paths, *csvs] if not p.exists()]
    skipped += [(str(p), "does not exist") for p in missing]
    files = [f for p in paths if p not in missing for f in (iter_files(p) if p.is_dir() else [p])]
    files += [f for p in csvs if p not in missing for f in (sorted(p.rglob("*.csv")) if p.is_dir() else [p])]
    for f in files:
        try:
            docs += load_csv(f, s.sqlite_path) if f.suffix.lower() == ".csv" else load_file(f, max_bytes)
        except SkipFile as e:
            skipped.append((str(f), str(e)))
        except Exception as e:  # one broken file must not abort a 1000-file ingest
            skipped.append((str(f), f"{type(e).__name__}: {e}"))
    if urls:
        web = WebLoader(s.user_agent, s.url_max_pages)
        for url in urls:
            docs += web.crawl(url, crawl_depth)
        skipped += web.skipped
    for d in docs:
        d.page_content = clean(d.page_content)
    return docs, skipped


def ingest_documents(docs: list[Document], kb: KnowledgeBase, s: Settings) -> list[UpsertResult]:
    chunker, now = Chunker(s), datetime.now(UTC).isoformat(timespec="seconds")
    by_source: dict[str, list[Document]] = defaultdict(list)
    for d in docs:
        for c in chunker.split(d):
            c.metadata["ingested_at"] = now
            by_source[c.metadata["source"]].append(c)
    return [kb.upsert_source(src, chunks) for src, chunks in by_source.items()]


def prune(kb: KnowledgeBase, roots: list[Path]) -> list[str]:
    """Delete sources that live under one of `roots` but no longer exist on disk,
    so ingesting a folder makes the index mirror it (deletions included)."""
    roots = [r.resolve() for r in roots if r.is_dir()]
    gone = [src for src, info in kb.sources().items()
            if info["type"] != "web" and not Path(src).exists()
            and any(Path(src).resolve().is_relative_to(r) for r in roots)]
    for src in gone:
        kb.delete_source(src)
    return gone


def print_sources(kb: KnowledgeBase) -> None:
    sources = kb.sources()
    print(f"{len(sources)} sources, {kb.count()} chunks")
    for src, info in sources.items():
        print(f"  {info['chunks']:>5}  {info['type']:<6} {src}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest files, folders, URLs and CSVs into the knowledge base.")
    ap.add_argument("--path", type=Path, action="append", default=[], help="file or folder (repeatable)")
    ap.add_argument("--url", action="append", default=[], help="web page (repeatable)")
    ap.add_argument("--crawl-depth", type=int, default=0, help="follow same-site links this many hops")
    ap.add_argument("--csv", type=Path, action="append", default=[], help="CSV file or folder -> SQLite")
    ap.add_argument("--list", action="store_true", help="list ingested sources")
    ap.add_argument("--delete", action="append", default=[], metavar="SOURCE", help="remove a source")
    ap.add_argument("--no-prune", action="store_true",
                    help="keep indexed files that were deleted from an ingested folder")
    args = ap.parse_args()

    s = get_settings()
    kb = KnowledgeBase(s)
    for src in args.delete:
        print(f"deleted {kb.delete_source(src)} chunks of {src}")
    if not (args.path or args.url or args.csv):
        if args.list or not args.delete:
            print_sources(kb)
        return

    t0 = time.perf_counter()
    docs, skipped = collect(args.path, args.url, args.csv, args.crawl_depth, s)
    t_load = time.perf_counter() - t0
    results = ingest_documents(docs, kb, s)
    pruned = [] if args.no_prune else prune(kb, args.path + args.csv)
    t_total = time.perf_counter() - t0

    print(f"{'chunks':>7} {'added':>6} {'removed':>8} {'unchanged':>10}  source")
    for r in results:
        print(f"{r.chunks:>7} {r.added:>6} {r.removed:>8} {r.unchanged:>10}  {r.source}")
    for src in pruned:
        print(f"  pruned {src} (no longer on disk)")
    for what, why in skipped:
        print(f"  skipped {what}: {why}")
    added = sum(r.added for r in results)
    print(f"\n{len(results)} sources, {sum(r.chunks for r in results)} chunks ({added} embedded) "
          f"in {t_total:.1f}s (load {t_load:.1f}s). Collection now holds {kb.count()} chunks.")
    if args.list:
        print_sources(kb)


if __name__ == "__main__":
    main()
