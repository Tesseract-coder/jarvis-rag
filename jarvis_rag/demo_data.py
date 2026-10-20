"""Write a small demo corpus into data/ so ingestion has every input type to chew on.

    python -m jarvis_rag.demo_data        (or: make demo-data)

Creates, for the fictional Halcyon Robotics used since learn/:
    data/docs/*.md                  copies of learn/sample_docs
    data/docs/safety_report_2026.pdf  3-page PDF  (tests page citations)
    data/docs/kestrel_k2_maintenance.docx       (tests Word loading)
    data/csv/robot_sales.csv, data/csv/support_tickets.csv  (tests CSV -> SQLite)
    data/code/jarvis_rag/           a snapshot of this package (tests code ingestion)

The PDF and DOCX writers below are deliberately minimal (no reportlab or
python-docx dependency): enough structure for real parsers to read them.
"""

import csv
import random
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]

SAFETY_REPORT = [
    ("Halcyon Robotics 2026 Safety Report", [
        "Section 1: Summary",
        "This report covers all Kestrel K2 and Wren W1 robots in customer sites",
        "from January to September 2026. The fleet logged 1.9 million operating hours.",
        "There were 14 recorded safety incidents, down from 23 in 2025.",
        "No incident caused a lost-time injury. 9 incidents were classed minor,",
        "4 moderate and 1 serious (property damage above 10,000 euros).",
    ]),
    ("Section 2: Incident analysis", [
        "The most common root cause was an obstructed lidar sensor (6 incidents).",
        "Dust build-up on the lidar window reduced detection range from 12 m to 4 m.",
        "Second was floor contamination: oil or water causing wheel slip (4 incidents).",
        "The serious incident involved a Kestrel K2 colliding with a mezzanine column",
        "after firmware 4.1.0 misread a reflective safety vest as open floor.",
        "Firmware 4.1.2 fixed the reflective-surface bug and was rolled out in May.",
    ]),
    ("Section 3: Recommendations", [
        "1. Clean lidar windows weekly, or daily in dusty sites (see maintenance manual).",
        "2. Keep firmware at 4.1.2 or later on every Kestrel K2.",
        "3. Mark wet floors and pause robots in that zone until dry.",
        "4. Review speed limits in shared aisles: 1.2 m/s maximum near people.",
        "The safety committee will publish the next report in March 2027.",
    ]),
]

MAINTENANCE_MANUAL = [
    ("h", "Kestrel K2 Maintenance Manual"),
    ("p", "This manual lists routine maintenance for the Kestrel K2 autonomous mobile robot."),
    ("h", "Weekly checks"),
    ("p", "Wipe the lidar window with a dry microfibre cloth. Never use solvents: they cloud the coating."),
    ("p", "Check both drive wheels for embedded debris and tread wear below 3 mm."),
    ("p", "Confirm the emergency stop button halts the robot within 0.5 seconds."),
    ("h", "Every 4,000 operating hours"),
    ("p", "Replace the wheel motor brushes (part HK-2231). Allow 45 minutes per wheel."),
    ("p", "Recalibrate the lift platform with the HK-Cal tool; tolerance is plus or minus 2 mm."),
    ("h", "Firmware updates"),
    ("p", "Updates are installed from the fleet console under Robots > Firmware. Dock the robot "
          "first: an update on battery below 30 percent is refused. Installation takes about 12 minutes."),
    ("h", "Error E-217 after maintenance"),
    ("p", "If E-217 (wheel motor overcurrent) appears right after a brush replacement, the brush "
          "spring is usually seated backwards. Reseat it before contacting support."),
]


# --------------------------------------------------------------------------- minimal writers

def _pdf_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages: list[tuple[str, list[str]]], title: str = "") -> None:
    """Text-only PDF 1.4: one Helvetica content stream per page, correct xref offsets."""
    objs: list[bytes] = []
    n_pages = len(pages)
    page_ids = [4 + 2 * i for i in range(n_pages)]
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n_pages} >>".encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for heading, lines in pages:
        ops = ["BT", "/F1 16 Tf", "72 740 Td", "20 TL", f"({_pdf_str(heading)}) Tj", "T*", "/F1 11 Tf", "15 TL"]
        ops += [f"({_pdf_str(line)}) Tj T*" for line in lines]
        stream = "\n".join(ops + ["ET"]).encode("latin-1")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                    f"/Contents {len(objs) + 2} 0 R >>".encode())
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(f"<< /Title ({_pdf_str(title)}) >>".encode())
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, len(objs), xref)
    path.write_bytes(bytes(out))


def write_docx(path: Path, blocks: list[tuple[str, str]]) -> None:
    """Smallest valid .docx: content types, package rels, and word/document.xml."""
    paras = []
    for kind, text in blocks:
        style = '<w:pPr><w:pStyle w:val="Heading1"/></w:pPr>' if kind == "h" else ""
        paras.append(f'<w:p>{style}<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>')
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f'<w:body>{"".join(paras)}</w:body></w:document>')
    content_types = ('<?xml version="1.0" encoding="UTF-8"?>'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Override PartName="/word/document.xml" ContentType="application/'
                     'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)


def write_sales_csv(path: Path, seed: int = 7) -> None:
    rng = random.Random(seed)
    price = {"Kestrel K2": 48000, "Wren W1": 14500}
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Quarter", "Region", "Product", "Units", "Revenue EUR"])
        for q in ("2025-Q4", "2026-Q1", "2026-Q2", "2026-Q3"):
            for region in ("EMEA", "Americas", "APAC"):
                for product, p in price.items():
                    units = rng.randint(8, 60) if product == "Kestrel K2" else rng.randint(30, 180)
                    w.writerow([q, region, product, units, units * p])


def write_tickets_csv(path: Path, seed: int = 11) -> None:
    rng = random.Random(seed)
    codes = ["E-104", "E-217", "E-330", "E-512"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticket_id", "opened", "product", "error_code", "hours_to_resolve", "status"])
        for n in range(1, 121):
            w.writerow([f"T-{1000 + n}", f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}",
                        rng.choice(["Kestrel K2", "Wren W1"]), rng.choices(codes, [3, 5, 2, 1])[0],
                        round(rng.uniform(0.5, 48), 1), rng.choices(["closed", "open"], [9, 1])[0]])


def write_demo_data(data_dir: Path) -> list[Path]:
    docs, csvs = data_dir / "docs", data_dir / "csv"
    docs.mkdir(parents=True, exist_ok=True)
    csvs.mkdir(parents=True, exist_ok=True)
    written = []
    for md in sorted((ROOT / "learn" / "sample_docs").glob("*.md")):
        written.append(Path(shutil.copy(md, docs / md.name)))
    write_pdf(docs / "safety_report_2026.pdf", SAFETY_REPORT, title="Halcyon Robotics 2026 Safety Report")
    write_docx(docs / "kestrel_k2_maintenance.docx", MAINTENANCE_MANUAL)
    write_sales_csv(csvs / "robot_sales.csv")
    write_tickets_csv(csvs / "support_tickets.csv")
    code = data_dir / "code" / "jarvis_rag"
    # demo_data.py holds the demo documents as strings; copying it would
    # make code chunks compete with the documents themselves.
    shutil.copytree(ROOT / "jarvis_rag", code, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "demo_data.py"))
    written += [code, docs / "safety_report_2026.pdf", docs / "kestrel_k2_maintenance.docx",
                csvs / "robot_sales.csv", csvs / "support_tickets.csv"]
    return written


if __name__ == "__main__":
    from jarvis_rag.config import get_settings

    for p in write_demo_data(get_settings().data_dir):
        print(f"  wrote {p}")
