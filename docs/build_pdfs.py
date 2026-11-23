"""Render the Chatterbox docs to PDF (markdown -> styled HTML -> headless Chrome).

Usage:
    python docs/build_pdfs.py            # all *.md in this folder
    python docs/build_pdfs.py 03         # only files whose name starts with 03

The HTML is written next to the markdown as <name>.html so the print output can be
inspected without re-running the pipeline.
"""

from __future__ import annotations

import html
import os
import shutil
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parent
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.4pt; line-height: 1.52; color: #1a1a1a; margin: 0;
  max-width: 100%;
}
h1 { font-size: 21pt; line-height: 1.2; margin: 0 0 4mm; color: #111;
     border-bottom: 2.5pt solid #2d6cdf; padding-bottom: 3mm; page-break-before: auto; }
h2 { font-size: 14.5pt; margin: 9mm 0 2mm; color: #14304f; page-break-after: avoid;
     border-left: 3.5pt solid #2d6cdf; padding-left: 3mm; }
h3 { font-size: 11.6pt; margin: 6mm 0 1.5mm; color: #1d4470; page-break-after: avoid; }
h4 { font-size: 10.4pt; margin: 4mm 0 1mm; color: #333; page-break-after: avoid; }
p { margin: 0 0 2.4mm; }
ul, ol { margin: 0 0 2.6mm 5mm; padding-left: 3mm; }
li { margin: 0.8mm 0; }
li > p { margin: 0.5mm 0; }
code {
  font-family: "Cascadia Mono", Consolas, "Courier New", monospace;
  font-size: 8.6pt; background: #f2f4f7; padding: 0.3mm 1.1mm;
  border-radius: 1.5pt; color: #b3005f; word-break: break-word;
}
pre {
  background: #f7f9fb; border: 0.5pt solid #dde3ea; border-left: 3pt solid #6f8fb8;
  border-radius: 2pt; padding: 2.5mm 3mm; margin: 2mm 0 3mm; overflow-wrap: normal;
  page-break-inside: avoid; white-space: pre-wrap;
}
pre code {
  background: none; padding: 0; color: #16324f; font-size: 8.35pt; line-height: 1.38;
  word-break: normal;
}
blockquote {
  margin: 2.5mm 0; padding: 2mm 3mm; background: #fff8e6;
  border-left: 3pt solid #e0a800; color: #4a3b00; page-break-inside: avoid;
}
blockquote p:last-child { margin-bottom: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 2mm 0 4mm; font-size: 9.05pt;
  page-break-inside: auto;
}
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th {
  background: #eaf0f9; text-align: left; padding: 1.4mm 1.8mm;
  border: 0.5pt solid #c3cfdd; font-weight: 600; color: #16324f;
}
td { padding: 1.3mm 1.8mm; border: 0.5pt solid #d7dee7; vertical-align: top; }
tbody tr:nth-child(even) td { background: #fafbfd; }
td code, th code { font-size: 8.1pt; }
hr { border: none; border-top: 0.6pt solid #ccd4de; margin: 7mm 0; }
a { color: #1c5bb8; text-decoration: none; }
.docmeta {
  font-size: 8.6pt; color: #5b6b7d; margin: 0 0 7mm; letter-spacing: 0.2pt;
}
h1 + .docmeta { margin-top: 2mm; }
strong { color: #0d2b45; }
"""


def md_html(text: str) -> str:
    md = MarkdownIt("gfm-like", {"linkify": False, "typographer": False})
    return md.render(text)


def build(md_path: Path) -> Path:
    body = md_html(md_path.read_text(encoding="utf-8"))
    title = md_path.stem
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            title = html.escape(line[2:].strip())
            break
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head>
<body>{body}</body></html>
"""
    html_path = md_path.with_suffix(".html")
    html_path.write_text(page, encoding="utf-8")
    return html_path


def to_pdf(html_path: Path, browser: str) -> Path:
    pdf_path = html_path.with_suffix(".pdf")
    if pdf_path.exists():
        pdf_path.unlink()
    url = "file:///" + str(html_path).replace("\\", "/")
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-first-run",
        "--disable-extensions",
        "--virtual-time-budget=20000",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        "--no-pdf-header-footer",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not pdf_path.exists():
        raise RuntimeError(f"no PDF produced for {html_path}\n{proc.stderr[-2000:]}")
    return pdf_path


def main(argv: list[str]) -> int:
    browser = next((b for b in BROWSERS if shutil.which(b) or os.path.exists(b)), None)
    if browser is None:
        print("no Chrome/Edge found; HTML was generated but no PDF")
        return 1
    prefix = argv[1] if len(argv) > 1 else ""
    docs = sorted(p for p in HERE.glob("*.md")
                  if not p.name.startswith("_") and p.name.lower() != "readme.md")
    docs = [p for p in docs if p.name.startswith(prefix)]
    if not docs:
        print("no markdown files matched")
        return 1
    for md_path in docs:
        html_path = build(md_path)
        pdf_path = to_pdf(html_path, browser)
        size = pdf_path.stat().st_size / 1024
        print(f"{md_path.name:52s} -> {pdf_path.name}  ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
