"""Collect every chart with its caption, finding and decision into one review PDF.

Run after src.eda, src.compare, src.checks, src.diagnostics:
    python make_figure_book.py
Writes report/Figure_Book.html and report/Figure_Book.pdf (PDF needs Playwright + Chromium).
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

SECTIONS = [
    ("eda", "1. Exploratory data analysis", "What the data looks like, what is wrong with it, and what that implies."),
    ("models", "2. Model comparison", "21 candidates on the same three time-based folds, same rows, same metrics."),
    ("checks", "3. Robustness checks", "Each check answers one review question about leakage, assumptions or claims."),
    ("diagnostics", "4. Diagnostics of the chosen model", "Where the submitted model errs and whether its "
                                                             "predictions look sane."),
]
DEC = Path("scorer_results/candidate_december.png")

CSS = """
:root { --ink:#1F2D33; --accent:#064A56; --muted:#5d6f74; --rule:#d9e2e4; --bg:#ffffff; --soft:#f3f7f8; }
* { box-sizing: border-box; }
body { font-family: Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif; color: var(--ink);
       background: var(--bg); margin: 0; padding: 32px 40px; line-height: 1.45; font-size: 13px; }
h1 { color: var(--accent); font-size: 26px; margin: 0 0 4px; }
h2 { color: var(--accent); font-size: 19px; border-bottom: 2px solid var(--accent); padding-bottom: 4px;
     margin-top: 36px; page-break-before: always; }
.sub { color: var(--muted); margin: 0 0 18px; }
.toc { background: var(--soft); padding: 12px 18px; border-radius: 6px; }
.toc li { margin: 2px 0; }
figure { margin: 18px 0 26px; page-break-inside: avoid; border: 1px solid var(--rule); border-radius: 6px;
         padding: 12px 14px; }
figure h3 { margin: 0 0 8px; font-size: 14px; }
figure .id { color: var(--muted); font-weight: 400; font-size: 11px; margin-left: 6px; }
figure img { width: 100%; height: auto; display: block; margin: 4px 0 10px; }
dl { display: grid; grid-template-columns: 92px 1fr; gap: 4px 12px; margin: 0; }
dt { font-weight: 600; color: var(--accent); }
dd { margin: 0; }
.keep { display: inline-block; margin-top: 8px; font-size: 11px; color: var(--muted); }
@media print { body { padding: 0 6px; } h2 { margin-top: 0; } }
"""


def img_tag(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(path.stem)}">'


def main():
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Figure Book</title>",
             f"<style>{CSS}</style></head><body>",
             "<h1>Freight Rate Prediction — Figure Book</h1>",
             "<p class='sub'>Every chart produced by the pipeline, with what it shows, what it found and the "
             "decision it supports. Figure IDs match the files in <code>outputs/figures/</code>. Use this to "
             "choose which charts go into the report and video.</p><div class='toc'><b>Contents</b><ol>"]
    loaded = []
    for key, title, blurb in SECTIONS:
        meta = Path(f"outputs/figures/{key}/captions.json")
        items = json.loads(meta.read_text()) if meta.exists() else []
        items = sorted(items, key=lambda i: i["file"])
        loaded.append((key, title, blurb, items))
        parts.append(f"<li>{html.escape(title)} ({len(items)} charts)</li>")
    parts.append("<li>5. Required deliverable: December chart from score.py</li></ol></div>")
    n = 0
    for key, title, blurb, items in loaded:
        parts.append(f"<h2>{html.escape(title)}</h2><p class='sub'>{html.escape(blurb)}</p>")
        for it in items:
            n += 1
            fid = f"{key}/{it['file']}"
            rows = [("Shows", it["shows"]), ("Finding", it["finding"])]
            if it.get("decision"):
                rows.append(("Decision", it["decision"]))
            dl = "".join(f"<dt>{k}</dt><dd>{html.escape(v)}</dd>" for k, v in rows)
            parts.append(f"<figure><h3>{html.escape(it['title'])}<span class='id'>{fid}</span></h3>"
                         f"{img_tag(Path('outputs/figures') / key / it['file'])}<dl>{dl}</dl></figure>")
    parts.append("<h2>5. Required deliverable: December chart</h2>")
    if DEC.exists():
        parts.append(f"<figure><h3>Output of the provided score.py<span class='id'>{DEC}</span></h3>"
                     f"{img_tag(DEC)}<dl><dt>Shows</dt><dd>Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb, "
                     "priced for every day of December 2025 by the submitted model.</dd></dl></figure>")
    parts.append("</body></html>")
    out = Path("report")
    out.mkdir(exist_ok=True)
    (out / "Figure_Book.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote report/Figure_Book.html ({n} charts)")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page()
            page.goto((out / "Figure_Book.html").resolve().as_uri())
            page.pdf(path=str(out / "Figure_Book.pdf"), format="A4", print_background=True,
                     margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"})
            b.close()
        print("Wrote report/Figure_Book.pdf")
    except Exception as exc:  # PDF is optional
        print(f"PDF skipped ({exc}); open the HTML file instead.")


if __name__ == "__main__":
    main()
