"""Convert session summary markdown to PDF via weasyprint."""
import markdown
from weasyprint import HTML
from pathlib import Path

DOCS_DIR = Path(__file__).parent.resolve()
INPUT = DOCS_DIR / "session_summary_22_verify_detections.md"
OUTPUT = DOCS_DIR / "session_summary_22_verify_detections.pdf"

with open(INPUT) as f:
    md_text = f.read()

# Strip YAML front matter if present
if md_text.startswith("---"):
    end = md_text.index("---", 3)
    md_text = md_text[end + 3:].strip()

html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

CSS = """
body {
    font-family: "Courier New", "Courier", monospace;
    font-size: 10pt;
    line-height: 1.45;
    max-width: 7.5in;
    margin: 0.75in auto;
    color: #222;
}
h1 { font-size: 16pt; border-bottom: 2px solid #0072B2; padding-bottom: 4px; }
h2 { font-size: 13pt; color: #0072B2; margin-top: 1.2em; }
h3 { font-size: 11pt; color: #444; margin-top: 1em; }
table { border-collapse: collapse; width: 100%; margin: 0.5em 0; font-size: 9pt; }
th, td { border: 1px solid #ccc; padding: 3px 6px; text-align: left; }
th { background: #f0f0f0; }
code { background: #f5f5f5; padding: 1px 4px; font-size: 9pt; }
pre { background: #f5f5f5; padding: 8px; font-size: 8pt; overflow-x: auto; }
pre code { background: none; padding: 0; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
ol, ul { padding-left: 1.5em; }
strong { color: #0072B2; }
em { font-style: italic; }
"""

full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body></html>"""

HTML(string=full_html).write_pdf(str(OUTPUT))
print(f"PDF saved: {OUTPUT}")
print(f"Size: {OUTPUT.stat().st_size / 1024:.0f} KB")
