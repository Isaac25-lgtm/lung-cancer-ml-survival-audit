"""Build the submission manuscript (.docx) from docs/manuscript.md, embedding figures.

Usage: python -m src.build_manuscript_docx
Output: outputs/manuscript/manuscript.docx
"""
from __future__ import annotations

import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from src.paths import FIGURES, OUTPUTS, ROOT

SRC = ROOT / "docs" / "manuscript.md"
OUT = OUTPUTS / "manuscript" / "manuscript.docx"

MATH = [(r"\sum_{a,b} w_{ab} c_{ab}", "Σ w_ab·c_ab"), (r"\tilde Y", "Ỹ"), (r"\mathcal A", "𝒜"),
        (r"\mathcal{A}", "𝒜"), ("$", "")]
FIG_FILES = {"0_three_datasets": FIGURES / "readme" / "0_three_datasets.png",
             "fig2_polarity_mechanism": FIGURES / "fig2_polarity_mechanism.png",
             "3_hidden_in_average": FIGURES / "readme" / "3_hidden_in_average.png",
             "6_primary_contrast": FIGURES / "readme" / "6_primary_contrast.png",
             "5_wrong_winner": FIGURES / "readme" / "5_wrong_winner.png",
             "4_cohort_flip": FIGURES / "readme" / "4_cohort_flip.png"}


def clean(text: str) -> str:
    for a, b in MATH:
        text = text.replace(a, b)
    return text.replace("\\*", "*")


def add_inline(par, text: str) -> None:
    """Render **bold**, *italic* and `code` runs."""
    text = clean(text)
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            par.add_run(tok[1:-1]).italic = True
        elif tok.startswith("`") and tok.endswith("`"):
            r = par.add_run(tok[1:-1])
            r.font.name = "Consolas"
        else:
            par.add_run(tok)


def build() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)
    st.paragraph_format.line_spacing = 1.5
    st.paragraph_format.space_after = Pt(6)
    figure_legends = []
    i = 0
    in_legends = False
    while i < len(lines):
        ln = lines[i]
        if ln.strip() in ("---", ""):
            i += 1
            continue
        if ln.startswith("# "):
            p = doc.add_heading(clean(ln[2:]), level=0)
            i += 1
            continue
        if ln.startswith("## "):
            in_legends = ln.strip() == "## Figure legends"
            doc.add_heading(clean(ln[3:]), level=1)
            i += 1
            continue
        if ln.startswith("### "):
            doc.add_heading(clean(ln[4:]), level=2)
            i += 1
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            t = doc.add_table(rows=len(rows), cols=len(rows[0]))
            t.style = "Table Grid"
            for r, row in enumerate(rows):
                for c, val in enumerate(row):
                    cell = t.cell(r, c)
                    cell.text = ""
                    par = cell.paragraphs[0]
                    add_inline(par, val)
                    for run in par.runs:
                        run.font.size = Pt(9)
                        if r == 0:
                            run.bold = True
            doc.add_paragraph()
            continue
        m = re.match(r"^(\s*)(-|\d+\.)\s+(.*)$", ln)
        if m:
            style = "List Bullet" if m.group(2) == "-" else "List Number"
            if m.group(1):
                style += " 2"
            add_inline(doc.add_paragraph(style=style), m.group(3))
            i += 1
            continue
        par = doc.add_paragraph()
        add_inline(par, ln)
        if in_legends:
            key = re.search(r"`([^`]+)`", ln)
            if key and key.group(1).split("/")[-1].replace(".png", "") in FIG_FILES:
                figure_legends.append((ln, FIG_FILES[key.group(1).split("/")[-1].replace(".png", "")]))
        i += 1

    doc.add_page_break()
    doc.add_heading("Figures", level=1)
    for legend, path in figure_legends:
        if path.exists():
            doc.add_picture(str(path), width=Inches(6.3))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph()
            add_inline(cap, re.sub(r"\s*\(`[^`]+`\)", "", legend))
            for run in cap.runs:
                run.font.size = Pt(9.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print("manuscript ->", OUT, f"({len(figure_legends)} figures)")


if __name__ == "__main__":
    build()
