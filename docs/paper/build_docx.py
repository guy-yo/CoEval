#!/usr/bin/env python
"""Build CoEval.docx from index.html using the html2doc skill.

Pipeline:
  1. Preprocess index.html for the Word build (drop the web-only ".docx"
     download link; turn the title <br/> into a space so it does not render
     "TasksWithout").
  2. html2doc three-stage conversion (KaTeX -> MathML -> DOCX/OMML -> styled),
     camera-ready-generic profile, native editable Word equations.
  3. Post-style: center the FULL author/affiliation block (the skill's
     front-matter detector handles a single author; this paper has two) and give
     every figure breathing room so it does not glue to the preceding table.

Run from docs/paper/:  python build_docx.py
"""
from __future__ import annotations
import os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL = Path(os.environ.get("HTML2DOC_SKILL", r"C:/Users/apart/.claude/skills/html2doc"))
SRC = HERE / "index.html"
OUT = HERE / "CoEval.docx"
PRE = HERE / "_forword.html"
MML = HERE / "_mathml.html"
RAW = HERE / "_converted.docx"


def preprocess() -> None:
    s = SRC.read_text(encoding="utf-8")
    s = re.sub(r'<a class="docxlink"[^>]*>.*?</a>\s*', "", s, flags=re.DOTALL)
    s = s.replace("Custom Tasks<br />Without", "Custom Tasks Without")
    s = s.replace("Custom Tasks<br/>Without", "Custom Tasks Without")
    PRE.write_text(s, encoding="utf-8")


def convert() -> None:
    env = dict(os.environ, NODE_PATH=str(SKILL / "node_modules"))
    subprocess.run(["node", str(SKILL / "scripts/katex_to_mathml.js"),
                    "--input", str(PRE), "--output", str(MML)], check=True, env=env)
    subprocess.run([sys.executable, str(SKILL / "scripts/convert_to_docx.py"),
                    "--input", str(MML), "--output", str(RAW),
                    "--profile", "camera-ready-generic"], check=True)
    subprocess.run([sys.executable, str(SKILL / "scripts/apply_academic_style.py"),
                    "--input", str(RAW), "--output", str(OUT),
                    "--profile", "camera-ready-generic"], check=True)


def poststyle() -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt
    from docx.oxml.ns import qn

    d = Document(str(OUT))
    paras = d.paragraphs
    ti = ai = None
    for i, p in enumerate(paras):
        if ti is None and p.style.name == "Title":
            ti = i
        if p.text.strip() == "Abstract":
            ai = i
            break
    if ti is not None and ai is not None:
        for p in paras[ti + 1:ai]:
            if p.text.strip():
                p.style = d.styles["Subtitle"]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for p in paras:
        if p._element.findall(".//" + qn("w:drawing")):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(6)
    d.save(str(OUT))


def cleanup() -> None:
    for f in (PRE, MML, RAW):
        try:
            f.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    preprocess()
    convert()
    poststyle()
    cleanup()
    print(f"Built {OUT}")
