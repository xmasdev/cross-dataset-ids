"""Markdown -> PDF (pandoc -> typst) plus DOCX, or PPTX for files with "presentation" in the name.

    python tools/build_docs.py reports/09_final_report.md reports/10_final_presentation.md
"""
import sys, os, pypandoc, typst
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for src in [a for a in sys.argv[1:] if a.endswith(".md")]:
    d, base = os.path.dirname(os.path.abspath(src)), os.path.splitext(os.path.basename(src))[0]
    slides = "presentation" in base
    cwd = os.getcwd(); os.chdir(d)
    try:
        typ = base + ".typ"
        args = ["--standalone", "--columns=200"] + ([] if slides else ["-V", "papersize=a4", "-V", "margin.x=2.2cm", "-V", "margin.y=2.2cm", "-V", "fontsize=10.5pt"])
        pypandoc.convert_file(base + ".md", "typst", outputfile=typ, extra_args=args)
        typst.compile(typ, output=base + ".pdf"); os.remove(typ)
        if slides:
            import build_pptx; build_pptx.build(base + ".md")
        elif "--no-docx" not in sys.argv:
            pypandoc.convert_file(base + ".md", "docx", outputfile=base + ".docx")
        print("built", base)
    finally:
        os.chdir(cwd)
