"""Slides markdown (one '# Title' per slide; bullets, numbered lists, pipe tables, one image) -> 16:9 PPTX.

    python tools/build_pptx.py reports/10_final_presentation.md
"""
import os, re, sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from PIL import Image

W, H, MX = 13.333, 7.5, 0.55
INK, INK2, BLUE, RULE = RGBColor(0x0b, 0x0b, 0x0b), RGBColor(0x52, 0x51, 0x4e), RGBColor(0x1c, 0x5c, 0xab), RGBColor(0xc3, 0xc2, 0xb7)


def runs(par, text, size, color=INK, bold=False):
    """Write text with **bold** / *italic* markdown spans into a paragraph."""
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not tok:
            continue
        r = par.add_run(); r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold
        if tok.startswith("**"):
            r.text = tok[2:-2]; r.font.bold = True
        elif tok.startswith("*"):
            r.text = tok[1:-1]; r.font.italic = True
        elif tok.startswith("`"):
            r.text = tok[1:-1]; r.font.name = "Consolas"
        else:
            r.text = tok


def parse(md):
    md = re.sub(r"```\{=typst\}.*?```", "", md, flags=re.S)
    meta = dict(re.findall(r'^(\w+): "(.*)"$', md.split("---")[1], flags=re.M)) if md.startswith("---") else {}
    body = md.split("---", 2)[2] if md.startswith("---") else md
    slides = []
    for chunk in re.split(r"^# ", body, flags=re.M)[1:]:
        lines = chunk.strip("\n").split("\n"); title = lines[0].strip(); blocks = []; i = 1
        while i < len(lines):
            ln = lines[i].rstrip()
            if not ln.strip():
                i += 1; continue
            m = re.match(r"!\[.*?\]\((.*?)\)", ln)
            if m:
                blocks.append(("img", m.group(1))); i += 1
            elif ln.startswith("|"):
                rows = []
                while i < len(lines) and lines[i].startswith("|"):
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                        rows.append(cells)
                    i += 1
                blocks.append(("table", rows))
            else:
                para = []
                while i < len(lines) and lines[i].strip() and not lines[i].startswith("|") and not lines[i].startswith("!["):
                    s = lines[i].rstrip()
                    if re.match(r"^(\s*)(- |\d+\. )", s) or not para:
                        para.append(s)
                    else:
                        para[-1] += " " + s.strip()
                    i += 1
                blocks.append(("text", para))
        slides.append((title, blocks))
    return meta, slides


def text_height(items, size, width):
    cpl = width * 72 / (size * 0.5)                       # rough characters per line
    n = sum(max(1, -(-len(re.sub(r"[*`]", "", s)) // int(cpl))) for s in items)
    return n * size * 1.32 / 72 + 0.12


def build(src):
    meta, slides = parse(open(src, encoding="utf-8").read()); base = os.path.dirname(os.path.abspath(src))
    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H); blank = prs.slide_layouts[6]

    s = prs.slides.add_slide(blank)                        # title slide
    tb = s.shapes.add_textbox(Inches(0.9), Inches(1.7), Inches(W - 1.8), Inches(2.2)); tf = tb.text_frame; tf.word_wrap = True
    runs(tf.paragraphs[0], meta.get("title", ""), 32, INK, True)
    for key, size, col in (("subtitle", 20, BLUE), ("author", 14, INK2), ("date", 14, INK2)):
        p = tf.add_paragraph(); p.space_before = Pt(16); runs(p, meta.get(key, ""), size, col)

    for n, (title, blocks) in enumerate(slides, start=2):
        s = prs.slides.add_slide(blank)
        tb = s.shapes.add_textbox(Inches(MX), Inches(0.3), Inches(W - 2 * MX), Inches(0.8)); tb.text_frame.word_wrap = True
        runs(tb.text_frame.paragraphs[0], title, 26, INK, True)
        ln = s.shapes.add_connector(1, Inches(MX), Inches(1.08), Inches(W - MX), Inches(1.08)); ln.line.color.rgb = RULE; ln.line.width = Pt(1)
        pn = s.shapes.add_textbox(Inches(W - 1.2), Inches(H - 0.45), Inches(0.8), Inches(0.3)); pn.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
        runs(pn.text_frame.paragraphs[0], str(n), 10, INK2)

        avail = H - 1.25 - 0.5; width = W - 2 * MX
        has_img = any(k == "img" for k, _ in blocks); size = 16 if has_img or any(k == "table" for k, _ in blocks) else 18
        while True:                                        # shrink text until the fixed blocks fit
            fixed = sum(text_height(v, size, width) if k == "text" else (len(v) * (size - 2) * 1.75 / 72 + 0.15 if k == "table" else 0) for k, v in blocks)
            if fixed <= avail - (2.6 if has_img else 0) or size <= 11:
                break
            size -= 1
        y = 1.25
        for kind, v in blocks:
            if kind == "text":
                h = text_height(v, size, width); tb = s.shapes.add_textbox(Inches(MX), Inches(y), Inches(width), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
                for j, item in enumerate(v):
                    p = tf.paragraphs[0] if j == 0 else tf.add_paragraph(); p.space_after = Pt(4)
                    m = re.match(r"^(\s*)- (.*)", item)
                    if m:
                        runs(p, "•  " + m.group(2), size, INK); p.level = 1 if len(m.group(1)) >= 2 else 0
                    else:
                        runs(p, item.strip(), size, INK)
                y += h
            elif kind == "table":
                rh = (size - 2) * 1.75 / 72; ncol = len(v[0]); lens = [max(len(re.sub(r"[*`]", "", r[c])) for r in v) + 2 for c in range(ncol)]
                tw = min(width, sum(lens) * (size - 2) * 0.62 / 72 + 0.3); x0 = MX + (width - tw) / 2
                gt = s.shapes.add_table(len(v), ncol, Inches(x0), Inches(y), Inches(tw), Inches(rh * len(v))).table
                for c in range(ncol):
                    gt.columns[c].width = Inches(tw * lens[c] / sum(lens))
                for r, row in enumerate(v):
                    for c, cell in enumerate(row):
                        tc = gt.cell(r, c); tc.text_frame.paragraphs[0].text = ""; tc.margin_top = tc.margin_bottom = Inches(0.03)
                        runs(tc.text_frame.paragraphs[0], cell, size - 2, RGBColor(0xff, 0xff, 0xff) if r == 0 else INK, r == 0)
                        tc.fill.solid(); tc.fill.fore_color.rgb = BLUE if r == 0 else (RGBColor(0xf4, 0xf6, 0xfb) if r % 2 else RGBColor(0xff, 0xff, 0xff))
                y += rh * len(v) + 0.2
            else:
                path = os.path.join(base, v); iw, ih = Image.open(path).size
                rest = sum(text_height(b, size, width) for k, b in blocks[blocks.index((kind, v)) + 1:] if k == "text")
                hmax = max(1.5, H - 0.5 - y - rest); wmax = width
                sc = min(wmax / iw, hmax / ih); pw, ph = iw * sc, ih * sc
                s.shapes.add_picture(path, Inches(MX + (width - pw) / 2), Inches(y), Inches(pw), Inches(ph)); y += ph + 0.1
    out = os.path.splitext(src)[0] + ".pptx"; prs.save(out); print("built", out, len(prs.slides), "slides")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        build(a)
