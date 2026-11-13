"""Export the reviewed Markdown to a portable HTML and a styled Word document."""
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

OUT = Path(__file__).resolve().parent
STEM = "CVTSP_实验完整复盘_500epoch至LKH"
PANDOC = "/home/marmot/anaconda3/bin/pandoc"
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", NS)


def q(name):
    return f"{{{NS}}}{name}"


def child(parent, name):
    e = parent.find(q(name))
    if e is None:
        e = ET.SubElement(parent, q(name))
    return e


def transform_docx(source, target, edits):
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as dst:
        for entry in src.infolist():
            raw = src.read(entry.filename)
            if entry.filename in edits:
                root = ET.fromstring(raw)
                edits[entry.filename](root)
                raw = b'<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="utf-8")
            dst.writestr(entry, raw)


def edit_styles(root):
    for rpr in [*root.findall(f".//{q('rPrDefault')}/{q('rPr')}"),
                *root.findall(f"{q('style')}/{q('rPr')}")]:
        fonts = child(rpr, "rFonts")
        for key, value in [("ascii", "Noto Sans"), ("hAnsi", "Noto Sans"),
                           ("eastAsia", "Noto Sans CJK SC"), ("cs", "Noto Sans")]:
            fonts.set(q(key), value)
        child(rpr, "lang").set(q("eastAsia"), "zh-CN")
    for style in root.findall(q("style")):
        sid = style.get(q("styleId"), "")
        if sid in ["Normal", "BodyText", "FirstParagraph", "Compact"]:
            rpr = child(style, "rPr")
            child(rpr, "sz").set(q("val"), "21")
            ppr = child(style, "pPr")
            spacing = child(ppr, "spacing")
            spacing.set(q("line"), "310")
            spacing.set(q("lineRule"), "auto")
            spacing.set(q("after"), "100")
        if sid.startswith("Heading"):
            child(child(style, "rPr"), "color").set(q("val"), "174773")


def edit_document(root):
    body=root.find(q('body'))
    toc=body.find(q('sdt'))
    if toc is not None:
        content=toc.find(q('sdtContent'))
        title=content.find(q('p'))
        # Populate a real, clickable contents list. A bare TOC field exports
        # blank in LibreOffice unless explicitly updated interactively.
        for element in list(content):
            if element is not title:
                content.remove(element)
        for paragraph in body.findall(q('p')):
            style=paragraph.find(f"{q('pPr')}/{q('pStyle')}")
            bookmark=paragraph.find(q('bookmarkStart'))
            if style is None or style.get(q('val'))!='Heading2' or bookmark is None:
                continue
            item=ET.SubElement(content,q('p'))
            ppr=child(item,'pPr')
            child(ppr,'pStyle').set(q('val'),'TOC1')
            child(ppr,'spacing').set(q('after'),'30')
            link=ET.SubElement(item,q('hyperlink'),{q('anchor'):bookmark.get(q('name'))})
            run=ET.SubElement(link,q('r'))
            child(child(run,'rPr'),'sz').set(q('val'),'19')
            ET.SubElement(run,q('t')).text=''.join(paragraph.itertext())
        end=ET.SubElement(content,q('p'))
        ET.SubElement(ET.SubElement(end,q('r')),q('br'),{q('type'):'page'})
    for sec in root.findall(f".//{q('sectPr')}"):
        size = child(sec, "pgSz")
        size.set(q("w"), "11906")
        size.set(q("h"), "16838")
        margins = child(sec, "pgMar")
        for k in ["top", "bottom", "left", "right"]:
            margins.set(q(k), "1000")
    for tbl in root.findall(f".//{q('tbl')}"):
        # Let Word fit the long numeric comparison tables inside the page.
        props = child(tbl, "tblPr")
        width=child(props,'tblW')
        width.set(q('type'),'dxa')
        width.set(q('w'),'9906')
        layout = child(props, "tblLayout")
        layout.set(q("type"), "fixed")
        columns=max(len(row.findall(q('tc'))) for row in tbl.findall(q('tr')))
        cell_width=9906//columns
        grid=child(tbl,'tblGrid')
        for element in list(grid):grid.remove(element)
        for column in range(columns):
            ET.SubElement(grid,q('gridCol'),{q('w'):str(cell_width)})
        for cell in tbl.findall(f"{q('tr')}/{q('tc')}"):
            cw=child(child(cell,'tcPr'),'tcW')
            cw.set(q('type'),'dxa')
            cw.set(q('w'),str(cell_width))
        tr = tbl.find(q("tr"))
        if tr is not None:
            child(child(tr, "trPr"), "tblHeader")
        for run in tbl.findall(f".//{q('r')}"):
            child(child(run, "rPr"), "sz").set(q("val"), "18")


css = """
html {scroll-behavior:smooth} body {max-width:1100px;margin:48px auto;padding:0 28px;
font-family:'Noto Sans CJK SC','Microsoft YaHei',sans-serif;color:#182838;line-height:1.85;font-size:15px}
h1 {font-size:30px;line-height:1.5;color:#113959;border-bottom:3px solid #267cb0;padding-bottom:18px}
h2 {margin-top:54px;padding:10px 0;border-bottom:1px solid #bcd0df;color:#174773;font-size:23px}
h3 {margin-top:32px;font-size:18px;color:#225b7d}
table {border-collapse:collapse;width:100%;margin:20px 0;font-size:13px;line-height:1.6}
th,td {border:1px solid #cfdae3;padding:9px 10px;vertical-align:top;word-break:normal;overflow-wrap:anywhere}
th {background:#eaf2f8;color:#183d59} tr:nth-child(even){background:#f7fafc}
img {max-width:100%;height:auto} figure {margin:30px 0} figcaption {font-size:13px;color:#526879}
code {font-family:'Noto Sans Mono',monospace;font-size:.9em;background:#eef3f6;padding:2px 4px;overflow-wrap:anywhere}
pre {white-space:pre-wrap;padding:15px;background:#f1f5f9} a {color:#0969a6}
#TOC {padding:22px 30px;background:#f1f6fa;border:1px solid #d4e3ee;border-radius:6px;margin-bottom:36px}
#TOC ul {list-style:none;padding-left:20px;margin:4px 0} #TOC>ul {padding-left:0}
@media print {body{max-width:none;margin:0;font-size:10pt} #TOC{break-after:page}
h2,h3{break-after:avoid} table{font-size:8pt}thead{display:table-header-group}tr{break-inside:avoid}
a{color:inherit;text-decoration:none}img{max-height:23cm} @page{size:A4;margin:18mm}}
"""
(OUT / "report.css").write_text(css, encoding="utf-8")

with tempfile.TemporaryDirectory(prefix="cvtsp_report_export_") as tmp:
    tmp = Path(tmp)
    raw = subprocess.run([PANDOC, "--print-default-data-file", "reference.docx"],
                         check=True, stdout=subprocess.PIPE).stdout
    (tmp / "default.docx").write_bytes(raw)
    transform_docx(tmp / "default.docx", tmp / "reference.docx", {"word/styles.xml":edit_styles})
    subprocess.run([
        PANDOC, str(OUT / f"{STEM}.md"), "--from=markdown+pipe_tables+tex_math_dollars",
        "--to=docx", "--toc", "--toc-depth=2", "--metadata=lang:zh-CN",
        "--metadata=toc-title:目录", f"--reference-doc={tmp/'reference.docx'}",
        f"--resource-path={OUT}", "--output", str(tmp / "report.docx"),
    ], check=True)
    transform_docx(tmp / "report.docx", OUT / f"{STEM}.docx", {"word/document.xml":edit_document})

subprocess.run([
    PANDOC, str(OUT / f"{STEM}.md"), "--from=markdown+pipe_tables+tex_math_dollars",
    "--to=html5", "--standalone", "--self-contained", "--mathml", "--toc", "--toc-depth=2",
    "--metadata=lang:zh-CN", "--metadata=toc-title:目录", "--metadata=pagetitle:CVTSP实验完整复盘",
    f"--css={OUT/'report.css'}", f"--resource-path={OUT}",
    "--output", str(OUT / f"{STEM}.html"),
], check=True)
html_path=OUT/f"{STEM}.html"
html=html_path.read_text(encoding='utf-8')
# Pandoc 2 retains an obsolete Internet Explorer fallback CDN reference even
# with --self-contained. Modern browsers render the native MathML directly.
html=re.sub(r'<!--\[if lt IE 9\]>.*?<!\[endif\]-->', '', html, flags=re.S)
html_path.write_text(html,encoding='utf-8')
print("EXPORTED", OUT / f"{STEM}.docx", OUT / f"{STEM}.html", sep="\n")
