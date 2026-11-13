"""Split the revised LaTeX draft into translatable units, or build the bilingual file.
usage: units.py list <main.tex>            -> prints id|type|excerpt
       units.py build <main.tex> <zh.json> <out.tex>
"""
import re, sys, json

def match_brace(s, i):
    # s[i] == '{' ; return index of matching '}'
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == '\\':
            continue
        if c == '{' and (j == 0 or s[j-1] != '\\'):
            depth += 1
        elif c == '}' and (j == 0 or s[j-1] != '\\'):
            depth -= 1
            if depth == 0:
                return j
    raise ValueError('unbalanced')

def units(tex):
    """Yield (id, kind, start, end, text) with positions in tex where a translation goes AFTER end."""
    out = []
    b = tex.index('\\begin{document}'); e = tex.index('\\bibliographystyle')
    # headings
    for m in re.finditer(r'\\(section\*?|subsection|paragraph)\{', tex[b:e]):
        s = b + m.start(); o = b + m.end() - 1; c = match_brace(tex, o)
        out.append(['H', m.group(1), o + 1, c, tex[o+1:c]])
    # title
    m = re.search(r'\\title\{', tex); o = m.end() - 1; c = match_brace(tex, o)
    out.append(['T', 'title', o + 1, c, tex[o+1:c]])
    # captions
    for m in re.finditer(r'\\caption\{', tex[b:e]):
        o = b + m.end() - 1; c = match_brace(tex, o)
        out.append(['C', 'caption', o + 1, c, tex[o+1:c]])
    # environments translated as a whole (insert after \end)
    for env in ['abstract', 'proposition', 'proof', 'remark']:
        for m in re.finditer(r'\\begin\{%s\}(.*?)\\end\{%s\}' % (env, env), tex, re.S):
            out.append(['E', env, m.start(), m.end(), m.group(1)])
    # items (all enumerate/itemize)
    for m in re.finditer(r'\\item (.*?)(?=\\item |\\end\{(?:enumerate|itemize)\})', tex[b:e], re.S):
        s = b + m.start(1); en = b + m.end(1)
        txt = m.group(1).rstrip()
        out.append(['I', 'item', s, s + len(txt), txt])
    # prose paragraphs: blank-line separated blocks in body not starting with \ commands of structure
    body_start = tex.index('\\end{frontmatter}') + len('\\end{frontmatter}')
    pos = body_start
    for blk in re.split(r'\n\s*\n', tex[body_start:e]):
        s = tex.index(blk, pos); pos = s + len(blk)
        t = blk.strip()
        if not t: continue
        first = t.split('\n')[0]
        if re.match(r'\\(section|subsection|begin\{(table|figure|enumerate|itemize|proposition|proof|remark)\}|input|label|end\{|item|centering|hrule|noindent|setcounter|small|bibliography|caption|toprule|\\midrule)', first):
            continue
        if first.startswith('%') or first.startswith('\\begin{tabular}'):
            continue
        # paragraph block may start with \label{...} line after a heading; skip pure label lines
        if re.fullmatch(r'\\label\{[^}]*\}', t): continue
        st = s + (len(blk) - len(blk.lstrip()))
        out.append(['P', 'para', st, s + len(blk.rstrip()), t])
    out.sort(key=lambda u: (u[2], u[0]))
    res = []
    for k, u in enumerate(out, 1):
        res.append(['U%03d' % k] + u)
    return res

if __name__ == '__main__':
    mode = sys.argv[1]; tex = open(sys.argv[2]).read()
    us = units(tex)
    if mode == 'list':
        for uid, cls, kind, s, en, txt in us:
            ex = re.sub(r'\s+', ' ', txt)[:90]
            print(f'{uid}|{cls}|{kind}|{len(txt)}|{ex}')
    elif mode == 'dump':
        for uid, cls, kind, s, en, txt in us:
            print(f'### {uid} {kind}\n{txt.strip()}\n')

def load_zh(paths):
    zh = {}
    for p in paths:
        txt = open(p).read()
        for m in re.finditer(r'^### (\S+)\n(.*?)(?=^### |\Z)', txt, re.S | re.M):
            zh[m.group(1)] = m.group(2).strip()
    return zh

PREAMBLE = r"""\usepackage{xeCJK}
\setCJKmainfont[AutoFakeBold=2.5,AutoFakeSlant=0.2]{Noto Serif CJK SC}
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans Mono CJK SC}
\definecolor{zhc}{RGB}{25,65,145}
\newcommand{\zh}[1]{\par\vspace{2pt}{\color{zhc}\small #1\par}\vspace{6pt}}
\newcommand{\zhitem}[1]{\par{\color{zhc}#1}}
\DeclareRobustCommand{\zhcap}[1]{{\color{zhc}#1}}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.6}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{5}
"""

def build(tex, zh):
    us = units(tex)
    missing = [u[0] for u in us if u[0] not in zh]
    if missing:
        raise SystemExit('missing translations: ' + ' '.join(missing))
    ins = []  # (pos, text)
    for uid, cls, kind, s, e, txt in us:
        z = zh[uid]
        if cls == 'T':
            ins.append((e, r'\\[6pt]{\large ' + z + '}'))
        elif cls == 'H':
            sep = '' if kind == 'paragraph' else r'\quad '
            ins.append((e, sep + z))
        elif cls == 'C':
            ins.append((e, r'\protect\newline\zhcap{' + z + '}'))
        elif cls == 'E':
            if kind == 'abstract':
                endpos = e - len('\\end{abstract}')
                ins.append((endpos, '\n\\zh{' + z + '}\n'))
            else:
                ins.append((e, '\n\n\\zh{' + z + '}\n'))
        elif cls == 'I':
            ins.append((e, ' \\zhitem{' + z + '}'))
        elif cls == 'P':
            parts = z.split('||')
            lm = re.search(r'\\begin\{(enumerate|itemize)\}', txt)
            if lm:
                ins.append((s + lm.start(), '\\zh{' + parts[0].strip() + '}\n'))
                if len(parts) > 1 and parts[1].strip():
                    ins.append((e, '\n\n\\zh{' + parts[1].strip() + '}'))
            else:
                ins.append((e, '\n\n\\zh{' + z + '}'))
    for pos, t in sorted(ins, key=lambda x: x[0], reverse=True):
        tex = tex[:pos] + t + tex[pos:]
    tex = tex.replace('\\begin{document}', PREAMBLE + '\n\\begin{document}', 1)
    tex = tex.replace('\\input{fig_method}', '\\input{fig_method_bi}')
    tex = tex.replace('\\begin{table}[t]', '\\begin{table}[!htbp]').replace('\\begin{figure}[t]', '\\begin{figure}[!htbp]')
    return tex

if __name__ == '__main__' and sys.argv[1] == 'build':
    zh = load_zh(sys.argv[3].split(','))
    out = build(open(sys.argv[2]).read(), zh)
    open(sys.argv[4], 'w').write(out)
    fig = open(sys.argv[5]).read()
    m = re.search(r'\\caption\{', fig); o = m.end() - 1; c = match_brace(fig, o)
    fig = fig[:c] + r'\protect\newline\zhcap{' + zh['FIG'] + '}' + fig[c:]
    fig = fig.replace('\\begin{figure}[tbp]', '\\begin{figure}[!htbp]')
    open(sys.argv[6], 'w').write(fig)
    print('built', len(out))
