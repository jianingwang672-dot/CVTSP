"""Verify the bounded manuscript change and bilingual rebuild, offline."""
from pathlib import Path
import hashlib
import json
import re
import runpy

RUN = Path(__file__).resolve().parent
PAPER = RUN.parent


def para(source, start):
    block = source[source.index(start):]
    return block.split('\n\n',1)[0]


def table(source, label):
    at = source.index(r'\label{'+label+'}')
    begin = source.rfind(r'\begin{table}',0,at)
    end = source.index(r'\end{table}',at)+len(r'\end{table}')
    return source[begin:end]


def redact(source):
    related = para(source,'Recent work has moved')
    source = source.replace(related,'<RELATED-CONTEXT>')
    begin = source.index(r'Table~\ref{tab:meta} compares')
    end = source.index(r'\subsection{Results on synthetic instances}',begin)
    source = source[:begin]+'<META-COMPARISON>\n\n'+source[end:]
    begin = source.index(r'Table~\ref{tab:breakeven} sets')
    end = source.index(r'\subsection{What the loose bound does and does not show}',begin)
    source = source[:begin]+'<RELATED-DISCUSSION>\n\n'+source[end:]
    conclusion = para(source,'This paper asked whether')
    return source.replace(conclusion,'<CONCLUSION-SENTENCE>')


def main():
    old = (RUN/'before/main.tex').read_text()
    new = (PAPER/'main.tex').read_text()
    assert redact(old)==redact(new), 'Out-of-scope source change'
    abstract = r'\\begin\{abstract\}.*?\\end\{abstract\}'
    assert re.search(abstract,old,re.S).group()==re.search(abstract,new,re.S).group()
    assert para(old,'Recent work has moved').replace(
        ', and their results supply the reference values used in this paper','')==para(new,'Recent work has moved')
    assert para(old,'This paper asked whether').replace(
        ', and the strongest published heuristic, obtain better solutions,',
        ' obtain better solutions than E133,')==para(new,'This paper asked whether')
    assert set(re.findall(r'\\cite\w*\{([^}]+)\}',old))==set(re.findall(r'\\cite\w*\{([^}]+)\}',new))
    for label in ['tab:real','tab:test500','tab:gen','tab:ablationBCD']:
        assert table(old,label)==table(new,label)
    for removed in ['Merge the best values of Vieira','After the Vieira merge',
                    'poorly calibrated','4.53\\%','4.52\\%',
                    '154.3 core-hours','251 against 180']:
        assert removed not in new, removed
    assert new.count('original unaugmented validation protocol')==1

    ns = runpy.run_path(str(PAPER/'bilingual/build_bilingual.py'))
    bi = PAPER/'bilingual'
    zh = ns['load_zh']([str(bi/'zh_part1.txt'),str(bi/'zh_part2.txt')])
    assert ns['build'](new,zh)==(bi/'main_bilingual.tex').read_text()
    assert set(zh)=={u[0] for u in ns['units'](new)}|{'FIG'}
    assert len(ns['units'](new))==142
    old_units = {(u[1],u[2],u[5]):u[0] for u in ns['units'](old)}
    old_zh = ns['load_zh']([str(RUN/'before/zh_part1.txt'),str(RUN/'before/zh_part2.txt')])
    changed = []
    for uid,cls,kind,start,end,text in ns['units'](new):
        key = cls,kind,text
        if key in old_units:
            assert zh[uid]==old_zh[old_units[key]]
        else:
            changed.append(uid)
    assert changed==['U019','U090','U091','U092','U093','U094','U126','U127','U128','U135']

    evidence = json.loads((RUN/'evidence.json').read_text())
    for rel,digest in evidence['source_sha256'].items():
        assert hashlib.sha256((RUN.parents[3]/rel).read_bytes()).hexdigest()==digest
    generated = (RUN/'generated_rows.tex').read_text().strip()
    assert generated in table(new,'tab:meta')
    assert 'G200 &' not in table(new,'tab:meta')
    assert 'E133 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00' in table(new,'tab:meta')
    for key,expected in [('ils_1x',1.77),('ils_3x',.35),('ils_10x',-.60),
                         ('vns_10x',-.55),('sa_1x',3.41),('sa_3x',3.22),('sa_10x',2.01)]:
        assert round(evidence['comparisons'][key]['groups']['All']['gap'],2)==expected
    assert round(evidence['g200_vs_e133']['gap'],2)==-.20
    assert all(r['calls_at_most_21']==0 for r in evidence['sa_audit'].values())
    result = dict(scope_check=True,abstract_regenerated=False,bilingual_exact_rebuild=True,
                  changed_translation_units=changed,unchanged_translations=132,
                  untouched_primary_policy_tables=True,table_matches_raw_results=True,
                  source_sha256=hashlib.sha256((PAPER/'main.tex').read_bytes()).hexdigest(),
                  overleaf_status='pending_not_uploaded')
    (RUN/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: bounded edits, original abstract and primary tables preserved; E133-relative data and bilingual rebuild match.')


if __name__=='__main__':
    main()
