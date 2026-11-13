"""Read-only source and saved-result checks for augmented-only reporting.

No training, solver, external-model, or network calls. Writes only evidence.json.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import runpy
import statistics as st

RUN = Path(__file__).resolve().parent
PAPER = RUN.parent
ROOT = RUN.parents[3]
SOURCES = {}


def load(relative):
    p = ROOT / relative
    raw = p.read_bytes()
    SOURCES[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def summarize(records, ours, baseline):
    a = [r['objectives'][ours] for r in records]
    b = [r['objectives'][baseline] for r in records]
    gaps = [100*(x-y)/y for x,y in zip(a,b)]
    mean = st.mean(gaps)
    half = 1.96*st.stdev(gaps)/math.sqrt(len(gaps))
    w = sum(x<y-1e-3 for x,y in zip(a,b))
    l = sum(x>y+1e-3 for x,y in zip(a,b))
    return dict(n=len(a),ours=st.mean(a),baseline=st.mean(b),gap=mean,
                ci=[mean-half,mean+half],wins=w,ties=len(a)-w-l,losses=l)


def table_body(source, label):
    body = source[source.index(r'\label{'+label+'}'):]
    return body[:body.index(r'\end{table}')].replace('$-$','-')


def cost_row(size, result):
    return (f"{size} & {result['ours']:.3f} & {result['baseline']:.3f} & "
            f"{result['gap']:+.2f} & [{result['ci'][0]:+.2f}, {result['ci'][1]:+.2f}] & "
            f"{result['wins']}/{result['ties']}/{result['losses']}")


def main():
    old = (RUN/'before/main.tex').read_text()
    new = (PAPER/'main.tex').read_text()
    fixed = r'\subsection{Progressive search}'
    assert old[:old.index(fixed)] == new[:new.index(fixed)]
    real = r'\subsection{Results on the literature benchmark}'
    synthetic = r'\subsection{Results on synthetic instances}'
    general = r'\subsection{Generalisation to larger instances}'
    exact = r'\subsection{Comparison with a general-purpose exact solver}'
    assert new.index(real) < new.index(synthetic) < new.index(general) < new.index(exact)
    old_real = old[old.index(real):old.index(exact)]
    old_real = old_real.replace('[$-$1.78, $-$0.72]', '[$-$1.78, $-$0.71]')
    assert old_real == new[new.index(real):new.index(synthetic)]
    assert re.findall(r'\\cite\w*\{[^}]+\}',old) == re.findall(r'\\cite\w*\{[^}]+\}',new)
    assert new.count('unaugmented') == 1
    assert 'original unaugmented validation protocol' in new
    for stale in ['Without augmentation','without augmentation','no augmentation',
                  'fig:efficiency','tab:ablationA','data_efficiency.pdf',
                  '17.703','17.705','17.772','18.488','17.755','17.309',
                  '4.44\\%','0.076','0.18\\%','3.4 times as many solves']:
        assert stale not in new, stale
    ns = runpy.run_path(str(PAPER/'bilingual/build_bilingual.py'))
    bi = PAPER/'bilingual'
    zh = ns['load_zh']([str(bi/'zh_part1.txt'),str(bi/'zh_part2.txt')])
    assert set(zh) == {u[0] for u in ns['units'](new)} | {'FIG'}
    assert ns['build'](new,zh) == (bi/'main_bilingual.tex').read_text()
    assert (bi/'main_bilingual.tex').read_text().count('无增强') == 1

    test = load('outputs/pomo_gurobi_equal_budget_20261027/eval_equal_solves_test500_aug8/results.json')
    assert test['complete'] and '8-view' in test['protocol']
    rows = test['records']
    assert len(rows)==len({r['instance_id'] for r in rows})==500
    results = {}
    table = new[new.index(r'\label{tab:test500}'):]
    table = table[:table.index(r'\end{table}')]
    for size in [20,40,60,80,100,'All']:
        group = [r for r in rows if size=='All' or r['problem_size']==size]
        result = summarize(group,'e133','g200')
        results[str(size)] = result
        expected=(f"{size} & {result['ours']:.3f} & {result['baseline']:.3f} & "
                  f"{result['gap']:+.2f} & [{result['ci'][0]:+.2f}, {result['ci'][1]:+.2f}] & "
                  f"{result['wins']}/{result['ties']}/{result['losses']} \\")
        assert expected in table, expected
    equal_test = summarize(rows,'pomo_equal_solves','e133')
    assert round(equal_test['gap'],2)==3.27
    assert (equal_test['wins'],equal_test['ties'],equal_test['losses'])==(94,37,369)
    real_data = load('outputs/pomo_gurobi_equal_budget_20261027/eval_equal_solves_real529/results.json')
    assert real_data['complete'] and real_data['params']['augmentation'] is True
    equal_real = summarize(real_data['records'],'pomo_equal_solves','new_e133')
    assert round(equal_real['gap'],2)==3.79
    assert (equal_real['wins'],equal_real['ties'],equal_real['losses'])==(65,111,353)
    equal_real_by_size = {}
    for size, expected in [(100, 11.7), (200, 24.0)]:
        group = [r for r in real_data['records'] if r['problem_size']==size]
        result = summarize(group,'pomo_equal_solves','new_e133')
        assert round(result['gap'],1)==expected
        equal_real_by_size[str(size)] = result
    real_primary = load('outputs/progressive_recipe_20261020/final_external_e133/real529/results.json')
    assert real_primary['complete'] and real_primary['params']['augmentation'] is True
    assert real_primary['params']['aug_factor']==8
    real_results = {}
    for lo,hi in [(10,19),(20,50),(100,100),(200,200),(0,999)]:
        group = [r for r in real_primary['records'] if lo<=r['problem_size']<=hi]
        result = summarize(group,'new_e133','epoch200')
        assert cost_row(result['n'],result) in table_body(new,'tab:real')
        real_results[f'{lo}-{hi}'] = result
    gen = load('outputs/generalization_20261029/eval_aug8/results.json')
    assert gen['complete'] and '8-view' in gen['protocol']
    assert len(gen['records'])==len({r['instance_id'] for r in gen['records']})==400
    gen_results = {}
    for size in [150,200,300,500,'All']:
        group = [r for r in gen['records'] if size=='All' or r['problem_size']==size]
        result = summarize(group,'e133','g200')
        assert cost_row(size,result) in table_body(new,'tab:gen')
        gen_results[str(size)] = result
    bcd = load('outputs/ablation_20261028/test500_aug8_e100/results.json')
    assert bcd['complete'] and '8-view' in bcd['protocol']
    assert {r['instance_fingerprint'] for r in bcd['records']} == {r['instance_fingerprint'] for r in rows}
    bcd_results = {}
    for size in [20,40,60,80,100,'All']:
        group = [r for r in bcd['records'] if size=='All' or r['problem_size']==size]
        b = summarize(group,'exact_search_e100','main_e100')
        d = summarize(group,'no_diversity_e100','main_e100')
        expected = (f"{size} & {b['baseline']:.3f} & {b['ours']:.3f} & {b['gap']:+.2f} & "
                    f"[{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}] & {d['ours']:.3f} & {d['gap']:+.2f} & "
                    f"[{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]")
        assert expected in table_body(new,'tab:ablationBCD')
        bcd_results[str(size)] = {'B_vs_C':b,'D_vs_C':d}
    assert (b['wins'],b['ties'],b['losses'])==(251,62,187)
    assert (d['wins'],d['ties'],d['losses'])==(61,44,395)
    a = load('outputs/pomo_gurobi_ablationA_20261030/validation2000/results.json')
    assert a['params']['augmentation'] is False
    assert 'No eight-view evaluation of its checkpoints is available' in new
    result = dict(section_order=['Setup','Real529','Test500','Generalisation','Full Gurobi','Training efficiency','Ablation'],
                  real529_block_preserved_except_ci_rounding=True,abstract_and_theory_unchanged=True,
                  bilingual_exact_rebuild=True,translation_units=len(ns['units'](new)),
                  test500=results,equal_budget_test500=equal_test,equal_budget_real529=equal_real,
                  equal_budget_real529_by_size=equal_real_by_size,
                  real529=real_results,generalization=gen_results,ablations=bcd_results,
                  A_aug8_quality_available=False,source_sha256=SOURCES,
                  manuscript_sha256=hashlib.sha256((PAPER/'main.tex').read_bytes()).hexdigest())
    (RUN/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS: Real529 first; all aug8 policy tables and equal-budget comparisons match raw data; bilingual equality')
    print('Test500 overall:',results['All'])
    print('Equal-budget Real529:',equal_real)
    print('Equal-budget Test500:',equal_test)


if __name__=='__main__':
    main()
