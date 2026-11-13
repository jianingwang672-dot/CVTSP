"""Recompute the added Section 5 numbers from saved experimental records.

No model, training, solver or network calls. The only generated file is this
run's evidence.json. Source manuscripts and experiment outputs are read-only.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import statistics as st

RUN = Path(__file__).resolve().parent
PAPER = RUN.parent
ROOT = RUN.parents[3]
SOURCES = {}


def load(path):
    p = ROOT / path
    raw = p.read_bytes()
    SOURCES[path] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def metrics(path, end):
    p = ROOT / path
    SOURCES[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open() as f:
        return [r for r in csv.DictReader(f) if float(r['epoch']) <= end]


def total(rows, key):
    return sum(float(r.get(key, '') or 0) for r in rows)


def summary(a, b):
    assert len(a) == len(b) and len(a) > 1
    gaps = [100 * (x - y) / y for x, y in zip(a, b)]
    mean = st.mean(gaps)
    half = 1.96 * st.stdev(gaps) / math.sqrt(len(gaps))
    wins = sum(x < y - 1e-3 for x, y in zip(a, b))
    losses = sum(x > y + 1e-3 for x, y in zip(a, b))
    return dict(n=len(a), variant_mean=st.mean(a), control_mean=st.mean(b),
                gap=mean, ci=[mean-half, mean+half],
                ratio_of_means=100*(st.mean(a)-st.mean(b))/st.mean(b),
                wins=wins, ties=len(a)-wins-losses, losses=losses)


def check_scope_and_bilingual():
    before = RUN / 'before'
    for old_name, new_path, start, end in [
        ('main.tex', PAPER/'main.tex', r'\section{Computational experiments}',
         r'\section{Discussion and implications for practice}'),
        ('main_bilingual.tex', PAPER/'bilingual/main_bilingual.tex',
         r'\section{Computational experiments\quad 计算实验}',
         r'\section{Discussion and implications for practice\quad 讨论与实践启示}')]:
        old, new = (before/old_name).read_text(), new_path.read_text()
        assert old[:old.index(start)] == new[:new.index(start)]
        assert old[old.index(end):] == new[new.index(end):]
    ns = runpy.run_path(str(PAPER/'bilingual/build_bilingual.py'))
    source = (PAPER/'main.tex').read_text()
    bi = PAPER/'bilingual'
    zh = ns['load_zh']([str(bi/'zh_part1.txt'), str(bi/'zh_part2.txt')])
    assert ns['build'](source, zh) == (bi/'main_bilingual.tex').read_text()
    assert set(zh) == {u[0] for u in ns['units'](source)} | {'FIG'}
    section = source[source.index(start := r'\section{Computational experiments}'):
                     source.index(r'\section{Discussion and implications for practice}')]
    # Literature-reference merge needs unavailable per-instance source data;
    # those two original placeholders are intentionally retained.
    assert section.count(r'\placeholder{') == 2
    assert 'queued after Group A' not in section and 'running since' not in section
    return dict(outside_section5_unchanged=True, bilingual_exact_rebuild=True,
                remaining_placeholders=2, translation_units=len(ns['units'](source)))


def main():
    output = {'scope': check_scope_and_bilingual()}
    data = load('outputs/ablation_20261028/test500_aug8_e100/results.json')
    assert data['complete'] and '8-view' in data['protocol']
    records = data['records']
    assert len(records) == len({r['instance_id'] for r in records}) == 500
    assert len({r['instance_fingerprint'] for r in records}) == 500
    ablations = {}
    for j in [20, 40, 60, 80, 100, 'all']:
        rows = [r for r in records if j == 'all' or r['problem_size'] == j]
        vals = lambda key: [r['objectives'][key] for r in rows]
        ablations[str(j)] = {k: summary(vals(key), vals('main_e100')) for k,key in
                            [('B','exact_search_e100'),('D','no_diversity_e100')]}
    assert abs(ablations['all']['B']['gap'] - (-0.096141)) < 1e-5
    assert abs(ablations['all']['D']['gap'] - 2.137422) < 1e-5
    output['BCD_quality'] = ablations

    paths = {'B':'outputs/ablation_20261028/exact_search/train/metrics.csv',
             'C':'outputs/progressive_recipe_20261020/train/metrics.csv',
             'D':'outputs/ablation_20261028/no_diversity/train/metrics.csv'}
    budgets = {}
    for key, path in paths.items():
        rows = metrics(path, 100)
        assert len(rows) == 100
        budgets[key] = dict(logged_calls=total(rows,'total_gurobi_calls_all'),
                            refresh=total(rows,'c0_refresh_gurobi_calls'),
                            targets=total(rows,'target_reward_gurobi_calls'),
                            validation=total(rows,'validation_gurobi_calls'),
                            training_hours=total(rows,'epoch_runtime_seconds')/3600)
    progress_path = ROOT/'outputs/ablation_20261028/exact_search/train/exact_progress.jsonl'
    search_calls = generation_calls = count = 0
    digest = hashlib.sha256()
    with progress_path.open('rb') as f:
        for raw in f:
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            assert row['search_calls'] == 100
            assert row['termination'] == 'exact_budget_exhausted'
            search_calls += row['search_calls']
            generation_calls += row['total_calls']
            count += 1
    SOURCES[str(progress_path.relative_to(ROOT))] = digest.hexdigest()
    assert count == 1_000_000 and search_calls == 100_000_000
    budgets['B']['total_calls'] = budgets['B']['logged_calls'] + search_calls
    assert budgets['B']['total_calls'] == generation_calls + budgets['B']['targets'] == 102_762_944
    for key, expected in [('C',2_643_875),('D',2_111_419)]:
        budgets[key]['total_calls'] = budgets[key]['logged_calls']
        assert budgets[key]['total_calls'] == expected
    budgets['B_over_C_calls'] = budgets['B']['total_calls']/budgets['C']['total_calls']
    budgets['B_over_C_training_time'] = budgets['B']['training_hours']/budgets['C']['training_hours']
    output['BCD_budget'] = budgets

    a = load('outputs/pomo_gurobi_ablationA_20261030/validation2000/results.json')
    reference = load('outputs/pomo_gurobi_equal_budget_20261027/eval_equal_solves_val2000/results.json')
    assert a['complete'] and reference['complete']
    ref = {r['instance_id']:r for r in reference['records']}
    for r in a['records']:
        assert r['instance_fingerprint'] == ref[r['instance_id']]['instance_fingerprint']
    a_result = {}
    for label in ['pomoA_equal_solves','pomoA_equal_instances','pomoA_time024h']:
        a_result[label] = summary([r['objectives'][label] for r in a['records']],
                                 [ref[r['instance_id']]['objectives']['new_e133'] for r in a['records']])
    for j in [20,40,60,80,100]:
        rows = [r for r in a['records'] if r['problem_size'] == j]
        a_result[f'24h_J{j}'] = summary([r['objectives']['pomoA_time024h'] for r in rows],
                                      [ref[r['instance_id']]['objectives']['new_e133'] for r in rows])
    assert a_result['pomoA_time024h']['gap'] < 0
    assert a_result['pomoA_time024h']['ratio_of_means'] > 0
    a_result['budget_checkpoints'] = load('outputs/pomo_gurobi_ablationA_20261030/train/budget_checkpoints.json')
    a_result['24h_checkpoint'] = load('outputs/pomo_gurobi_ablationA_20261030/train/time_checkpoints.json')[-1]
    output['A_validation'] = a_result

    rows = metrics('outputs/progressive_recipe_20261020/train/metrics.csv',133)
    solver = total(rows,'gurobi_runtime_seconds')/3600
    initial = total(rows,'local_search_runtime_seconds')/3600
    refresh = total(rows,'c0_refresh_runtime_seconds')/3600
    calls = total(rows,'total_gurobi_calls_all')
    assert calls == 3_839_970 and abs(solver-59.8834363597)<1e-8
    output['E133_timing'] = dict(solver_hours=solver,initial_search_hours=initial,
        refresh_mixed_hours=refresh, seconds_per_call=solver*3600/calls,
        combined_hours_bounds=[solver+initial,solver+initial+refresh],
        refresh_solver_included_in_solver_hours=True,process_cpu_time_measured=False)

    gurobi = {}
    gurobi_quality = {}
    for dataset in ['real529','test500']:
        folder = ROOT/f'outputs/full_gurobi_20261030/{dataset}_3600s'
        rows = [load(str(p.relative_to(ROOT))) for p in sorted(folder.glob('*.json'))]
        assert len(rows) == (529 if dataset == 'real529' else 500)
        evaluation_path = ('outputs/progressive_recipe_20261020/final_external_e133/real529/results.json'
                           if dataset == 'real529' else
                           'outputs/pomo_gurobi_equal_budget_20261027/eval_equal_solves_test500_aug8/results.json')
        evaluation = load(evaluation_path)
        assert evaluation['complete']
        policy_key = 'new_e133' if dataset == 'real529' else 'e133'
        policy = {r['instance_id']:r['objectives'][policy_key] for r in evaluation['records']}
        assert set(policy) == {r['instance_id'] for r in rows}
        matched = [r for r in rows if r['solution_count']>0 and r['objective'] is not None
                   and math.isfinite(r['objective'])]
        optimal = [r for r in rows if r['optimal']]
        gurobi_quality[dataset] = {
            'incumbent_subset':summary([policy[r['instance_id']] for r in matched],
                                      [r['objective'] for r in matched]),
            'optimal_subset':summary([policy[r['instance_id']] for r in optimal],
                                    [r['objective'] for r in optimal])}
        for r in rows:
            assert r['config']['threads'] == 1 and r['config']['time_limit'] == 3600
            assert r['config']['mip_gap'] == 1e-6
        groups = ([("10-19",lambda j:10<=j<=19),("20-50",lambda j:20<=j<=50),
                   ("100",lambda j:j==100),("200",lambda j:j==200)] if dataset=='real529' else
                  [(str(j),lambda size,j=j:size==j) for j in [20,40,60,80,100]])
        groups.append(('all',lambda j:True))
        gurobi[dataset] = {}
        for name, predicate in groups:
            group=[r for r in rows if predicate(r['problem_size'])]
            finite=[r for r in group if r['solution_count']>0 and r['objective'] is not None and math.isfinite(r['objective'])]
            gurobi[dataset][name] = dict(n=len(group),optimal=sum(r['optimal'] for r in group),
                time_limit=sum(r['status']=='TIME_LIMIT' for r in group),incumbents=len(finite),
                UB=st.mean(r['objective'] for r in finite) if finite else None,
                LB=st.mean(r['best_bound'] for r in group),
                gap=100*st.mean(r['mip_gap'] for r in finite) if finite else None,
                seconds=st.mean(r['solve_time'] for r in group))
    assert gurobi['real529']['all']['optimal']==136 and gurobi['real529']['all']['incumbents']==454
    assert gurobi['test500']['all']['optimal']==6 and gurobi['test500']['all']['incumbents']==451
    output['full_gurobi'] = gurobi
    output['full_gurobi_vs_E133'] = gurobi_quality
    output['sources_sha256'] = SOURCES
    (RUN/'evidence.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print('PASS: Section 5 scope, bilingual equality, A/B/C/D statistics and budgets, E133 timings, full Gurobi results')
    print('B/C calls:',budgets['B_over_C_calls'],'B/C training time:',budgets['B_over_C_training_time'])
    print('B overall:',ablations['all']['B'])
    print('D overall:',ablations['all']['D'])
    print('A24h:',a_result['pomoA_time024h'])
    print('Evidence:',RUN/'evidence.json')


if __name__ == '__main__':
    main()
