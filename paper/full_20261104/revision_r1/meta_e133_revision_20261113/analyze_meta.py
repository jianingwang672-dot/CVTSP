"""Recompute saved Real529 metaheuristic comparisons against augmented E133.

Reads saved files only. No model, solver, network, or new experiment calls.
"""
from pathlib import Path
import hashlib
import json
import math
import statistics as st

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[3]
HASHES = {}
GROUPS = ['Gam72', 'Erd132 16-20', 'Erd132 25-50',
          'Poi325 Regular', 'Poi325 Cluster', 'All']


def read(relative):
    raw = (ROOT / relative).read_bytes()
    HASHES[relative] = hashlib.sha256(raw).hexdigest()
    return raw.decode()


def summary(costs, baseline, ids):
    gaps = [100*(costs[i]-baseline[i])/baseline[i] for i in ids]
    mean = st.mean(gaps)
    half = 1.96*st.stdev(gaps)/math.sqrt(len(ids))
    wins = sum(costs[i] < baseline[i]-1e-3 for i in ids)
    losses = sum(costs[i] > baseline[i]+1e-3 for i in ids)
    return dict(n=len(ids),cost=st.mean(costs[i] for i in ids),
                gap=mean,ci=[mean-half,mean+half],
                wins=wins,ties=len(ids)-wins-losses,losses=losses)


def main():
    for relative in ['src/Metaheuristics.py', 'scripts/run_metaheuristics_real529.py',
                     'scripts/after_pomo_budget.sh',
                     'outputs/pomo_gurobi_equal_budget_20261027/after.log']:
        read(relative)
    timing = json.loads(read('outputs/bks_comparison_20261023/timing/new_e133.json'))
    assert '8-view' in timing['protocol']
    model = {r['instance_id']:r for r in timing['records']}
    assert len(model)==len(timing['records'])==529
    baseline = {i:r['objective'] for i,r in model.items()}
    primary = json.loads(read('outputs/progressive_recipe_20261020/final_external_e133/real529/results.json'))
    assert primary['complete'] and primary['params']['augmentation'] is True
    assert primary['params']['aug_factor']==8
    assert {r['instance_id'] for r in primary['records']}==set(model)
    for r in primary['records']:
        assert abs(r['objectives']['new_e133']-baseline[r['instance_id']])<1e-7

    by_group = {g:[] for g in GROUPS}
    metadata = []
    for i,r in model.items():
        first = (ROOT/'instance/Data'/f'{i}.txt').read_text().splitlines()[0]
        metadata.append((i,r['problem_size'],first))
        if 'Clustered' in first:
            group = 'Poi325 Cluster'
        elif 'Regular' in first:
            group = 'Poi325 Regular'
        elif r['problem_size']<=15:
            group = 'Gam72'
        elif r['problem_size']<=20:
            group = 'Erd132 16-20'
        else:
            group = 'Erd132 25-50'
        by_group[group].append(i)
        by_group['All'].append(i)
    assert [len(by_group[g]) for g in GROUPS]==[72,60,72,175,150,529]

    original = 'outputs/metaheuristics_real529_20261026'
    directories = [original, 'outputs/metaheuristics_real529_sa_1x_20261027',
                   'outputs/metaheuristics_real529_sa_3x_20261027']
    runs = {}
    for directory in directories:
        params = json.loads(read(directory+'/params.json'))
        assert params['seed']==0 and params['granular_k']==8
        assert params['solver']=='Gurobi fixed-sequence SOCP, 1 thread'
        methods = ['sa','ils','vns'] if directory==original else ['sa']
        for method in methods:
            rows = [json.loads(line) for line in read(directory+f'/{method}.jsonl').splitlines() if line.strip()]
            assert len(rows)==len({r['instance_id'] for r in rows})==529
            assert {r['instance_id'] for r in rows}==set(model)
            for r in rows:
                assert r['method']==method and r['seed']==0
                assert r['problem_size']==model[r['instance_id']]['problem_size']
                unit = model[r['instance_id']]['decode_seconds']+model[r['instance_id']]['gurobi_cpu_seconds']
                assert math.isclose(r['time_budget'],params['factor']*unit,rel_tol=1e-9)
                assert abs(min(t[2] for t in r['trace'])-r['best_makespan'])<1e-7
            runs[(method,int(params['factor']))] = rows

    def within(rows, factor, total):
        costs = {}
        for r in rows:
            cutoff = r['time_budget']*factor/total
            values = [v for _,seconds,v in r['trace'] if seconds<=cutoff]
            assert values and all(math.isfinite(v) for v in values)
            costs[r['instance_id']] = min(values)
        return costs

    results = {}
    latex = ['E133 & '+' & '.join(['0.00']*len(GROUPS))+r' \\']
    for factor in [1,3,10]:
        for method in ['ils','vns','sa']:
            source_factor = factor if method=='sa' else 10
            costs = within(runs[(method,source_factor)],factor,source_factor)
            stats = {g:summary(costs,baseline,by_group[g]) for g in GROUPS}
            sizes = {str(j):summary(costs,baseline,[i for i,r in model.items() if r['problem_size']==j])
                     for j in [100,200]}
            key = f'{method}_{factor}x'
            results[key] = dict(groups=stats,by_size=sizes,
                                source_factor=source_factor,
                                budget_specific_sa=method=='sa')
            cells = [f"{stats[g]['gap']:+.2f}".replace('-','$-$') for g in GROUPS[:-1]]
            all_ = stats['All']
            cells.append((f"{all_['gap']:+.2f} [{all_['ci'][0]:+.2f}, {all_['ci'][1]:+.2f}]").replace('-','$-$'))
            latex.append(f'{method.upper()}, ${factor}\\times$ & '+' & '.join(cells)+r' \\')
            print(key, f"gap={all_['gap']:+.6f}", f"CI={all_['ci']}",
                  f"WTL={all_['wins']}/{all_['ties']}/{all_['losses']}")

    audit = {}
    for factor in [1,3,10]:
        rows = runs[('sa',factor)]
        costs = within(rows,factor,factor)
        initial = {r['instance_id']:r['initial_makespan'] for r in rows}
        audit[str(factor)] = dict(
            instances=529,mean_evaluations=st.mean(r['evaluations'] for r in rows),
            calls_at_most_21=sum(r['evaluations']+r['cache_hits']<=21 for r in rows),
            not_improved_initial=sum(abs(costs[i]-initial[i])<=1e-9 for i in model),
            raw_final_not_improved_initial=sum(abs(r['best_makespan']-r['initial_makespan'])<=1e-9 for r in rows),
            at_budget=summary(costs,baseline,by_group['All']))
        if factor!=10:
            old = within(runs[('sa',10)],factor,10)
            audit[str(factor)]['old_10x_prefix'] = summary(old,baseline,by_group['All'])
            audit[str(factor)]['standalone_vs_old_prefix'] = summary(costs,old,by_group['All'])
            audit[str(factor)]['old_prefix_temperature_ratio'] = 1e-3**(factor/10)
        for r in rows:
            original_row = next(q for q in runs[('sa',10)] if q['instance_id']==r['instance_id'])
            assert r['initial_makespan']==original_row['initial_makespan']

    other = json.loads(read('outputs/bks_comparison_20261023/timing/epoch200.json'))
    other_costs = {r['instance_id']:r['objective'] for r in other['records']}
    assert set(other_costs)==set(baseline)
    (RUN/'generated_rows.tex').write_text('\n'.join(latex)+'\n')
    out = dict(metric='mean_i 100*(method_i-E133_i)/E133_i',
               baseline='eight-view E133, exact fixed-order SOCP costs',
               cutoff='best completed exact evaluation at or before each time budget',
               win_tie_loss_absolute_tolerance=1e-3,
               groups={g:len(v) for g,v in by_group.items()},comparisons=results,
               g200_vs_e133=summary(other_costs,baseline,by_group['All']),
               mean_e133_single_core_seconds=st.mean(r['decode_seconds']+r['gurobi_cpu_seconds'] for r in model.values()),
               sa_audit=audit,source_sha256=HASHES,
               code_provenance='Current implementation and saved runner comments; historical source commit hashes were not recorded.',
               instance_group_metadata_sha256=hashlib.sha256(json.dumps(sorted(metadata)).encode()).hexdigest(),
               no_new_experiments=True)
    (RUN/'evidence.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':
    main()
