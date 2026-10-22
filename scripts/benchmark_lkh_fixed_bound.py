"""Paired search-only pilot on fresh instances and frozen nm20 policy state."""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing
import os
import shutil
from pathlib import Path
import time
from types import SimpleNamespace

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('CVTSP_VECTORIZED_LOCAL_SEARCH','1')

import numpy as np
import torch

from scripts.lkh_fixed_bound import LKHConfig,prepare_problem,run_start
from scripts.summarize_search_ablation import paired_values
from src.CertifiedLocalSearch import FixedAnchorLocalSearchConfig,run_fixed_anchor_local_search
from src.CertifiedUpperBound import FixedSolutionUpperBound,is_optimal_gurobi_solution,validate_tour
from src.CVPSolver import solve
from src.DualContributions import directed_total_contribution_matrix
from src.FixedValidation import load_or_create_fixed_validation_sets
from src.JointSelfImprovement import (directed_route_upper_bound,select_quality_diverse_start_indices,
                                      _deduplicate_tours,_best_non_anchor_local_candidate)
from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer
from src.PseudoExpert import instance_from_dict,instance_to_dict
from src.SearchDiagnostics import atomic_json,fingerprint,tensor_digest
from src.TSPUtils import set_seed

PROFILES=('swap_two_opt','lkh_all_starts','lkh_matched_budget')
_WARMED_SIZES=set()


def summarize_branch(anchor,anchor_value,finals,epsilon):
    best=list(anchor)
    upper=float(anchor_value)
    for item in finals:
        if item['final_upper_bound']<upper:
            best=list(item['final_sequence'])
            upper=float(item['final_upper_bound'])
    near,near_upper,gap=_best_non_anchor_local_candidate(
        [SimpleNamespace(final_sequence=x['final_sequence'],final_upper_bound=x['final_upper_bound']) for x in finals],
        anchor,anchor_value)
    return {'best_sequence':best,'best_upper_bound':upper,
            'certified_candidate':best!=list(anchor) and upper<anchor_value-epsilon,
            'near_sequence':near,'near_upper_bound':near_upper if np.isfinite(near_upper) else None,
            'near_relative_gap':gap if np.isfinite(gap) else None,
            'distinct_final_routes':len({tuple(x['final_sequence']) for x in finals})}


def search_case(task):
    instance,candidates,anchor,solver,search_config,lkh_config,index,output,feasibility,tolerance=task
    started=time.perf_counter()
    anchor_solution=solve(instance,anchor,solver)
    if not is_optimal_gurobi_solution(anchor_solution): raise RuntimeError('Anchor Gurobi solve failed')
    majorizer=FixedSolutionUpperBound.from_solution(instance,anchor_solution,
                                                   feasibility_tolerance=feasibility,require_optimal=True)
    matrix=directed_total_contribution_matrix(majorizer)
    anchor_value=float(anchor_solution.objective)
    if abs(directed_route_upper_bound(anchor,matrix)-anchor_value)>tolerance:
        raise RuntimeError('Anchor strict upper-bound identity failed')
    candidates=_deduplicate_tours(candidates,instance.J)
    if tuple(anchor) not in {tuple(x) for x in candidates}: candidates.append(list(anchor))
    bounds=np.asarray([directed_route_upper_bound(route,matrix) for route in candidates])
    selected,quality_count,diversity_count=select_quality_diverse_start_indices(candidates,bounds,
        top_k=30,diversity_count=10,diversity_pool_size=60,diversity_max_relative_gap=.05)
    starts=[candidates[int(k)] for k in selected]
    epsilon=search_config.acceptance_epsilon(anchor_value)
    directory=Path(output)/'cases'/f'case-{index:03d}-J{instance.J}'
    directory.mkdir(parents=True,exist_ok=True)
    torch.save({'instance':instance,'anchor_solution':anchor_solution,'matrix':matrix,'candidates':candidates,
                'starts':starts,'quality_count':quality_count,'diversity_count':diversity_count},directory/'input.pt')
    warmup_started=time.perf_counter()
    if instance.J not in _WARMED_SIZES:
        run_fixed_anchor_local_search(majorizer,starts[0],search_config)
        _WARMED_SIZES.add(instance.J)
    warmup=time.perf_counter()-warmup_started
    began=time.perf_counter()
    searches=[run_fixed_anchor_local_search(majorizer,route,search_config) for route in starts]
    baseline_seconds=time.perf_counter()-began
    finals=[{'initial_sequence':x.initial_sequence,'final_sequence':x.final_sequence,
             'initial_upper_bound':x.initial_upper_bound,'final_upper_bound':x.final_upper_bound,
             'accepted_moves':x.accepted,'termination':x.termination_reason,
             'wall_seconds':x.runtime} for x in searches]
    for x in finals:
        if abs(directed_route_upper_bound(x['final_sequence'],matrix)-x['final_upper_bound'])>tolerance:
            raise RuntimeError('Original search did not match the shared C0 matrix')
    profiles={'swap_two_opt':{**summarize_branch(anchor,anchor_value,finals,epsilon),
        'search_seconds':baseline_seconds,'native_calls':0,'completed_starts':len(starts),
        'timed_out_starts':0,'searched_start_indices':list(range(len(starts))),'finals':finals}}
    export_started=time.perf_counter()
    export=prepare_problem(matrix,directory,lkh_config)
    export_seconds=time.perf_counter()-export_started
    # Full-pool arm measures one LK trial for every starting tour; it is NOT
    # equated to the original 30 accepted moves. The matched arm is primary.
    for profile in ('lkh_all_starts','lkh_matched_budget'):
        began=time.perf_counter()
        deadline=(None if profile=='lkh_all_starts' else began+max(0.,baseline_seconds-export_seconds))
        order=list(range(len(starts)))
        if deadline is not None:
            # Include diverse starts early when time does not cover all30.
            order=[]
            for k in range(max(quality_count,diversity_count)):
                if k<quality_count: order.append(k)
                if k<diversity_count: order.append(quality_count+k)
        output_routes=[{'initial_sequence':list(route),'final_sequence':list(route),
            'initial_upper_bound':directed_route_upper_bound(route,matrix),
            'final_upper_bound':directed_route_upper_bound(route,matrix),
            'called':False,'completed':False,'timed_out':False,'wall_seconds':0.} for route in starts]
        visited=[]
        for k in order:
            if deadline is not None and time.perf_counter()>=deadline: break
            result=run_start(matrix,starts[k],directory,lkh_config,prefix=f'{profile}-{k:02d}',
                seed=20261014+index*100+k,deadline=deadline,
                absolute_epsilon=search_config.absolute_epsilon,relative_epsilon=search_config.relative_epsilon)
            output_routes[k]=result
            if result['called']: visited.append(k)
        profiles[profile]={**summarize_branch(anchor,anchor_value,output_routes,epsilon),
            'search_seconds':export_seconds+time.perf_counter()-began,
            'native_calls':sum(x['called'] for x in output_routes),
            'completed_starts':sum(x['completed'] for x in output_routes),
            'timed_out_starts':sum(x['timed_out'] for x in output_routes),
            'searched_start_indices':visited,'finals':output_routes,
            'budget_seconds':baseline_seconds if deadline is not None else None}
    record={'index':index,'fingerprint':fingerprint(instance),'J':instance.J,'anchor_sequence':list(anchor),
        'anchor_exact':anchor_value,'actual_top_k':len(starts),'quality_count':quality_count,
        'diversity_count':diversity_count,'matrix_sha256':hashlib.sha256(matrix.tobytes()).hexdigest(),
        'starts_sha256':hashlib.sha256(json.dumps(starts).encode()).hexdigest(),
        'warmup_seconds_excluded':warmup,'export_seconds':export_seconds,'quantization':export,
        'profiles':profiles,'total_case_seconds':time.perf_counter()-started}
    atomic_json(directory/'search.json',record)
    return record


def exact_candidate(task):
    instance,route,solver,upper,tolerance=task
    solution=solve(instance,list(validate_tour(route,instance.J)),solver)
    if not is_optimal_gurobi_solution(solution) or not np.isfinite(solution.objective):
        raise RuntimeError('Candidate Gurobi exact solve failed')
    if list(solution.sequence)!=list(route) or solution.objective>upper+tolerance:
        raise RuntimeError('Candidate route/strict upper-bound validation failed')
    return float(solution.objective),float(solution.solve_time)


def evaluate_gates(records,instances,executor,solver,search_config,tolerance):
    batches=defaultdict(list)
    for record,instance in zip(records,instances): batches[instance.J].append((record,instance))
    for profile in PROFILES:
        tasks,mapping=[],[]
        for size,group in sorted(batches.items()):
            eligible=[]
            for record,instance in group:
                branch=record['profiles'][profile]
                branch.update(expert_exact=record['anchor_exact'],expert_sequence=record['anchor_sequence'],
                    expert_source='anchor',expert_gurobi_calls=0,near_miss_attempted=False,near_miss_accepted=False)
                if branch['certified_candidate']:
                    mapping.append((record,branch,'certified'))
                    tasks.append((instance,branch['best_sequence'],solver,branch['best_upper_bound'],tolerance))
                    branch['expert_gurobi_calls']=1
                elif branch['near_sequence'] is not None:
                    eligible.append((record,instance,branch))
            eligible.sort(key=lambda x:x[2]['near_relative_gap'])
            selected=eligible[:math.ceil(.2*len(group))]
            for record,instance,branch in selected:
                branch['near_miss_attempted']=True
                branch['expert_gurobi_calls']=1
                mapping.append((record,branch,'near_miss'))
                tasks.append((instance,branch['near_sequence'],solver,branch['near_upper_bound'],tolerance))
        solved=list(executor.map(exact_candidate,tasks,chunksize=1))
        for (record,branch,kind),(value,seconds) in zip(mapping,solved):
            branch['exact_candidate_makespan']=value
            branch['exact_candidate_seconds']=seconds
            threshold=record['anchor_exact']-search_config.acceptance_epsilon(record['anchor_exact'])
            if value<threshold:
                branch['expert_exact']=value
                branch['expert_sequence']=branch['best_sequence'] if kind=='certified' else branch['near_sequence']
                branch['expert_source']=kind
                branch['near_miss_accepted']=kind=='near_miss'
            elif kind=='certified': raise RuntimeError('Certified candidate failed strict true improvement')
        for record in records:
            b=record['profiles'][profile]
            b['changed_expert']=b['expert_sequence']!=record['anchor_sequence']
            b['true_improvement']=record['anchor_exact']-b['expert_exact']
            b['true_improvement_percent']=100*b['true_improvement']/record['anchor_exact']
            if b['expert_gurobi_calls']>1 or b['true_improvement']< -tolerance:
                raise RuntimeError('Expert budget or fallback invariant failed')


def summary(records,metadata,output):
    tables={}
    for size in ('all',20,40,60,80,100):
        group=records if size=='all' else [r for r in records if r['J']==size]
        tables[str(size)]={}
        for profile in PROFILES:
            values=[r['profiles'][profile] for r in group]
            if not values: continue
            tables[str(size)][profile]={'count':len(group),
                'anchor_mean':float(np.mean([r['anchor_exact'] for r in group])),
                'expert_mean':float(np.mean([v['expert_exact'] for v in values])),
                'improved_count':sum(v['changed_expert'] for v in values),
                'true_improvement_percent':float(np.mean([v['true_improvement_percent'] for v in values])),
                'search_seconds_mean':float(np.mean([v['search_seconds'] for v in values])),
                'certified_count':sum(v['expert_source']=='certified' for v in values),
                'near_miss_attempts':sum(v['near_miss_attempted'] for v in values),
                'near_miss_accepted':sum(v['near_miss_accepted'] for v in values),
                'expert_calls':sum(v['expert_gurobi_calls'] for v in values),
                'completed_starts_mean':float(np.mean([v['completed_starts'] for v in values])),
                'timed_out_starts':sum(v['timed_out_starts'] for v in values),
                'distinct_finals_mean':float(np.mean([v['distinct_final_routes'] for v in values]))}
    paired={profile:paired_values([r['profiles']['swap_two_opt']['expert_exact'] for r in records],
                [r['profiles'][profile]['expert_exact'] for r in records]) for profile in PROFILES[1:]}
    result={'complete':True,'metadata':metadata,'by_J':tables,'paired_expert_makespan':paired,'records':records,
        'limitations':['Frozen nm20 Reference/Predictor; search-only pilot, not a cold-start training experiment.',
            'LKH trials differ from accepted swap/2-opt moves; all-starts arm is not equal-cost.',
            'Matched-budget arm considers the same30 starts but may only run a subset; fallback starts are retained.',
            'Process startup/kill can overshoot a short deadline; all observed wall costs are reported unclamped.',
            'This tests the subprocess adapter for LKH-3.0.6, not every possible LKH implementation.',
            'Near-miss quota20% is evaluated on each same-J pilot batch, not training batch64 ceil rounding.',
            'Gurobi solves only common Anchor and at most one additional route per profile/instance; no mid-search solves.']}
    atomic_json(output/'comparison.json',result)
    lines=['# LKH与swap＋2-opt：小规模同输入搜索对照','',
        '同一批新实例、同一Anchor精确解/C0、同一Top-K起点；冻结nm20 Epoch20策略生成候选，无训练更新。',
        'matched为近似同墙钟预算，all_starts为每条起点一次LK trial；trials不是接受move数。','',
        '| J | 方法 | Expert makespan | 改进Expert | 平均真实改善 | 搜索秒/例 | 完成搜索起点/例 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for size,values in tables.items():
        for profile,v in values.items():
            lines.append(f"| {size} | {profile} | {v['expert_mean']:.6f} | {v['improved_count']}/{v['count']} | {v['true_improvement_percent']:.3f}% | {v['search_seconds_mean']:.4f} | {v['completed_starts_mean']:.2f} |")
    lines+=['','配对差值为LKH−swap＋2-opt，负值更好：']
    for profile,v in paired.items():lines.append(f"- {profile}: {v['mean_right_minus_left']:.6f}，95%区间 {v['ci95']}；胜/平/负={v['right_wins']}/{v['ties']}/{v['right_losses']}。")
    lines+=['','局限：',*(f'- {x}' for x in result['limitations'])]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(output/'completion_audit.json',{'passed':True,'instances':len(records),
        'same_C0_and_starts':True,'same_exact_Anchor':True,'float_bounds_revalidated':True,
        'one_candidate_call_per_profile_verified':True,'no_training_updates':True})
    print('LKH_SEARCH_PILOT_COMPLETE',json.dumps(tables['all']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/lkh_search_pilot_20261014'))
    parser.add_argument('--per-size',type=int,default=20)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--seed',type=int,default=20261014)
    parser.add_argument('--checkpoint',type=Path,default=Path('outputs/near_miss_quota_20261013/nm20/checkpoint-20.pt'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output=args.output_dir.resolve()
    if (output/'comparison.json').exists() or (output/'cases').exists(): raise RuntimeError('Refusing existing pilot outputs')
    output.mkdir(parents=True,exist_ok=True)
    config=json.loads((args.checkpoint.parent/'resolved_config.json').read_text())
    config['trainer_params'].update(use_cuda=False,strict_resume=False,checkpoint_path=str(args.checkpoint.resolve()),
        self_improvement_parallel_workers=0,result_folder=str(output/'policy_state'))
    trainer=JointSelfImprovementTrainer(config['env_params'],config['model_params'],config['optimizer_params'],config['trainer_params'])
    try:
        if trainer.device.type!='cpu': raise RuntimeError('Pilot must leave the GPU to training')
        trainer.model.eval()
        hashes={k:tensor_digest(m.state_dict()) for k,m in [('target',trainer.model),('reference',trainer.reference_policy),('predictor',trainer.edge_decoder)]}
        groups=load_or_create_fixed_validation_sets(dataset_dir='instance/Data',validation_dir=output/'dataset',
            problem_sizes=(20,40,60,80,100),per_size=args.per_size,seed=args.seed)
        instances=[x for size in sorted(groups) for x in groups[size]]
        fps={fingerprint(x) for x in instances}
        if len(fps)!=len(instances): raise RuntimeError('Pilot has duplicate instances')
        checked=[]
        for directory in ('validation/e500_logic_e100_20261010/validation2000',
                          'validation/e500_logic_e100_20261010/test500','validation/search_ablation_20261012/diagnostic100'):
            old=[]
            for size in groups:
                payload=torch.load(Path(directory)/f'J{size}.pt',map_location='cpu',weights_only=False)
                old.extend(instance_from_dict(x) for x in payload['instances'])
            if fps & {fingerprint(x) for x in old}: raise RuntimeError(f'Pilot overlaps {directory}')
            checked.append(directory)
        atomic_json(output/'dataset_manifest.json',{'seed':args.seed,'count':len(instances),
            'per_size':args.per_size,'fingerprints':[fingerprint(x) for x in instances],'disjoint_checked':checked})
        tasks=[]
        lkh=LKHConfig(str(Path('third_party/lkh3_20261014/LKH').resolve()))
        count=0
        set_seed(args.seed)
        for size,group in sorted(groups.items()):
            for offset in range(0,len(group),8):
                batch=group[offset:offset+8]
                with torch.no_grad():
                    trainer.env.load_problems(batch,trainer.device)
                    state,_,_=trainer.env.reset()
                    trainer.model.pre_forward(state.node_features)
                    predicted=trainer._predict_contribution_matrices(batch)
                    candidates=trainer._generate_reference_candidates(batch)
                    anchors,_,_=trainer._select_anchor_sequences(batch,candidates,predicted)
                for instance,pool,anchor in zip(batch,candidates,anchors):
                    tasks.append((instance,pool,anchor,trainer.si_final_solver_config,trainer.si_local_search_config,
                        lkh,count,str(output),trainer.si_feasibility_tolerance,trainer.si_upper_bound_tolerance))
                    count+=1
        if any(tensor_digest(m.state_dict())!=hashes[k] for k,m in [('target',trainer.model),('reference',trainer.reference_policy),('predictor',trainer.edge_decoder)]):
            raise RuntimeError('Candidate generation changed model weights')
        metadata={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_epoch':20,'frozen_policy_hashes':hashes,
            'instances':len(instances),'workers':args.workers,'device':'cpu','seed':args.seed,
            'reference_candidates':200,'top_k':30,'diverse':10,'quality_pool':60,'quality_gap':.05,
            'original_search':asdict(trainer.si_local_search_config),'lkh_config':asdict(lkh),
            'lkh_binary_sha256':hashlib.sha256(Path(lkh.binary).read_bytes()).hexdigest(),
            'lkh_source':json.loads(Path('third_party/lkh3_20261014/SOURCE_PROVENANCE.json').read_text()),
            'near_miss_nominal':.2,'pilot_per_J_batch':args.per_size,'no_training_updates':True}
        metadata['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path('scripts/lkh_fixed_bound.py'),Path('src/CertifiedLocalSearch.py'),
             Path('src/CertifiedUpperBound.py'),Path('src/JointSelfImprovement.py')]}
        for name in metadata['source_sha256']:
            path=Path(name)
            relative=path.resolve().relative_to(Path.cwd())
            destination=output/'source_snapshot'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
        atomic_json(output/'protocol.json',metadata)
        print(f'PILOT_INPUTS_READY {len(tasks)} instances; CPU {args.workers} workers, no GPU or training updates',flush=True)
        records=[]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as executor:
            for record in executor.map(search_case,tasks,chunksize=1):
                records.append(record)
            evaluate_gates(records,instances,executor,trainer.si_final_solver_config,
                           trainer.si_local_search_config,trainer.si_upper_bound_tolerance)
        summary(records,metadata,output)
    finally:trainer.close()


if __name__=='__main__':main()
