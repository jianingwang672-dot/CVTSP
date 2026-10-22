"""Audit B's actual execution and compare with the archived nm20 reference."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from src.SearchDiagnostics import atomic_json
from src.QualityILTrainer import quality_multiplier
from scripts.watch_search_ablation import completed_metric_rows
from scripts.summarize_search_ablation import epoch_cycle_times, paired_values, stage_transitions


def summarize(root):
    directory=root/'b'
    reference=Path('outputs/near_miss_quota_20261013/nm20')
    rows=completed_metric_rows(directory/'metrics.csv')
    assert [int(float(x['epoch'])) for x in rows] == list(range(1,21)), 'B needs all20 epochs'
    assert 'TRAIN_EXIT=0 ' in (directory/'live.log').read_text(), 'B did not exit successfully'
    config=json.loads((directory/'resolved_config.json').read_text())
    params=config['trainer_params']
    assert params['quality_weighted_il'] and params['near_miss_ablation_rate']==.2
    assert params['train_episodes']==20000 and params['train_batch_size']==64
    assert params['imitation_inner_epochs_schedule']==[[1,5]]
    assert params['changed_expert_il_weight']==5 and params['expert_replay_fraction']==.3
    assert params['self_improvement_max_local_moves']==params['move_schedule_final']==30
    init=json.loads((directory/'initialization.json').read_text())
    old_init=json.loads((reference/'initialization.json').read_text())
    assert init['fresh_initialization'] and init['start_epoch']==1 and init['replay_size']==0
    for key in ('target_sha256','reference_sha256','predictor_sha256','data_rng_sha256','problem_size_rng_sha256'):
        assert init[key]==old_init[key], f'initialization differs: {key}'
    provenance=json.loads((directory/'provenance.json').read_text())
    for name,digest in provenance['source_sha256'].items():
        assert hashlib.sha256((directory/'source_snapshot'/name).read_bytes()).hexdigest()==digest
    old_rows=completed_metric_rows(reference/'metrics.csv')
    quality=[]
    for row,old in zip(rows,old_rows):
        epoch=int(float(row['epoch']))
        assert row['train_data_sha256']==old['train_data_sha256'], f'training stream differs E{epoch}'
        assert float(row['max_local_moves'])==30
        assert float(row['near_miss_rate'])==.2
        cp=directory/f'checkpoint-{epoch}.pt'
        assert cp.is_file() and cp.stat().st_size>0
        qs=[json.loads(x) for x in (directory/'diagnostics'/f'epoch-{epoch:03d}_quota_batches.jsonl').read_text().splitlines()]
        assert len(qs)==313 and sum(x['instances'] for x in qs)==20000
        for x in qs:
            assert x['nominal_rate']==.2 and x['quota_limit']==math.ceil(.2*x['instances'])
            assert x['attempted']==min(x['quota_limit'],x['eligible']) and 0<=x['accepted']<=x['attempted']
        audit=json.loads((directory/'diagnostics'/f'epoch-{epoch:03d}_quality_il.json').read_text())
        labels=[json.loads(x) for x in (directory/'diagnostics'/f'epoch-{epoch:03d}_il_labels.jsonl').read_text().splitlines()]
        assert audit['passes']==5 and audit['frozen_across_passes']
        assert sum(x['effective_copies'] for x in labels)==audit['effective_labels']
        assert audit['extra_target_gurobi_calls']==audit['unique_labels']
        assert len({(x['fingerprint'],tuple(x['expert_sequence'])) for x in labels})==audit['unique_labels']
        assert sum(x['effective_copies'] for x in labels if x['source']=='replay')==audit['replay_examples']
        assert sum(x['effective_copies'] for x in labels if x['source']=='current_anchor')+sum(x['effective_copies']//5 for x in labels if x['source']=='current_changed')==20000
        for x in labels:
            gap,q=quality_multiplier(x['target_exact'],x['expert_exact'])
            assert math.isclose(gap,x['relative_gap'],abs_tol=1e-12)
            assert math.isclose(q,x['quality_multiplier'],abs_tol=1e-12)
        assert math.isclose(sum(audit['normalized_ce_contribution_share'].values()),1.,abs_tol=1e-8)
        assert audit['extra_target_gurobi_calls']==int(float(row['quality_il_extra_target_calls']))
        quality.append(audit)
    # Check the actual final resume state, including cached bounds for every replay label.
    import torch
    from src.QualityILTrainer import QualityILTrainer
    cp=torch.load(directory/'checkpoint-20.pt',map_location='cpu',weights_only=False)
    for key in ('model_state_dict','optimizer_state_dict','reference_policy_state_dict','edge_decoder_state_dict',
                'edge_optimizer_state_dict','joint_rng_state','expert_replay_pool','quality_il_label_cache','strict_resume_signature'):
        assert key in cp,key
    assert cp['epoch']==20 and len(cp['expert_replay_pool'])<=50000
    assert cp['main_scheduler_config']['total_epochs']==100 and cp['main_scheduler_config']['min_lr']==1e-5
    cache=cp['quality_il_label_cache']
    expected={QualityILTrainer._label_key(x['instance'],x['expert_sequence']) for x in cp['expert_replay_pool']}
    assert set(cache)==expected
    assert all(math.isfinite(v['expert_exact']) and v['expert_exact']>0 for v in cache.values())
    del cp,cache
    paired={}
    for name in ('epoch-020_after_il_free_greedy.json','epoch-020_after_il_full_pomo_greedy.json'):
        a=json.loads((reference/'diagnostics'/name).read_text())
        b=json.loads((directory/'diagnostics'/name).read_text())
        assert [x['fingerprint'] for x in a['records']]==[x['fingerprint'] for x in b['records']]
        paired[name]={'reference_mean':a['mean'],'B_mean':b['mean'],
                      'B_minus_reference':paired_values([x['makespan'] for x in a['records']],[x['makespan'] for x in b['records']])}
    best=min(rows,key=lambda x:float(x['target_mean_objective']))
    cycles=epoch_cycle_times(directory/'live.log')
    summary={'complete':True,'historical_reference_only':True,'same_initialization':True,'same_training_instances':True,
             'best_epoch':int(float(best['epoch'])),'best_fixed_validation':float(best['target_mean_objective']),
             'epoch20_fixed_validation':float(rows[-1]['target_mean_objective']),
             'reference_epoch20_fixed_validation':float(old_rows[-1]['target_mean_objective']),
             'quality_by_epoch':quality,'stage_transitions':stage_transitions(directory),'paired_diagnostics':paired,
             'mean_cycle_minutes':sum(cycles.values())/len(cycles)/60,
             'extra_target_calls_total':sum(x['extra_target_gurobi_calls'] for x in quality),
             'measurement_minutes_total':sum(x['measurement_seconds'] for x in quality)/60}
    atomic_json(root/'comparison.json',summary)
    atomic_json(root/'completion_audit.json',{'passed':True,'epochs':20,'all_label_weights_verified':True,
                'checkpoint_and_replay_cache_verified':True,'all_training_data_matches_nm20':True})
    text=['# IL质量加权B：20轮结果','',f"固定验证best：E{summary['best_epoch']}，{summary['best_fixed_validation']:.6f}。",
          f"Epoch20：B={summary['epoch20_fixed_validation']:.6f}，历史nm20={summary['reference_epoch20_fixed_validation']:.6f}。",'',
          '| 独立诊断 | 历史nm20 | B | B−历史：95%区间 |','|---|---:|---:|---|']
    for name,v in paired.items():
        text.append(f"| {name} | {v['reference_mean']:.6f} | {v['B_mean']:.6f} | {v['B_minus_reference']['ci95']} |")
    text+=['',f"每轮平均墙钟：{summary['mean_cycle_minutes']:.2f}分钟；额外精算：{summary['extra_target_calls_total']}次。",
           '单seed、20轮，且仅运行B；历史nm20作为参考，不能单凭差异确定改进因果或100轮效果。',
           '测量比较一条上界选中的精算Target路线；质量权重并不保证IL时或每个训练步持续优于Target。',
           '标签贡献份额、IL前后质量与额外耗时见comparison.json；正式test500/Real529未用于调参。']
    (root/'report.md').write_text('\n'.join(text)+'\n')
    print('QUALITY_IL_B_E20_AUDITED_AND_SUMMARIZED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/quality_il_b_20261014'))
    summarize(parser.parse_args().root)
