"""ATSP adapter shared by the offline diagnostic and the opt-in LKH trainer."""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import time

import numpy as np

from src.CertifiedUpperBound import validate_tour
from src.JointSelfImprovement import directed_route_upper_bound


@dataclass(frozen=True)
class LKHConfig:
    binary: str
    desired_scale: int = 1_000_000
    trials: int = 1
    max_candidates: int = 5
    move_type: int = 5
    native_seconds: float = 2.
    process_timeout_seconds: float = 5.


def quantize_matrix(matrix, desired_scale=1_000_000):
    costs=np.asarray(matrix,dtype=np.float64)
    if costs.ndim!=2 or costs.shape[0]!=costs.shape[1] or costs.shape[0]<3:
        raise ValueError('ATSP requires a square matrix with depot and at least two targets')
    if not np.isfinite(costs).all() or np.min(costs)<0 or desired_scale<=0:
        raise ValueError('invalid ATSP matrix or scale')
    # Precision=1. Leave ample room in 32-bit edge arithmetic; tour GainType
    # in this source is long long. Acceptance never uses these integer costs.
    maximum=float(costs.max())
    safe=int((np.iinfo(np.int32).max//8)/max(maximum,1e-12))
    scale=min(int(desired_scale),safe)
    if scale<1: raise ValueError('ATSP costs exceed the safe integer range')
    quantized=np.rint(costs*scale).astype(np.int64)
    np.fill_diagonal(quantized,0)
    return quantized,scale, costs.shape[0]/(2.*scale)


def prepare_problem(matrix, directory, config):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    weights,scale,error=quantize_matrix(matrix,config.desired_scale)
    path=directory/'problem.atsp'
    path.write_text('\n'.join(['NAME: fixed_anchor','TYPE: ATSP',f'DIMENSION: {len(weights)}',
        'EDGE_WEIGHT_TYPE: EXPLICIT','EDGE_WEIGHT_FORMAT: FULL_MATRIX','EDGE_WEIGHT_SECTION',
        *(' '.join(map(str,row)) for row in weights),'EOF'])+'\n')
    return {'scale':scale,'route_rounding_error_bound':error,'matrix_file':str(path)}


def read_tour(path, problem_size):
    text=Path(path).read_text()
    if 'TOUR_SECTION' not in text: raise ValueError('LKH tour has no TOUR_SECTION')
    ids=[]
    for token in text.split('TOUR_SECTION',1)[1].split():
        if token in ('-1','EOF'): break
        ids.append(int(token))
    n=problem_size+1
    if len(ids)!=n or sorted(ids)!=list(range(1,n+1)):
        raise ValueError('LKH tour must visit every target and depot exactly once')
    # Rotate to depot, never reverse: the edge table is directed.
    depot=ids.index(1)
    ids=ids[depot:]+ids[:depot]
    return list(validate_tour([node-2 for node in ids[1:]],problem_size))


def run_start(matrix, initial, directory, config, *, prefix, seed, deadline=None,
              absolute_epsilon=1e-4, relative_epsilon=1e-8):
    started=time.perf_counter()
    j=len(matrix)-1
    initial=list(validate_tour(initial,j))
    initial_bound=directed_route_upper_bound(initial,matrix)
    result={'initial_sequence':initial,'final_sequence':initial,'initial_upper_bound':initial_bound,
        'final_upper_bound':initial_bound,'called':False,'completed':False,'timed_out':False,
        'accepted_output':False,'wall_seconds':0.,'native_output_upper_bound':None}
    remaining=(float('inf') if deadline is None else deadline-time.perf_counter())
    if remaining<=0: return result
    directory=Path(directory)
    tour=directory/f'{prefix}.tour'
    if tour.exists(): raise FileExistsError(f'Refusing stale LKH result: {tour}')
    initial_path=directory/f'{prefix}.initial'
    initial_path.write_text('\n'.join(['TYPE: TOUR',f'DIMENSION: {j+1}','TOUR_SECTION','1',
                                     *(str(node+2) for node in initial),'-1','EOF'])+'\n')
    native_time=min(config.native_seconds,remaining)
    parameter=directory/f'{prefix}.par'
    parameter.write_text('\n'.join(['PROBLEM_FILE = problem.atsp',f'INITIAL_TOUR_FILE = {initial_path.name}',
        'INITIAL_TOUR_FRACTION = 1.0',f'OUTPUT_TOUR_FILE = {tour.name}',
        'RUNS = 1',f'MAX_TRIALS = {config.trials}',f'MAX_CANDIDATES = {config.max_candidates}',
        f'MOVE_TYPE = {config.move_type}','PRECISION = 1','SCALE = 1',f'SEED = {seed}',
        f'TIME_LIMIT = {native_time:.9f}','TRACE_LEVEL = 0'])+'\n')
    remaining=(config.process_timeout_seconds if deadline is None else deadline-time.perf_counter())
    if remaining<=0:
        result['wall_seconds']=time.perf_counter()-started
        return result
    result['called']=True
    with (directory/f'{prefix}.log').open('w') as log:
        try:
            process=subprocess.run([str(Path(config.binary).resolve()),parameter.name],cwd=directory,
                stdout=log,stderr=subprocess.STDOUT,timeout=min(config.process_timeout_seconds,remaining))
            if process.returncode:
                raise RuntimeError(f'LKH failed ({process.returncode}); see {directory/prefix}.log')
            result['completed']=True
        except subprocess.TimeoutExpired:
            result['timed_out']=True
    if tour.exists():
        try:
            route=read_tour(tour,j)
        except ValueError:
            if not result['timed_out']: raise
        else:
            upper=directed_route_upper_bound(route,matrix)
            result['native_output_upper_bound']=upper
            epsilon=max(absolute_epsilon,relative_epsilon*max(1.,abs(initial_bound)))
            if upper<initial_bound-epsilon:
                result.update(final_sequence=route,final_upper_bound=upper,accepted_output=True)
    elif result['completed']:
        raise RuntimeError('LKH completed without writing its best tour')
    result['wall_seconds']=time.perf_counter()-started
    return result
