# Sparse-Gurobi joint self-improvement training

This is the formal training path. It uses neither CVXPYLayer nor REBAR.

## Data flow for one instance

```text
frozen reference POMO candidates       target POMO rollout
                 |                            |
                 +--> directed predictor ----+  (no solver)
                              |
                         choose one anchor
                              |
                         Gurobi #1 once
                 V(anchor), primal, dual, C_anchor^U
                              |
                  score every route by C_anchor^U
                              |
                    Top-K starts (default 4)
                              |
             swap + asymmetric 2-opt, <=30 moves
                       (zero Gurobi calls)
                              |
                         pseudo expert
                              |
              unchanged: reuse Gurobi #1 result
              changed:   Gurobi #2 exact validation
                              |
        RL update -> separate IL update -> separate edge update
```

The current reward path also exactly evaluates one selected Target route, so
training makes `2B + B_changed <= 3B` Gurobi calls for a batch of `B`
instances. The number is independent of POMO size. Validation calls are
reported separately.

## Upper level

- The existing POMO encoder/decoder is retained.
- The frozen reference policy samples pseudo-expert candidates.
- The target policy samples the RL routes.
- In `hybrid_pomo_exact_expert` mode, every Target POMO route receives a dense
  reward. One selected route uses its exact Gurobi objective and the rest use
  certified directed upper bounds. Per-instance advantages are centered and
  standardized.
- A bounded relative exact Target/Expert advantage is added as an auxiliary.
  Its fixed training-only schedule rises from zero as prior-epoch IL accuracy
  moves from 0.50 to 0.85.
- Diverse Top-K starts can be restricted by both score rank and relative score
  gap. Unfilled diversity slots fall back to the next-best routes, preventing
  poor cold-start routes from entering solely because they are different.
- Adaptive near-miss validation uses only prior-epoch IL accuracy and exact
  Target/Expert wins to interpolate between its configured minimum and maximum.
- Top-K fixed-anchor local search accepts only a strict upper-bound decrease.
- Directed 2-opt includes all reversed internal edges.
- The unranked swap/2-opt neighborhood uses cached per-J candidate indices and
  vectorized `float64` delta scoring. The selected minimum and numerical
  near-ties are re-evaluated by the original scalar formulas, preserving the
  candidate order, acceptance threshold, and first-candidate tie-breaking.
  Set `CVTSP_VECTORIZED_LOCAL_SEARCH=0` only for scalar A/B verification.
- RL and teacher-forced cross-entropy are two independent optimizer steps.
- On a fixed validation set, reference and target candidates are compared by
  exact Gurobi makespan; target replaces reference only after a strict win.

## Lower level

- A directed edge decoder predicts a nonnegative `(J+1)x(J+1)` table from
  detached POMO encodings. It ranks reference routes and chooses the anchor.
- The predictor has its own optimizer. Its loss cannot update the POMO encoder.
- Gurobi #1 preserves the existing fixed-tour convex QCP and returns exact
  makespan, primal positions/times, and Pi/QCPi values.
- Relabeling the primal solution by target constructs a rigorous reusable
  directed edge table. A depot cycle sum is exactly the fixed-solution feasible
  objective, so `V(X) <= U_anchor(X)`.
- Dual values are collected. Dual pruning is disabled by default and can be
  enabled with `--dual-pruning`; it is not required by the main path.
- If local search changes the expert, Gurobi #2 verifies its real makespan.

## Run

Short end-to-end check:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 train_joint_self_improvement.py \
  --smoke-test \
  --output-dir outputs/joint_si_sparse_smoke
```

Formal 200-epoch command:

```bash
bash run_joint_training_200.sh
```

The formal defaults use random `J` in `[20,100]`, 100,000 instances/epoch,
Top-K 4, 30 solver-free local moves, swap plus asymmetric 2-opt, and 32
instance-level self-improvement workers. Start the long run only after checking
the short-run metrics on the target machine.

## Outputs

- `metrics.csv`: batch/POMO size, anchor/expert exact makespan, exact
  improvement, dense and exact auxiliary RL losses, auxiliary weight, quality
  gate counts, effective near-miss rate, IL/edge losses, training and
  validation call counts, Gurobi time, local-search time, and call-reduction
  factor.
- `exact_improvement_epoch_<n>.json`: several concrete `anchor V`, `expert U`,
  `expert V` examples.
- `last.pt` / numbered checkpoints: target, route optimizer, edge decoder and
  optimizer, frozen reference, fixed validation instances, scheduler, and
  schema-v3 method metadata plus adaptive training state.
- Pass `--checkpoint-interval 1` to retain `checkpoint-<epoch>.pt` for every
  completed epoch instead of only retaining the rolling `last.pt`.
- `reference.pt`: latest exactly validated promoted reference.
