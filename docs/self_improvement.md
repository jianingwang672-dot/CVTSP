# Offline self-improvement ablation (legacy)

This file documents the earlier stand-alone pseudo-expert experiment.  It is
kept for ablation and reproducibility, but it is **not** the formal joint
method: it replaces POMO RL with offline imitation and trains a separate edge
encoder.  Use `docs/joint_self_improvement_training.md` and
`train_joint_self_improvement.py` for the requested shared-encoder, in-batch
POMO/Gurobi-dual/self-improvement method.

This is the complete CVTSP self-improvement loop. It reuses the POMO policy,
Gurobi fixed-tour solver, certified local search, teacher-forced imitation, and
the directed edge-cost predictor. The active workflow does not use REBAR or
cvxpylayer.

The implementation follows the two-stage self-improvement structure described
in [Multi-Action Self-Improvement for Neural Combinatorial
Optimization](https://arxiv.org/abs/2510.12273): the current best policy
generates multiple solutions, the best solution becomes a pseudo expert, and a
student learns the expert state-action pairs by imitation.  The paper's
multi-agent set loss is not copied here because this CVTSP policy emits one
next target at a time; the appropriate current loss is autoregressive
next-target cross-entropy.

## Closed-loop schedule

```text
round 1: current teacher checkpoint
  -> one free-start route + diverse POMO routes per instance
  -> exact Gurobi solve and complete certified local search
  -> select the lowest true final objective as the pseudo expert
  -> reconstruct every expert prefix and next-target label
  -> train a student with teacher-forced cross-entropy
  -> exact Gurobi promotion gate
  -> generate directed edge labels and train edge predictor

round 2 and later:
  -> load the preceding round's edge predictor
  -> rank all swap / relocate / asymmetric 2-opt moves in O(1) per move
  -> calculate the exact fixed-solution UB only for predicted Top-K moves
  -> accept only an exact-UB-certified move and refresh it with Gurobi
  -> train/gate the next student and update the predictor again
```

If promotion fails, the current teacher remains active.  A later round can
generate a new dataset while retaining that teacher.  If promotion succeeds,
the selected student checkpoint becomes the teacher for the next round.

## Pseudo experts

For each instance the policy produces up to `teacher_candidates` distinct
routes.  Candidate zero is a free-start route, so the decoder predicts its
first target.  The remaining candidates retain the existing POMO multi-start
behavior.

Every candidate enters phase-one local search:

```text
candidate -> Gurobi -> fixed-solution directed UB search -> Gurobi refreshes
```

When a predictor from the preceding round is available, the middle search is
Top-K guided. The predictor only changes which candidates receive an exact UB
calculation. It never changes the acceptance inequality, and every accepted
move is still refreshed by Gurobi. A poor predictor can miss an improvement but
cannot certify a false improvement.

The stored expert is the candidate with the lowest refreshed, true Gurobi
objective.  Because the best raw teacher candidate is itself included in the
search, the selected pseudo expert cannot be worse than the best teacher route
apart from the checked numerical tolerance.

`pseudo_experts.json` contains the complete instance parameters, every raw and
improved candidate, all accepted moves, objectives, Gurobi calls, the selected
expert, dataset split, teacher checkpoint hash, generated moves, exact-UB
evaluations, pruned moves, and aggregate generation statistics. The file is
atomically rewritten after every completed instance. Pass `--resume` to reuse
both completed rounds and a partially completed pseudo-expert dataset.

## Cross-entropy training

For expert route `q=(q1,...,qn)`, the loss is

```text
L_CE = -(1/n) sum_t log pi_theta(q_t | depot, q_1, ..., q_(t-1), instance).
```

The environment mask is advanced with the expert node rather than the model's
prediction.  In particular, after the first target, the decoder's `q_first`
state is explicitly overwritten with the expert first-target embedding.  This
is genuine teacher forcing: all later probabilities are conditioned on the
expert prefix.  No Gurobi call occurs during CE optimization.

Training reports per-token NLL, token accuracy, and exact-sequence accuracy on
train and validation records.  `best.pt` is selected by validation NLL (or
train NLL when no validation records are requested) and stores its model
configuration and source teacher checkpoint.

## Promotion gate

The promotion records are not used for gradient updates or validation model
selection.  Teacher raw objectives were recorded before student training.  The
student produces the same number of free-start/POMO routes, and every route is
solved exactly by Gurobi.  By default promotion follows the validation-style
policy update in the paper and requires:

```text
student_mean_objective < teacher_mean_objective - promotion_epsilon
```

This gate concerns policy quality; it is separate from the phase-one certified
move rule.  CE itself does not mathematically guarantee a better policy, so a
student that merely obtains a lower CE loss is never promoted automatically.
For a more conservative finite-sample gate,
`--require-no-promotion-losses` additionally requires `student_losses == 0`.

## Directed edge predictor update

At the end of each round, up to four distinct anchor tours per instance are
solved by Gurobi to form full directed carrier-time matrices. Training and
validation samples are accumulated across rounds; promotion samples are never
used for gradient updates or checkpoint selection. The predictor is initialized
from the active policy encoder, predicts from static instance features, and is
trained with directed edge Huber regression.

Without `--initial-edge-predictor-checkpoint`, round one intentionally uses the
complete certified neighborhood and produces the bootstrap predictor. Round two
automatically uses that checkpoint. Supplying an initial checkpoint enables
Top-K guidance from round one. `--disable-edge-predictor` restores the old
complete-neighborhood self-improvement loop.

## Run the complete loop

```bash
python3 run_self_improvement.py \
  --teacher-checkpoint outputs/train__gurobi_pomo/best.pt \
  --output-dir outputs/self_improvement_closed_loop \
  --rounds 3 \
  --train-instances 8 \
  --validation-instances 2 \
  --promotion-instances 2 \
  --min-problem-size 20 \
  --max-problem-size 100 \
  --teacher-candidates 4 \
  --local-search-refreshes 10 \
  --imitation-epochs 30 \
  --imitation-batch-size 4 \
  --learning-rate 1e-4 \
  --edge-top-k 30 \
  --edge-max-anchors-per-instance 4 \
  --edge-epochs 50 \
  --edge-batch-size 8 \
  --threads 1 \
  --device cpu \
  --seed 20260812
```

Each round has its own pseudo-expert data, student checkpoint, exact promotion
details, current and cumulative edge datasets, edge-predictor checkpoint,
ranking metrics, and summary. Restart the same command with `--resume` after an
interruption.

An independent checkpoint comparison can be run without local search:

```bash
python3 compare_policy_checkpoints.py \
  --teacher-checkpoint outputs/train_time_test/best.pt \
  --student-checkpoint outputs/self_improvement_phase2/round_001/student/best.pt \
  --instances 20 \
  --problem-size 10 \
  --candidates 4 \
  --seed 20269999 \
  --output-json outputs/self_improvement_phase2/independent_comparison.json
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The phase-two tests cover instance/dataset round trips and verify that
teacher-forced CE lowers expert NLL while improving token accuracy.
