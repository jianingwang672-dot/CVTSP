# Field Analysis Report (guided review, round 1)

Accepted by the author on 2026-11-06, unchanged. Reviewed file: `paper/full_20261104/phase4_draft/main.tex`.

## Paper Basic Information
- **Title**: Deep Reinforcement Learning with Sparse Exact Feedback for the Carrier–Vehicle Traveling Salesman Problem
- **Abstract length**: about 190 words
- **Full text length**: about 6,400 words of body text, excluding tables
- **Number of references**: 41

## Field Analysis

| Dimension | Analysis Result |
|---|---|
| Primary Discipline | Operations research and logistics optimisation (drone-assisted routing) |
| Secondary Disciplines | Neural combinatorial optimisation, convex optimisation (SOCP), maritime logistics operations |
| Research Paradigm | Quantitative: theoretical proposition plus computational experiments |
| Methodology Type | Algorithm design and comparative computational study |
| Target Journal Tier | Author declares Transportation Research Part E. `criteria_binding_unavailable`: no venue-alignment claim |
| Paper Maturity | First draft: structure complete, 15 placeholders, several covering core experimental results |

## Reviewer Configuration Cards

### Reviewer Configuration Card #1

**Role**: EIC
**Display role**: Journal-Fit Reviewer
**Identity Description**: Senior editorial-board perspective for logistics and transportation operations research journals, working on drone and carrier routing and its operational decisions.
**Review Focus**:
  1. Whether this is a logistics management paper, or an algorithm paper sent to a logistics journal.
  2. Whether the managerial insights follow from the results, or are attached afterwards.
  3. Whether placing the contribution in the use of a simple proposition in training can carry a journal paper.
**Will particularly care about**: Whether the selling point survives a 2.6% quality deficit to the baseline and better solutions from metaheuristics at ten times the time.
**Possible blind spots**: Proof details and network training details.

### Reviewer Configuration Card #2

**Role**: Peer Reviewer 1
**Display role**: Peer Reviewer 1
**Identity Description**: Computational-experiment methodologist in combinatorial optimisation, specialising in the design of algorithm comparisons and the reproducibility of DRL evaluation.
**Review Focus**:
  1. Whether the 78-fold saving can be attributed to the certified bound, given that the two models differ in network, batch size and hardware.
  2. Test-set contamination, single seed, checkpoint selection, and whether paired comparisons and confidence intervals are appropriate.
  3. Fairness of the time-budget comparison, including the estimated G200 CPU time and cross-machine wall time.
  4. Whether the ablation design can answer the questions it claims to answer.
**Will particularly care about**: Conclusions that are correlational but written as causal.
**Possible blind spots**: CVTSP literature and real operating settings.

### Reviewer Configuration Card #3

**Role**: Peer Reviewer 2
**Display role**: Peer Reviewer 2
**Identity Description**: Researcher in the CVTSP and mothership-drone routing, specialising in mixed-integer SOCP, exact algorithms and decomposition heuristics.
**Review Focus**:
  1. Correctness of the model and Proposition 1, and consistency with the definitions of Garone, Gambella and Li.
  2. Reliability of the comparison with Li 2025 and Vieira 2026, and of the handling of best known solutions.
  3. Accuracy of the research-gap table, missing key literature, and the risk from the three unread papers.
**Will particularly care about**: Whether the claims of new best solutions and of no related work found can stand.
**Possible blind spots**: RL gradients and network architecture.

### Reviewer Configuration Card #4

**Role**: Peer Reviewer 3
**Display role**: Peer Reviewer 3
**Identity Description**: Researcher in expensive-objective optimisation and multi-fidelity learning, with a background in neural combinatorial optimisation.
**Review Focus**:
  1. Treating the bound as a low-fidelity signal: how much gradient bias a loose bound causes, and whether its ranking quality was measured.
  2. Whether the explanation of why a loose bound suffices is evidence or conjecture.
  3. Links to surrogate-based optimisation and active learning, and whether those fields already hold similar conclusions.
  4. Explanations of the POMO diversity deficit other than the one offered, and whether the claimed extension to other problems goes too far.
**Will particularly care about**: Whether existing theory explains the conclusions and so weakens the novelty.
**Possible blind spots**: Maritime operations practice and CVTSP literature details.

The fifth execution seat is the fixed Devil's Advocate, which has no configuration card.

## Execution notes
- Five seats run as separate subagents with fresh contexts and no view of each other's outputs. All share one model family, which also drafted the manuscript, so correlated error is expected. This is role separation, not independence.
- `reviewer_guided` has no sprint contract template in this version, so the paper-blind Phase 1 split and the scripted checkers do not apply.
- Placeholders are out of scope for content judgement; they are assessed only as a statement of what the paper still needs.
