# Peer Review Report: Perspective Review Report (Peer Reviewer 3)

## Manuscript Information
- **Title**: Deep Reinforcement Learning with Sparse Exact Feedback for the Carrier–Vehicle Traveling Salesman Problem
- **Manuscript ID**: not assigned (draft `paper/full_20261104/phase4_draft/main.tex`)
- **Review Date**: 2026-11-06
- **Review Round**: Round 1, guided mode

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 3 (Perspective)

### Reviewer Identity
Researcher in expensive-objective optimisation and multi-fidelity learning, with a background in neural combinatorial optimisation.

### Review Focus
I read the certified bound as a low-fidelity signal. I ask how much gradient bias a loose bound causes, whether its ranking quality was measured, and whether the explanation of why looseness is harmless is evidence or conjecture. I also ask whether surrogate-based optimisation and active learning already hold the same conclusions, whether the POMO diversity story has rivals, and whether the claimed extension to other problems goes too far.

### Scope notes
- I am an outsider to maritime operations and to CVTSP literature details. Comments on operations are feasibility questions, not domain judgements.
- Repository evidence below comes from `docs/` and the named `src/` files. It is marked as repository evidence because it is not in the manuscript. Most of those diagnostics used earlier checkpoints (S500-E100, S500-E365, E150, E163), not E133, and each finding says which.
- Red placeholders are not faulted for missing content. Where one carries a risk, the risk is stated as a risk.
- Beyond the materials the brief named, I also viewed the header and execution notes of `phase0_field_analysis.md` (no other card), `src/PerStartILTrainer.py`, `scripts/run_queue_20261030.sh`, a grep over `scripts/`, directory listings of `outputs/`, `notes/` and `reports/`, and one memory note. No finding rests on any of these.

---

## Overall Assessment

### Recommendation
- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision** — Substantial revisions needed, re-review required after revision
- [ ] Reject

### Confidence Score
4 (multi-fidelity and neural combinatorial optimisation are within my expertise; maritime practice is not). Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Calibration Status
`NOT_CALIBRATED`

### Summary Assessment
The paper trains a POMO-type policy for the CVTSP mainly on a certified upper bound from one exact SOCP solution, with a few exact solves per instance, and reports 78 times fewer solver calls than an exact-reward baseline at a mean gap of +2.60% on synthetic instances. Proposition 1 is sound, and the paper is candid about where it loses. Seen as multi-fidelity learning, the method is a good design with an unmeasured low-fidelity model. The manuscript never reports how well the bound ranks routes or how well its gradient agrees with the exact one, although the authors' repository holds such measurements (cosine 0.61, 0.53 at J = 100, on earlier checkpoints). The account of why a loose bound suffices is therefore conjecture, and its one experiment changed the bound by 0.32%, too small a dose to test anything. The diversity explanation of the remaining gap conflicts with the authors' own earlier diagnostics. The 78-fold headline is mostly a 15-fold choice of training length, and only 5.2-fold comes from sparse feedback. Existing theory (majorise–minimise, trust-region model management) already explains much of the design, which helps the paper if cited. I recommend Major Revision because these points need new measurements and a rewritten §6.3 and §6.4, while the core construction survives.

---

## Strengths

### S1: A low-fidelity model with a proved error direction and a runtime check
Most surrogates in my field carry no sign guarantee. Here the error direction follows from feasibility, exactness at the anchor is proved, and the code enforces the inequality at every exact solve. I confirmed the checks in `src/JointSelfImprovement.py` and `src/JointSelfImprovementTrainer.py` (they raise on violation).
**Evidence Anchor**: `text: §4.2 "The training code also checks the inequality whenever an expert, near-miss or refresh route is solved exactly, and stops on a violation."`

### S2: High-fidelity verification gates the labels
The imitation channel keeps the high-fidelity model in the loop, which is the safeguard multi-fidelity methods rely on. The gate covers imitation only and not the dense reward (see W2), but it is correct and clearly stated.
**Evidence Anchor**: `text: §4.5 "Only verified experts enter this update, so an error in the bound can waste a solve but can never become a label."`

### S3: Candour about the price
The paper keeps the 2.60% gap and the metaheuristic result in the abstract instead of leading only with savings. This makes the practical advice usable.
**Evidence Anchor**: `text: §1 "We report both results alongside the savings, because a planner choosing a method needs all three."`

### S4: Budget broken down by purpose
A per-purpose call table is what a multi-fidelity reader wants, and it charges failed near-miss solves. It is the basis for the cost analysis in W8 and W9.
**Evidence Anchor**: `table: Table 8 — calls by purpose, total 3,839,970, 2.89 per instance`

---

## Weaknesses

### W1: The fidelity of the low-fidelity signal is never reported
**Problem**: The bound is the main training signal, yet the manuscript gives no rank correlation with exact costs, no agreement between bound-based and exact advantages, no gradient-level measure, and no slack as a function of distance from the anchor. The claim that the bound is "accurate nearby" has no number behind it. The repository holds relevant measurements that the paper omits. In `docs/reward_diversity_diagnostic_20261015.md`, within-group Spearman between bound and exact cost is 0.612 (S500-E100) and 0.681 (S500-E365). Cosine between the hybrid advantage and the all-exact advantage is 0.606 and 0.670, falling to 0.528 and 0.539 at J = 100. The best exact route receives a negative advantage in 8% to 12% of cases. On accuracy near the anchor, `docs/random_subset_reward_experiment_20261015.md` reports 96 near-miss candidates on a frozen policy, of which 29 were truly better, while the bound could certify only 4. That pilot suggests the certificate misses most nearby improvements.
**Evidence Anchor**: `absence: §4.5, §5.6 and §6.3 — expected a measured rank correlation or gradient agreement between bound and exact costs on policy routes; checked Abstract, §3.2, §4.2 to §4.5, §5.6, Tables 8 and 9, §6.3, §6.4`
**Why it matters**: In my field a low-fidelity model is characterised before it is trusted. Without this, the reader cannot tell whether the method works because the bound is informative, or despite its being weak. These numbers also make the paper's argument more interesting, because a signal with cosine near 0.6 trained a policy to within 3% of the exact baseline.
**Suggestion**: Rerun the existing zero-training diagnostic on E133 and G200 and report rank correlation, advantage cosine and slack by J. Add a slack-against-distance curve (for example, number of differing arcs from the anchor). Add a gradient-level cosine, because the repository cosine compares advantage vectors, not gradients. The cost is small: about 200 instances with all POMO routes solved exactly.
**Severity**: Major
**Confidence**: 5 — core expertise: multi-fidelity model assessment; repository values are from earlier checkpoints, not E133

### W2: "Only the ranking and the variance" understates the effect, because the bound biases the gradient
**Problem**: §6.3 places the consequences of looseness in ranking and variance. In a REINFORCE update, ranking is the gradient direction. The dense term follows the expectation of a different objective, the table value of a route, and not f. That difference is bias, and it does not average out. It also has structure: the table is tightest around the anchor, so the signal favours routes that resemble the anchor. The hybrid construction adds a second effect. In `_policy_rl_update`, one route chosen by lowest bound receives its exact cost while the others keep bounds, and the standardisation is applied to this mixture. Because exact cost sits below the bound values, that route becomes an outlier. For S500-E100 the repository records a selected-route z-score of 3.79 under the hybrid reward, 2.26 if its bound had been kept, and 1.28 under all-exact rewards. Part of the reward for that route therefore comes from its having been solved, not from its quality.
**Evidence Anchor**: `text: §6.3 "so the looseness affects only the ranking of candidates and the variance of the reward"`
**Why it matters**: Convergence behaviour differs between the two cases. Variance slows training and can be averaged away. Bias that is not bounded relative to the gradient leaves SGD in a neighbourhood of the solution (Ajalloeian and Stich, arXiv:2008.00051). A bias floor is a candidate explanation for the persistent synthetic gap, and the current wording rules it out without a test.
**Suggestion**: Rewrite the sentence to separate bias from variance, and disclose the mixed-fidelity artefact in §4.5. Test a control-variate estimator: bound for all routes, plus a correction (exact minus bound) on a uniformly random subset. This gives an unbiased multi-fidelity gradient, unlike correction on the lowest-bound route. It also addresses the variance that the authors suspect in their random-subset run, because the bound still covers every route. This is untested and may not help. Khairy and Balaprakash (arXiv:2206.05165) is a starting point for the form of the estimator.
**Severity**: Major
**Confidence**: 5 — core expertise: biased and multi-fidelity gradient estimators; code read directly

### W3: A 0.32% change cannot show that tightness is not limiting
**Problem**: The inference rests on the multi-table continuation. That experiment tightened a bound with 40% to 100% slack by 0.32% (`docs/recipe_diversity_20261022.md`, which itself calls this negligible). No outcome could have been expected from that dose, so the null result says nothing about the marginal value of tightness. The continuation also began at E133, a stage the same document describes as already collapsed. A single-seed, 30-epoch run from a late checkpoint cannot speak to the value of a tighter bound during early learning.
**Evidence Anchor**: `text: §6.3 "A much tighter bound is thus not easy to obtain from nearby solutions, and on the evidence so far it is not the limiting factor."`
**Why it matters**: This is the main evidence for the paper's wider lesson. The more informative experiment is omitted: the from-scratch random-subset study replaced the dense signal with exact comparisons and found early gains at large J, then worse results by E15 and E20 (B1 minus A1 at E20: +0.058 [+0.026, +0.091], at about 3.7 times the solver time, single seed, `docs/random_subset_reward_experiment_20261015.md`). That result partly supports the authors' position, and it is stronger than what they cite. It also shows that the question is open and depends on the training stage.
**Suggestion**: Withdraw "not the limiting factor" unless a real dose is tested. Report the random-subset study. Run a dose–response design from scratch that interpolates slack, for instance by mixing a proportion λ of exact costs into the bound on randomly chosen routes, with validation quality against λ and against solver budget.
**Severity**: Major
**Confidence**: 5 — core expertise: experimental evaluation of surrogates

### W4: The planned ablations do not isolate reward fidelity
**Problem**: §6.3 assigns the test of the claim to Group B, but §5.7 defines Group B as replacing table-based search with exact evaluation of moves, measured against training time. That tests the search role of the table. The dense reward, which is what looseness affects, is unchanged in that description. Group D concerns expert diversity. Group A retrains the baseline. No arm varies the fidelity of the reward while holding the rest of the loop fixed.
**Evidence Anchor**: `text: §6.3 "Group B of the ablation will quantify this claim."`
**Why it matters**: After the queued runs finish, the central explanatory claim will still lack direct evidence, and the attribution of the quality gap between reward, search and imitation will remain confounded. This is a risk carried by a placeholder, not a fault in its unfinished content.
**Suggestion**: Add one arm before the queue closes: the same loop with exact rewards on all routes (or on an unbiased sample) at the same instance count. Comparing it with the proposed method measures the quality cost of the bound alone, separated from the architecture and instance-count differences with G200.
**Severity**: Major
**Confidence**: 5 — core expertise: ablation design for surrogate-assisted learning

### W5: The certificate is never cashed in, so validity and usefulness are not separated
**Problem**: The paper credits the known error direction for making the bound an acceptance test. But every route that passes the test is then solved exactly (`run_sparse_self_improvement_task`), so the certificate never replaces a solve. It only avoids solving routes that fail. Near-miss then removes even that role. For J ≥ 60 the near-miss rate is ramped towards 1, and the best route by the bound is solved whether it is certified or not. From Table 8 and §5.6, 364,974 of 493,128 verification calls (74%) are near-miss checks, assuming all attempts are counted in that row. So what does the work in screening is the ranking the bound supplies. An uncertified surrogate of equal ranking quality might do equally well, which is the alternative §2.3 argues against.
**Evidence Anchor**: `text: §2.3 "has a known direction of error, which is what makes it usable as an acceptance test"`
**Why it matters**: Table 1 and the paper's positioning rest on "certified". If certification adds nothing beyond ranking, the distinctive contribution is cheap and informative, not guaranteed. Rank-based surrogate work (Loshchilov et al., 2010) shows that comparison-based optimisers can use surrogates that preserve ranks only. No control tells us which holds here.
**Suggestion**: Add a free-proxy control: an order-level proxy that costs no solve, such as a closed-form table in the manner of the initial matrix of Li et al., or a plain target tour length. Report its rank correlation against f alongside the bound's. If a zero-solve proxy ranks about as well as a 0.6 Spearman bound, that changes the story. Also train with an uncertified table of equal rank quality, for example the edge-predictor table. Separately, report how many solves the certified test saved and the hit rate of near-miss checks, neither of which is given.
**Severity**: Major
**Confidence**: 4 — core expertise on surrogate screening; the 74% relies on my reading of the Table 8 row

### W6: The attribution of the gap to route diversity is not supported, and the repository partly contradicts it
**Problem**: The manuscript gives one contrast: overlap 0.62 against 0.53 at J = 100, on E163. The fuller data point elsewhere.
- Across sizes the pattern runs the wrong way. From `docs/recipe_diversity_20261022.md`, overlap on E163 against G200 is 0.638/0.476 at J = 20, 0.670/0.546 at J = 60 and 0.622/0.529 at J = 100, so the difference shrinks with J (0.162, 0.124, 0.093). The E133 gap in Table 4 grows over the same sizes (+1.08%, +3.39%, +3.05%).
- `docs/reward_diversity_diagnostic_20261015.md` concludes that G200 is not more diverse than S500: edge Jaccard 0.526 against 0.553 at S500-E100, with a test500 gap of about 3%. It attributes the gap to per-route quality.
- In that diagnostic the mean POMO route is already 1.5% worse than G200 (20.46 against 20.16), against 2.6% for best-of-POMO (17.84 against 17.38). More than half of the deficit is therefore a shift in the whole route distribution, not a loss of spread.
- The argument from augmentation (`docs/progressive_recipe_20261020.md`) does not separate the hypotheses. S500-E100 and G200 have similar overlap but augmentation benefits of 2.37% and 0.98%. A policy with wider route-cost dispersion gains more from extra draws regardless of edge overlap.
- In the same-initialisation comparison, the random-subset reward lowered overlap (J = 20: 0.374 to 0.330), yet by E20 quality was worse than the control (W3). Lower overlap did not bring better quality in that window.

Rival explanations are also unaddressed: 15 times fewer training instances, a different architecture (4 layers at 256 against 6 layers at 128), bias in the reward (W2), and imitation of a single expert. The diversity evidence is fair for one narrower claim, the post-peak decline at small J, and even there `docs/e100_plateau_branches_20261019.md` labels it an inference not verified by experiment.
**Evidence Anchor**: `text: §6.4 "The evidence points to route diversity."`
**Why it matters**: This sentence sets the future-work agenda in the Conclusion. If the deficit is mainly per-route quality, work on diversity will not close it.
**Suggestion**: Present diversity as one hypothesis among several and report overlap by J on E133 with intervals. Decompose the gap at each J into shift (mean of POMO) and spread (best minus mean). Add a coverage measure such as the best route over a very large sample from each policy, which separates what the policy can reach from what greedy POMO finds.
**Severity**: Major
**Confidence**: 4 — adjacent expertise in neural combinatorial optimisation; figures come from earlier checkpoints and should be recomputed on E133

### W7: The reward is dismissed as a cause of collapse on the weakest test available
**Problem**: The statement rests on continuations from an already collapsed E133. Earlier same-initialisation runs in `docs/e100_plateau_branches_20261019.md` show the opposite. Swapping only the reward lowered J = 20 overlap from 0.374 to 0.330 and from 0.379 to 0.307. That document states that the table is tightest around the anchor and that the hybrid reward favours anchor-like routes, and `docs/progressive_recipe_20261020.md` lists the bound-based reward among three drivers of collapse. The manuscript's own §5.6 points the same way, since it reads the falling excess in Table 9 as policy routes moving closer to the anchors. The recipe document's list of drivers does not appear in the paper.
**Evidence Anchor**: `text: §6.4 "Attempts to recover diversity by changing the reward bound or the imitation weight had little effect."`
**Why it matters**: From outside the field, this mechanism has a familiar name. A cost estimate that is exact on visited data and pessimistic elsewhere resembles conservative value estimation in offline RL (Kumar et al., 2020), where the conservatism discourages departures from the data. In this reading the bound is a distance-from-anchor penalty, so some collapse is a designed-in consequence of its validity. That would make the gap a property of certified pessimism, which is a more important finding than a side effect of the recipe. This is a hypothesis of mine, not an established result.
**Suggestion**: Report the same-initialisation result. Test the mechanism directly by measuring, per POMO start, the correlation between a route's distance from the anchor and its advantage, and by training with a reward that removes the slack's dependence on anchor distance, or with per-start anchors. `docs/recipe_diversity_20261022.md` already points to per-start labels on the imitation side; the reward side deserves the same test.
**Severity**: Major
**Confidence**: 4 — cross-field analogy from offline RL; the repository data support the direction, but the mechanism is untested

### W8: Most of the 78-fold saving comes from training length, not from sparse feedback
**Problem**: Table 10 factorises exactly. G200 used 14.97 calls per instance and E133 used 2.89, a ratio of 5.2. Instances differ by 15.0. Their product is 78. So the sparse-feedback design accounts for a factor of 5.2, and the remaining factor of 15 comes from stopping after 1.33 million instances. Stopping earlier is a choice available to any method, and quality is not held equal: E133 is behind by 2.60% on test500. As a scale, 3.84 million calls at G200's rate buys about 257,000 instances, or about 2.6 of its 200 epochs. Quality at that point is the number that matters, and it is the question Group A addresses.
**Evidence Anchor**: `text: §4.1 "This independence is the source of the savings."`
**Why it matters**: In expensive-objective work, claims of efficiency are made on budget-to-quality curves, because a single endpoint pair mixes efficiency with stopping time. The abstract's "15 times fewer training instances and 78 times fewer solver calls" presents two facets of one number, and the claim would not survive if G200 reached E133's quality early in its run. This risk rests on the Group A and training-curve placeholders.
**Suggestion**: Report per-instance saving (5.2-fold) and instance saving (15-fold) separately. Make the primary result a quality-against-solver-budget curve with a quality-matched ratio, stating how many calls each recipe needs to reach a given validation value.
**Severity**: Major
**Confidence**: 5 — core expertise: budgeted evaluation; arithmetic from Table 10

### W9: Counting solver calls omits the cost of the low-fidelity tier and the unequal price of calls
**Problem**: A solver call is not a unit of cost here. From Table 10, the mean CPU per call is about 0.081 s for E133 and about 0.031 s for G200 (an estimate from a solve time at J ≈ 60), so the ratio falls from 78 to about 30. §4.7 adds 67.6 hours of local search to 86.7 hours of Gurobi, so the low-fidelity tier is 44% of E133's summed worker time. It is not the near-free tier that "microseconds" suggests. Counting both gives about 16.7-fold, on an estimated denominator. The paper reports these numbers but keeps calls as the primary measure, and the abstract uses only 78.
**Evidence Anchor**: `table: Table 10 — Gurobi calls ratio 78 against Gurobi CPU ratio about 30; §4.7 adds 67.6 hours of local search`
**Why it matters**: Multi-fidelity budgets are summed over fidelities weighted by cost. An operator pays for core-hours, not calls. The saving remains large, but the abstract's figure overstates what an operator would see by a factor of about 2.6 to 4.7.
**Suggestion**: Make total core-hours across all tiers the primary cost measure once G200's CPU is measured or retrained under Group A. Give calls as a secondary, hardware-free count, weighted by J where possible.
**Severity**: Major
**Confidence**: 4 — core expertise on cost models; relies on the manuscript's estimated G200 CPU

### W10: The retraining case for operators is untested and conflicts with the method's input design
**Problem**: The motivation in §1 and the advice in §6.1 treat each new vessel or drone as a retraining event. But §4.3 feeds the speed ratio and drone reach to the network so that one network can serve different parameters, and §5.1 trains over ranges of drone speed and endurance. Within those ranges a new vessel speed needs no retraining. Outside them, no shifted retraining was run. The "within a day" figure comes from one training on the distribution for which the recipe was developed. Its settings came from earlier development runs (§4.6), and that campaign is not counted as cost, nor is the validation needed to choose a checkpoint. An operator on a new distribution may need to retune the J ≥ 60 threshold and the ramp. A cheaper response than retraining, instance-level adaptation or fine-tuning in the manner of efficient active search, is also not considered.
**Evidence Anchor**: `text: §6.1 "An operator can therefore retrain for a new vessel speed, drone endurance or service region within a day on a single workstation"`
**Why it matters**: The paper's closing advice ("where retraining has to be affordable") rests on this. If the demand for retraining is lower than assumed, or if a retrained recipe needs retuning, the operational value of the saving changes. A transfer test also separates a recipe that is robust from one fitted to its development distribution.
**Suggestion**: Run one shifted retraining with settings frozen, such as clustered targets, which Table 7 shows to be the weak case. Compare it with E133 and G200 unmodified and with fine-tuning from E133. Report development runs as a disclosed one-off cost. State which operating changes actually require retraining. I defer to maritime reviewers on how often such changes arise.
**Severity**: Major
**Confidence**: 3 — cost reasoning is within expertise; maritime practice is outside it

### W11: The "wider implication" restates known theory that the paper does not cite
**Problem**: U_q is a majoriser of f at q: it dominates f everywhere and touches it at q. The code uses that word (`expert_majorizer`), but the manuscript does not. Accepting a route with U_q(π) < f(q) is the descent step of majorise–minimise (Hunter and Lange, 2004), and it needs domination and contact, not tightness. Surrogate search with exact verification is the setting of trust-region model management (Alexandrov et al., 1998), and of keeping the high-fidelity model in the loop (Peherstorfer et al., 2018). Judging designs by order and not by precise value is the premise of ordinal optimisation (Ho et al., 1992). §2.3 does not mention these lines of work, and neither the bibliography nor the manuscript text contains these terms.
**Evidence Anchor**: `text: §6.3 "a signal with a guaranteed direction of error and a verification step can substitute for an accurate one"`
**Why it matters**: A referee from optimisation would note that this is not new. The novelty claim in §1 and §2.4 is carefully bounded and survives, because the new element is the use in policy training. Note that MM theory covers the search step, not the REINFORCE step, so it explains half the design. That boundary is where the paper's actual question lies.
**Suggestion**: Name the construction as a majoriser, connect the screening rule to MM descent and trust-region acceptance, and restate the implication as a transfer of a known principle into policy-gradient training, which is what remains unexplained. The citations give the paper a stronger theoretical base at little cost.
**Severity**: Minor
**Confidence**: 5 — core expertise: surrogate and multi-fidelity optimisation

### W12: The extension claim states a condition for validity, not for usefulness
**Problem**: Feasibility of a frozen lower-level solution gives a valid bound. Whether the bound can train a policy also depends on slack and ranking quality, which depend on the instance structure. The sentence also leaves out additive order dependence, which the table needs and which §6.4 only partly states. No second problem was tested.
**Evidence Anchor**: `text: §7 "because all it requires is that a solution for one route remain feasible for others"`
**Why it matters**: On a problem with stronger coupling, a valid bound may be too loose to learn from. The claim could mislead readers about transferability.
**Suggestion**: Reword as a condition for validity, and add that usefulness is an empirical question. State the conditions: order-independent lower-level feasible set and a cost additive in pairwise terms. A one-problem test on the close-enough TSP, where frozen waypoints give the same construction, would be a cheap and informative check.
**Severity**: Minor
**Confidence**: 4 — general reasoning applied to the stated structure; no second problem examined

### W13: The Remark's "only" is incorrect, and a tighter certified bound exists
**Problem**: Sorties can be mixed per target without loss of validity. Each anchor solution supplies, for target i, a complete sortie tuple (s, l, t1, t2, τ), and by §3.2 any such tuple is feasible for any order. Pick one tuple per target, keeping s and l together, and set the legs by distance. That point is feasible, so its value bounds f(π). For a fixed π, the best choice over m tables per target is a shortest path through a layered graph, computed in O(Jm²) without a solver. It is never worse than min_q U_q(π), which is the special case of using the same table throughout. The Remark is right about entry-wise minima, which break the pairing of s and l, but wrong that complete routes are the only way to combine tables.
**Evidence Anchor**: `text: §4.2 Remark "Several tables can be combined only by minimising over complete routes"`
**Why it matters**: The statement needs correcting. It also bears on §6.3, since the 0.32% result applies to route-level minima only. A further tier is possible by reoptimising one sortie at a time with neighbours fixed, again producing feasible points and certified bounds. Together these form a fidelity ladder between the table and the SOCP. Whether they tighten the bound much is untested: tables from nearby solutions may share similar sorties. The construction would also change the local search, since moves on a table are no longer simple arc sums.
**Suggestion**: Correct the Remark. Measure the tightening of the layered bound on stored tables, which needs no training. If it is large, it supplies the real dose that W3 asks for.
**Severity**: Minor
**Confidence**: 5 — direct argument from the manuscript's constraints; numerical benefit unknown

### W14: The 33.6% excess is the best case among policy routes
**Problem**: §6.3 uses this figure as the looseness of the bound, but the Table 9 caption defines it on the lowest-bound route of each instance, which is selected for having a low bound. The repository shows the difference on earlier checkpoints: mean slack over POMO routes of 64.8% and 38.2%, against 39.3% and 20.7% for the selected route, and about 100% at J = 100 for S500-E100. `docs/recipe_diversity_20261022.md` gives 40% to 100%.
**Evidence Anchor**: `table: Table 9 — Bound excess 33.6 at epoch 133, measured on the lowest-bound policy route only`
**Why it matters**: The paper's argument is that a loose bound suffices, so understating looseness weakens its own point.
**Suggestion**: Report mean and selected-route slack by J in Table 9, or a by-J companion table.
**Severity**: Minor
**Confidence**: 5 — direct reading of the caption and repository tables

---

## Detailed Comments

### Assumption Audit
- **Explicit assumptions**: The bound needs an optimal anchor and order-independent sortie constraints. Both hold and are stated, and this assumption withstands scrutiny. The claim that the bound is accurate near the anchor is explicit but unmeasured (W1).
- **Implicit assumptions**: (a) Validity is what matters about the signal, whereas rank quality may do the work (W5). (b) Solver calls are the cost (W9). (c) Training cost recurs with each operating change (W10). (d) Route overlap measures useful diversity (W6). (e) The gap to G200 comes from the signal, whereas the comparison also varies architecture and instance count, which §5.1 admits.
- **Paradigmatic assumptions**: The field's habit of benchmarking against a single trained baseline at its endpoint hides the efficiency question. Expensive-objective research relies on anytime comparison (W8). The field also treats reward as a scalar given by the problem. Here the reward is a design choice with a bias, which deserves the same scrutiny as the architecture.

### Cross-Disciplinary Connections
- **Parallel research**: The conclusions in §6.3 have antecedents (W11). Multi-fidelity RL gives working forms: using the lowest fidelity that is still informative (Cutler et al., 2015) and control-variate estimators (Khairy and Balaprakash, 2024). Model management remains a recognised challenge in surrogate-assisted evolutionary computation (Jin, 2011). In my limited searches I found no neural routing method using an analogous certified surrogate, which is consistent with the gap claim but does not establish it.
- **Borrowing opportunities**: The near-miss and refresh rules are acquisition functions built by hand. The near-miss rule selects by smallest relative gap only. Active learning would frame the choice as expected information per unit of cost, and it would give a principled replacement for the rate schedule and for the 23.7% refresh yield in §5.6. The refresh resembles a trust-region re-centring and could be analysed as one. The anchor could also be chosen by exploration, not only predicted quality, because which anchor is solved determines the region where the reward is tight.
- **Methodological borrowing**: Dose–response on fidelity (W3), a free-proxy control (W5), budget-to-quality curves (W8), and gradient-level diagnostics (W1).

### Practical Impact
- **Real-world application**: The clearest practical statement is the seconds-versus-minutes tradeoff in §6.1, which follows from Table 7. A further tradeoff is worth stating: the saving comes at a quality cost that a planner must price. A gap of 2.6% in mission time may or may not outweigh one training run avoided, depending on how many missions are flown.
- **Implementation feasibility**: A break-even count would make this concrete: the number of planned instances beyond which a trained policy beats per-instance metaheuristics in total compute. The paper has the timing data for this. Test time also depends on 16 parallel solver workers (§5.4), so deployment is not solver-free. The solver licence at the planning site is a practical dependency that I cannot size.
- **Stakeholders**: Maritime reviewers should judge whether the deterministic model and the retraining triggers match operating practice.

### Broader Implications
- **Ethical dimensions**: No material ethical issue arises from this computational study, and I do not force one.
- **Social impact**: The humanitarian operator is mentioned in §1 as motivation with no analysis. This mention needs either support or removal. I rate this as a remark, not a finding.
- **Future directions**: The most valuable follow-up is a theory of when a policy can be trained on a majoriser. That would cover what bias a majoriser induces in a policy gradient, whether pessimism around an anchor must cause concentration, and how anchor selection can counteract it. The paper is well placed to pose this question.

---

## Cross-Disciplinary Reading Recommendations
- Hunter and Lange (2004), *A Tutorial on MM Algorithms*: names the construction and the descent logic behind the screening rule.
- Alexandrov, Dennis, Lewis and Torczon (1998), *A trust-region framework for managing the use of approximation models in optimization*: approximation models of varying fidelity with convergence to the original problem.
- Peherstorfer, Willcox and Gunzburger (2018), *Survey of Multifidelity Methods in Uncertainty Propagation, Inference, and Optimization*: the adaptation, fusion and filtering taxonomy gives the method a vocabulary.
- Jin (2011), *Surrogate-assisted evolutionary computation: Recent advances and future challenges*: model management.
- Ho, Sreenivas and Vakili (1992), *Ordinal optimization of DEDS*: finding good designs without precise value estimates.
- Loshchilov, Schoenauer and Sebag (2010), *Comparison-Based Optimizers Need Comparison-Based Surrogates*: rank-based surrogates, relevant to W5.
- Kumar, Zhou, Tucker and Levine (2020), *Conservative Q-Learning for Offline Reinforcement Learning*: conservative value estimates, relevant to W7.
- Khairy and Balaprakash (2024), *Multifidelity Reinforcement Learning with Control Variates*: control-variate estimators, relevant to W2.
- Cutler, Walsh and How (2015), *Real-World Reinforcement Learning via Multifidelity Simulators*: when to trust a lower-fidelity source.
- Ajalloeian and Stich (2020), *On the Convergence of SGD with Biased Gradients*: bias relative to gradient norm decides whether SGD reaches the solution or a neighbourhood.
- `[UNVERIFIED]` Search lead: layered-graph or shortest-path evaluation of a fixed order with discretised waypoints in the TSP-with-neighbourhoods literature, relevant to W13.
- `[UNVERIFIED]` Search lead: the term "false optima" in surrogate-assisted evolutionary computation, for the risk that minimising a surrogate moves the search to points that are poor on the true objective.

I confirmed the bibliographic records and abstract-level content of the listed works by web search and did not read their full texts, so the statements above stay at abstract level.

~~~yaml suggested-references
- title: "A Tutorial on MM Algorithms"
  authors: ["Hunter, D. R.", "Lange, K."]
  year: "2004"
  doi: "10.1198/0003130042836"
  arxiv_id: null
- title: "A trust-region framework for managing the use of approximation models in optimization"
  authors: ["Alexandrov, N.", "Dennis, J. E.", "Lewis, R. M.", "Torczon, V."]
  year: "1998"
  doi: "10.1007/bf01197433"
  arxiv_id: null
- title: "Survey of Multifidelity Methods in Uncertainty Propagation, Inference, and Optimization"
  authors: ["Peherstorfer, B.", "Willcox, K.", "Gunzburger, M."]
  year: "2018"
  doi: "10.1137/16M1082469"
  arxiv_id: "1806.10761"
- title: "Surrogate-assisted evolutionary computation: Recent advances and future challenges"
  authors: ["Jin, Y."]
  year: "2011"
  doi: "10.1016/j.swevo.2011.05.001"
  arxiv_id: null
- title: "Ordinal optimization of DEDS"
  authors: ["Ho, Y.-C.", "Sreenivas, R. S.", "Vakili, P."]
  year: "1992"
  doi: "10.1007/bf01797280"
  arxiv_id: null
- title: "Comparison-Based Optimizers Need Comparison-Based Surrogates"
  authors: ["Loshchilov, I.", "Schoenauer, M.", "Sebag, M."]
  year: "2010"
  doi: "10.1007/978-3-642-15844-5_37"
  arxiv_id: null
- title: "Conservative Q-Learning for Offline Reinforcement Learning"
  authors: ["Kumar, A.", "Zhou, A.", "Tucker, G.", "Levine, S."]
  year: "2020"
  doi: null
  arxiv_id: "2006.04779"
- title: "Multifidelity Reinforcement Learning with Control Variates"
  authors: ["Khairy, S.", "Balaprakash, P."]
  year: "2024"
  doi: null
  arxiv_id: "2206.05165"
- title: "Real-World Reinforcement Learning via Multifidelity Simulators"
  authors: ["Cutler, M.", "Walsh, T. J.", "How, J. P."]
  year: "2015"
  doi: "10.1109/TRO.2015.2419431"
  arxiv_id: null
- title: "On the Convergence of SGD with Biased Gradients"
  authors: ["Ajalloeian, A.", "Stich, S. U."]
  year: "2020"
  doi: null
  arxiv_id: "2008.00051"
~~~

---

## Questions for Authors

1. What is the rank correlation and the gradient-level agreement between the bound-based reward and the exact reward at E133, by J, and does either predict where the gap to G200 is largest?
2. How does a zero-solve proxy compare with the bound in rank quality, and what would you conclude about "certified" if it ranks about as well?
3. Is the gap to G200 mainly a shift in the whole route distribution or a loss of spread at each J, and would you still name diversity as the main source if it is a shift?
4. G200's 14.97 calls per instance implies about 15 routes per instance in training, although the mean J is 60 and §4.7 says cost grows with J. What was its training P, and did E133 use the same one? This bears on both the budget comparison and the overlap comparison.

---

## Minor Issues

### Language / Grammar
- No item in my remit.

### Citation Format
- No item in my remit.

### Figures and Tables
- Table 9 would be more informative with slack by J (see W14).

### Layout
- No item in my remit.

### Description of the method
- §4.4 says the near-miss rule fires "With a small probability". In `src/JointSelfImprovementTrainer.py`, eligible instances are sorted by relative gap and the first `ceil(rate × batch_size)` are attempted. This is a batch quota on the smallest gaps, not an independent draw per instance. Align the text with the code.

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Rubric Dimension 1; Card #4 "whether existing theory explains the conclusions" | MEETS | `text: §2.4 "Our claim concerns its use as the main training signal for a DRL policy."` | The novelty is bounded to use in training and survives known theory. The wider implication restates known principles (W11). | Limited to abstract-level reading of adjacent fields | No: repair is a framing change |
| Methodological Rigor | Rubric Dimension 2 | NOT_ASSESSED | — | Reviewer 1's remit | Ablation design covered under W4 only | No from this seat |
| Evidence Sufficiency | Rubric Dimension 3; Card #4 focus 1 and 2 | PARTLY_MEETS | W1, W3, W6 anchors | The experiments support the results reported. The explanations of why they occur lack the evidence named, and counter-evidence in the repository is omitted. | Repository figures come mostly from earlier checkpoints | Yes: requires new measurements |
| Argument Coherence | Rubric Dimension 4, limited to explanation to implication | PARTLY_MEETS | W2, W5, W12 anchors | The step from valid bound to usable training signal, and then to other problems, is not established. | Internal-consistency audit is the Devil's Advocate's remit | Yes: §6.3 needs rewriting |
| Writing Quality | Rubric Dimension 5 | NOT_ASSESSED | — | Outside my focus; the text was clear enough to review | none identified | No |
| Literature Integration | Rubric Dimension 6, limited to adjacent fields | PARTLY_MEETS | `absence: §2.3 — expected multi-fidelity, trust-region or majorise–minimise work; checked §1, §2.3, §2.4, §6.3, refs.bib` | The expensive-objective strand covers neural methods and omits the adjacent theory. | Coverage audit is Reviewer 2's remit | No: inexpensive to repair |
| Significance & Impact | Rubric Dimension 7 | PARTLY_MEETS | W8, W9, W10 anchors | A large real saving is probable, but the size and operational value are not yet shown on a quality-matched, full-cost basis. | Maritime practice is outside my expertise; depends on queued runs | Yes: headline and managerial claims rest on it |

The recommendation of Major Revision rests on three unresolved criteria, none of which is offset by the strength of Proposition 1. Evidence Sufficiency needs measured fidelity on E133 and a real dose test. Argument Coherence needs §6.3 and §6.4 rewritten around that evidence. Significance needs a quality-matched, full-cost comparison. All three are repairable, and the diagnostics in W1 and W6 need little compute.
