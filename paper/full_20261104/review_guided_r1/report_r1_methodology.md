# Peer Review Report

## Manuscript Information
- **Title**: Deep Reinforcement Learning with Sparse Exact Feedback for the Carrier–Vehicle Traveling Salesman Problem
- **Manuscript ID**: none assigned (`paper/full_20261104/phase4_draft/main.tex`)
- **Review Date**: 2026-11-06
- **Review Round**: Round 1, guided mode

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 1 (Methodology)

### Reviewer Identity
Computational-experiment methodologist in combinatorial optimisation, specialising in the design of algorithm comparisons and the reproducibility of DRL evaluation.

### Review Focus
Whether the 78-fold saving can be attributed to the certified bound; test contamination, single seed, checkpoint selection and interval validity; fairness of the time-budget comparison; whether the ablation design can answer its questions. Particular attention goes to correlational results written as causal. CVTSP literature and real operating settings are outside this seat.

---

## Overall Assessment

### Recommendation
- [ ] **Accept**
- [ ] **Minor Revision**
- [x] **Major Revision** — new control arms, seeds and re-accounting are needed, and re-review is required
- [ ] **Reject**

### Confidence Score
4 — core expertise in comparison design; the seat could also audit project files, which a real referee could not.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Calibration Status
`NOT_CALIBRATED`

### Summary Assessment
The paper trains a POMO-type policy mainly on a certified table bound with about 2.89 exact solves per instance and compares the resulting checkpoint E133 with G200, a POMO policy trained on exact rewards. Measurement and reporting are careful: every table number I checked reproduces from the raw files, paired per-instance intervals are valid, and unfavourable results are reported. The design, however, does not support the paper's inferences. No arm isolates the bound, because the two models differ in network, batch size, learning rate and number of training starts, and the planned ablations leave this open. The 78-fold figure is a cumulative count at a checkpoint chosen in hindsight, against an arbitrary baseline endpoint, with a 2.6% quality deficit on held-out data. Counting the run and selection that actually produced E133 gives about 22-fold, and a rough equal-quality estimate is about one order of magnitude. Causal statements on route diversity, loose bounds and transfer ability go beyond, and in part against, the project's own diagnostics. All results rest on one seed. The core idea survives and most repairs are clear, so I recommend Major Revision.

---

## Strengths

### S1: The baseline difference is declared as a recipe comparison
The manuscript states that the two models differ in architecture and that the comparison is between recipes. It reports its largest gap prominently. This disclosure is incomplete (W6), but the framing is the honest one, and later causal sentences depart from it rather than the other way round.
**Evidence Anchor**: `text: §5.1 "the comparison in this paper is between complete training recipes"`

### S2: Paired per-instance gaps, with the choice of metric justified
Gaps are computed instance by instance on identical instances, and the paper explains why it prefers this to a ratio of means. I recomputed the intervals of Tables 4 and 6 and the unaugmented generalisation mean from the per-instance files. All match. A 4,000-resample percentile bootstrap agrees with the normal interval to within about 0.03 points even for the tied Real529 data (30% exact ties) and for n = 25 at J = 200 ([−1.78, −0.69] against [−1.78, −0.72]).
**Evidence Anchor**: `text: §5.1 "We prefer the per-instance mean because a ratio of mean costs is dominated by the largest instances."`

### S3: Outcomes are exact, under a common test protocol
Every reported cost is an exact SOCP value, so test quality carries no surrogate error. Timing was run one model at a time on the same machine. The file values are 0.775 s and 0.783 s wall, with 257.4 and 259.9 routes per instance, so "the same test time" holds.
**Evidence Anchor**: `text: §5.1 "Costs are therefore exact in all tables."`

### S4: Checkpoint selection stayed on validation and went against the authors' interest
The validation procedure chose E133 over E150, although E150 had already looked slightly better on Real529 (ratio of means +0.065% against +0.120%). This argues against test-driven selection. Validation and test500 share no instances (0 common fingerprints among 2,000 and 500).
**Evidence Anchor**: `text: §5.1 "The selection used no test data."`

### S5: Solver counts are exact, and they reconcile
G200's 299,437,760 equals the sum of the last logged row per epoch in its `metrics.csv` (solves in reruns of restarted epochs are excluded, so spending was in fact about 338.5 million). E133's 3,839,970 equals the sum of the four purposes in Table 8. Near-miss count, refresh acceptance (162,888 of 686,842 = 23.7%) and every row of Table 9 reproduce.
**Evidence Anchor**: `table: Table 8 — Total row, 3,839,970 calls and 2.89 per instance`

### S6: The bound is audited at run time
Logged failures are zero across all 200 epochs, and the code also checks the bound on target solves, which is more than the text claims.
**Evidence Anchor**: `text: §4.2 "The final run completed without a stop."`

### S7: Metaheuristic budgets are matched per instance and an unfavourable result is reported
Budgets are per-instance multiples of E133's measured cost, and workers are pinned to P-cores. The paper reports that long budgets beat its method. W5 concerns the interpretation, not this design choice.
**Evidence Anchor**: `text: §5.4 "Each metaheuristic was given a multiple of the single-core time that E133 needs on that instance."`

---

## Weaknesses

### W1: No arm identifies the contribution of the bound
**Problem**: The Abstract and Conclusion credit the bound with the result, but the only comparator differs in many other factors. Network: 128-dim, 6 layers, 1.27M parameters against 256-dim, 4 layers, 2.44M. Batch: 512 against 64. Learning rate: constant 5e-5 against 1e-4 cosine. Training starts: J/4 against J (W6). E133 also has imitation learning, a gated reference policy, expert search and an edge predictor. None of the planned groups removes the dense table reward. Group A keeps the G200 network, so it inherits the confound. Group B keeps the table for the reward (the driver swaps only the search task and sets LKH and refresh to off). Group D also keeps it. The candidate arms `no_il`, `no_exact_expert` and `no_search` in `scripts/train_ablation.py` were stopped or never started, and they are not in the paper.
**Evidence Anchor**: `absence: §5.7 ablation groups — expected an arm that removes or replaces the table reward while holding network, batch size, starts and imitation fixed; checked §4.5, §5.1, §5.6, §5.7, §6.3, §7 and scripts/train_ablation.py`
**Why it matters**: The low call count is largely true by construction, because the table replaces solves. The open question is whether the certified bound buys quality per solve, relative to any other cheap signal. As designed, the paper cannot exclude that imitation of verified experts does the work, that a larger network does it, or that an uncertified surrogate such as Euclidean tour length would do equally well. The Conclusion's "It does not" and "this bound can supply most of the training signal" are therefore unsupported as causal statements.
**Suggestion**: Add arms on E133's network and batch size. (a) Exact reward at P = J/4, stopped at E133's budgets. (b) The full method without the dense term, leaving imitation and sparse exact feedback. (c) U_q replaced by an uncertified cheap surrogate, keeping exact verification. Run each with several seeds. If these are not run, reword every causal claim as a statement about the recipe.
**Severity**: Critical (repairable: new arms or a narrowed claim)
**Confidence**: 4 — core expertise in ablation design; code read directly

### W2: The 78-fold ratio does not cover the procedure that produced E133
**Problem**: The paper selects E133 from a 200-epoch run, but Table 10 charges only epochs 1 to 133. The stated procedure could not stop at 133, since the choice relied on later epochs. From `metrics.csv`, the full run used 2,000,000 instances and 6,239,185 solves, plus 400,000 in-training validation solves. Selection evaluated 56 recipe checkpoints (E10–E200 by tens, then 36 refined epochs) with full POMO on validation2000. That is up to 400 × (20+40+60+80+100) = 120,000 solves per checkpoint, about 6.72 million in all, as an upper bound because duplicate routes are merged. The total is about 13.4 million.
**Evidence Anchor**: `table: Table 10 — row Gurobi calls in training, 299,437,760 against 3,839,970, ratio 78`
**Why it matters**: The ratios become 10-fold in instances, 48-fold in training solves, 45-fold with in-training validation and about 22-fold with selection. Wall time is 22.3 h of epochs plus 6.0 h of validation, or 28.3 h, and selection adds up to about 4.3 h. This contradicts the Introduction's claim that training "finishes in less than a day", repeated as "within a day" in §6.1. G200 used no comparable selection. Development runs that fixed the recipe are also excluded; this is defensible for an operator but should be stated.
**Suggestion**: Fix a stopping and selection rule in advance, and report the cost of the whole pipeline, including selection. Alternatively, show a prospective early-stopping rule that would have stopped near E133 without seeing later epochs, and evaluate it on the extra seeds.
**Severity**: Major
**Confidence**: 5 — direct recomputation from logs, except the selection count, which is derived from the protocol

### W3: The saving is a two-point comparison at unequal quality against an arbitrary endpoint
**Problem**: G200 stopped at epoch 200 by choice, and its log continues to about 280 epochs, so the ratio scales with that choice. E133 is 2.60% worse on test500 without augmentation. "78 times fewer" does not measure the cost of reaching the same quality. The project already holds curve data that the paper does not use. An exact-reward retraining with the G200 recipe reaches 18.089 on validation2000 at 13.2M solves (E9), against 17.703 for E133. So the equal-quality ratio is above 3.4-fold and below 78-fold, and the manuscript gives no estimate within that range.
A rough estimate is possible. I adjusted G200's training score to J = 60 by regression (slope 0.112 per unit of J over E150–200) and applied an offset of 0.08–0.10 to validation, seen at two points. G200 then crosses E133's validation level at around epochs 20–30, which is 30–45M solves and 2–3M instances. That suggests about 8 to 12-fold in calls and 1.5 to 2.3-fold in instances relative to Table 10's E133 budget, and less relative to the W2 pipeline. This is an estimate from a different statistic and must not be reported as a measurement.
**Evidence Anchor**: `text: §1 "The method uses 15 times fewer training instances and 78 times fewer solver calls, with no change in test time."`
**Why it matters**: A saving of two orders and a saving of one order lead to different conclusions, and the paper's headline may be the wrong one. Reporting quality against budget is the established way to compare anytime training.
**Suggestion**: Make the primary comparison a quality-against-budget curve, with cost to reach stated quality targets, by size. Report the existing equal-solves results as a matching-budget point (the project files give +4.44% for that arm against E133 on validation2000). Note the limit of the planned Group A run. On past speed of about 80 minutes per epoch it reaches roughly 16–18 epochs in 24 hours, so it will likely give only a lower bound on cost-to-E133 quality. Evaluating G200-recipe checkpoints at epochs 20–50 would close the interval.
**Severity**: Major
**Confidence**: 4 — core expertise; the equal-quality figure is an explicit rough estimate

### W4: Reported intervals cover test instances only, and all results come from one seed
**Problem**: Each 95% interval is conditional on one trained E133 and one trained G200. Between-run variance is unknown, and it is the relevant variance for a claim about a method. The plan adds two E133 seeds and no G200-recipe seeds. Claims such as an interval excluding zero on Real529, or the sign at J = 200, are claims about two particular networks.
**Evidence Anchor**: `text: §6.4 "Second, the results rest on one training seed."`
**Why it matters**: Validation differences that shaped the recipe are of the order of 0.002 to 0.08. Without a seed standard deviation, nobody can tell whether the size pattern, the Real529 near-parity or any ablation difference exceeds run-to-run noise. If the seed standard deviation turns out near the 0.2–0.7% scale of the gaps, the +0.21% and +0.72% figures stop being estimates of anything general.
**Suggestion**: Use at least five seeds per recipe, including the baseline recipe at comparable budgets, and fix the selection rule across seeds. Report the seed-level mean with a t-interval on seeds (or a two-level bootstrap), and keep per-instance intervals as within-model precision.
**Severity**: Major
**Confidence**: 5 — core expertise in DRL evaluation

### W5: The time-budget conclusion does not hold across budgets or families
**Problem**: Table 7 omits the 1× budget, which is in `summary.md`. From the per-instance traces I compared ILS with E133, using the same trace rule as `scripts/summarize_metaheuristics.py`. On 150 Poikonen clustered instances, ILS is already better at 1×: mean gap −0.81% [−1.13, −0.48], 112 wins and 37 losses. At 3× it is −1.75% [−2.07, −1.43], 139 and 9. Over all 529 instances at 3×, the paired mean favours E133 by a small margin (+0.35% [+0.10, +0.59]), yet ILS wins more instances, 249 to 180. At J = 100 under 3×, ILS has a mean gap of −1.37%. The "seconds against minutes" framing is also off: the 1× budget has mean 10.3 s and median 1.84 s, so the 10× budget on a median instance is about 18 s of one core. E133's 0.78 s wall needs 16 workers and a GPU, whereas the metaheuristics are compared on single-core time and are not parallelised. Lower budgets are truncated traces of one 10× run on seed 0, not separate runs.
**Evidence Anchor**: `text: §6.1 "Where the plan must be available in seconds, the learned policy is the better option at the time budgets we tested"`
**Why it matters**: This sentence is the paper's advice to practitioners. As written it is true only of the pooled mean, and false for a large family at every budget tested. It also rests on means alone, while the paper uses win counts elsewhere.
**Suggestion**: Add the 1× column, paired gaps with intervals against E133 and win counts by family. State the recommendation conditionally by family and size. Report wall time on one defined resource (for example 16 cores, with parallel restarts for the metaheuristics, and with and without the GPU). Run several seeds. Add a table-guided metaheuristic if the aim is to compare the best available solver-sparing search.
**Severity**: Major
**Confidence**: 5 — recomputed from `ils.jsonl`, `vns.jsonl` and the timing records

### W6: The baseline description is wrong on training starts and silent on learning rate
**Problem**: G200's log shows about 15 solves per instance at mean J about 60, which is J/4 starts; `scripts/train_pomo_gurobi_budget.py` records this inference and sets `pomo_divisor` to 4 for that reason. E133 trains with P = J (mean about 60.5 routes). G200's `params.json` shows learning rate 5e-5 with no decay in the range used, which the paper does not mention.
**Evidence Anchor**: `text: §5.1 "It uses the same instance generator and POMO start rule as the proposed method"`
**Why it matters**: A reader who reproduces G200 from the text with J starts would pay about four times the calls, so the primary metric is not reproducible as described. It also adds an undeclared difference to a comparison that the paper uses to explain route diversity (W7), because a policy trained on fewer starts is then decoded from J starts at test.
**Suggestion**: Correct the sentence, and give G200's starts, learning rate, schedule, seed and restart history. Confirm the J/4 setting from the G200 run configuration, since `params.json` shows `pomo_divisor` as null and the setting is inferred from the log.
**Severity**: Major
**Confidence**: 4 — logged counts are unambiguous; the mechanism is inferred

### W7: Route diversity is presented as the cause of the gap without identifying evidence
**Problem**: The evidence is one overlap pair (0.62 against 0.53) from a different checkpoint (E163), at one size, with no interval, and between models that differ in everything listed in W1. The chosen size shows the smallest contrast in the project files: J = 20 is 0.638 against 0.476 and J = 60 is 0.670 against 0.546. The cited failed attempts do not test the hypothesis, because overlap barely moved: the multi-table branch left it unchanged, and lower imitation mass gave 0.587 against 0.598 at J = 100. A treatment that does not change the mediator cannot show anything about the mediator. Project data also point the other way. In `outputs/reward_diversity_diagnostic_20261015/report.md`, the predecessor model S500 at epoch 50 has lower overlap than G200 (0.404 against 0.526) but a worse best-of-POMO (17.926 against 17.381) and a worse mean-of-POMO (20.624 against 20.156). Per-route quality therefore also lags. Those are S500 models, not E133.
**Evidence Anchor**: `text: §6.4 "The evidence points to route diversity."`
**Why it matters**: The Conclusion directs future work to this explanation, using the phrase "appears to be the main source of the remaining gap". Competing explanations are not excluded: fewer instances, bound bias (W9), network or the J/4 training. The larger gain from augmentation (2.60% down to 1.13% on test500) fits diversity but also fits other explanations.
**Suggestion**: Decompose the gap into mean route quality and spread, by size, for E133 and G200. Report overlap for all sizes with intervals. To test causation, use a from-scratch manipulation that demonstrably moves overlap, and check that the gap follows. Until then, write "one candidate explanation".
**Severity**: Major
**Confidence**: 4 — design logic is firm; project diagnostics concern predecessor checkpoints

### W8: "Why a loose bound suffices" rests on a test with almost no dose
**Problem**: The tightening tested was 0.32% against a bound excess of 33.6%, that is about one hundredth of the slack. It was applied only on large-instance batches with refresh, in a 30-epoch continuation from an already selected checkpoint. A null from such a small dose cannot show that tightness has low marginal value. The method also has not been shown to suffice: it is 2.60% behind on held-out data, and looseness remains a candidate cause. "Accurate nearby" is asserted, and the manuscript contains no measurement of it. The promise that Group B "will quantify this claim" cannot be met (W10).
**Evidence Anchor**: `text: §6.3 "A much tighter bound is thus not easy to obtain from nearby solutions, and on the evidence so far it is not the limiting factor."`
**Why it matters**: The section draws a general design lesson for expensive objectives from this result. The stronger evidence the project holds is not reported: the 2×2 random-subset reward study raised the expected directional agreement with all-exact advantages from about 0.6 to about 0.85–0.95, at three to four times the solver cost, and showed no gain by epoch 20 (its later epochs were slightly worse without LKH and level with it). That study is short, but it has a real dose.
**Suggestion**: Report bound error as a function of distance from the anchor. Run a from-scratch dose-response on reward accuracy (for example 0, 4, 16 and all routes solved exactly at small J) with seeds. Limit the wider lesson to what that shows, and retitle the section if the result stays open.
**Severity**: Major
**Confidence**: 5 — core expertise; numbers from `docs/recipe_diversity_20261022.md`, `docs/random_subset_reward_experiment_20261015.md` and Table 9

### W9: The hybrid reward is a biased mixture and the paper does not say so
**Problem**: In the a^p equation, one route has its exact cost and all others have an upper bound that exceeds exact cost by a large margin. The solved route therefore looks better by construction, and it is always the top advantage. In the zero-training diagnostic (200 instances, S500 at E100 and E365, not E133), its z-score is 3.79 and 3.37, against 1.27 and 1.30 if all routes were exact, and 5.51 and 4.67 at J = 100. Cosine with the all-exact advantage is 0.61 and 0.67 (0.53 and 0.54 at J = 100), Spearman between bound and exact cost is 0.61 and 0.68, and the truly best route receives a negative advantage in 8% and 12% of instances (10% and 20% at J = 100).
**Evidence Anchor**: `equation: §4.5 definition of the dense advantage a^p — c^p is exact for the lowest-bound route and a bound for every other route`
**Why it matters**: §6.3 says looseness affects only "the ranking of candidates and the variance of the reward". The diagnostic shows a directional bias, which is a different and stronger effect. The bias concentrates credit on one route, so it is a plausible mechanism for reduced route diversity and for the growth of the gap with size. It may not matter in the end, given the random-subset result in W8, but it must be stated.
**Suggestion**: Describe the estimator and its bias in §4.5, report these diagnostics for E133 by size, and test the variant that scores all routes by the bound alone (z-score 2.26, cosine 0.62), which costs no additional solves beyond those already made.
**Severity**: Major
**Confidence**: 4 — code and diagnostic read directly; values are for predecessor checkpoints

### W10: Group B cannot separate the table's time saving from a change in search power
**Problem**: B is stated to measure what the table saves. In the driver, B also moves from 30 starts to one start, from bound-ranked candidates to the anchor, and turns off LKH and refresh. It stops at 100 exact solves, while one full 2-opt neighbourhood at J = 100 is 4,950 moves. Expert quality will therefore differ, so B and C differ in both time and quality, and no saving can be read off. B keeps the table for the dense reward, so it says nothing about bound tightness, contrary to §6.3.
**Evidence Anchor**: `text: §5.7 "Group B replaces table-based search with exact evaluation of candidate moves, under a budget of 100 exact solves per instance"`
**Why it matters**: The outcomes are hard to interpret in either direction. If B is worse, the cause may be the crude search and not the absence of the table. If B is equal or better at about 100 solves per instance and about four times the wall time (29 minutes per epoch against about 7), the table's role reduces to a time-for-quality trade, and §2.3's claim about the economics of screening weakens.
**Suggestion**: Compare at equal expert quality on a frozen policy: solves and time needed by exact search to match table search on the same starts, by size. Keep the from-scratch B as a secondary result, and report true solves from `exact_progress.jsonl`, since `metrics.csv` undercounts them.
**Severity**: Major
**Confidence**: 5 — driver and search code read directly

### W11: Group D changes several things, and tests a different "diversity" from the one the paper discusses
**Problem**: D sets the reference pool to one greedy route. This changes start quality (first-start greedy route rather than the best of 200 by bound), removes anchor selection, and makes the edge predictor's ranking moot, as well as removing diversity. The budget of 900 moves is likely nominal: project notes record that about 97% of sampled routes are already 2-opt optima on their own table, and in D the single start is the anchor itself. Effort therefore falls with diversity, unless LKH does most of the work. D also varies diversity of expert search starts, whereas the Limitations discuss diversity of POMO output routes.
**Evidence Anchor**: `text: §5.7 "one greedy route receives the full local-search budget of the 30 starts"`
**Why it matters**: The stated aim, to separate diversity from effort, is not met. Readers will also take D as evidence on the route-diversity explanation, which it does not test.
**Suggestion**: Hold start quality and effort fixed: for example 30 starts drawn as near-duplicates of the best route, against 30 diverse starts of matching mean bound, with the same realised move count. Report accepted moves and LKH time per instance in both arms, and use distinct terms for search diversity and route diversity.
**Severity**: Major
**Confidence**: 4 — driver read directly; the 97% figure comes from project notes and has not been run for D

### W12: Ablations are compared at the wrong point, with one seed and no decision rule
**Problem**: Comparison is at epoch 100, whereas the headline model is epoch 133. The progressive ramp finishes only at epoch 100, and the benefit arrives later. On the project's dev set, the recipe at E100 is −0.021 [−0.058, +0.015] against S500-E100, whereas at E150 it is −0.079 [−0.114, −0.044]. On validation2000 the two E100 values are 17.772 and 17.779. Any component that pays after E100 (refresh, LKH) will appear worthless. Each arm has one seed, and no threshold is set in advance for what counts as an effect.
**Evidence Anchor**: `text: §5.7 "Variants B and D train from scratch to one million instances (epoch 100)"`
**Why it matters**: Differences at this stage are near the paired noise of the validation set, and seed noise is unknown. Ablation claims would then be neither about E133 nor distinguishable from chance.
**Suggestion**: Extend at least C and D to the final selection point, or justify E100 with a pre-stated argument. Pre-register the primary metric, smallest effect of interest and seeds, and report paired differences by size.
**Severity**: Major
**Confidence**: 4 — numbers from `docs/progressive_recipe_20261020.md`, not re-derived from raw files

### W13: The operator claim was not tested
**Problem**: No experiment retrains on a shifted distribution. The recipe's thresholds (J ≥ 60, ramp over epochs 40–100, refresh after epoch 60) were set by development runs on one distribution, and the paper admits there was no formal search. Speed ratio and endurance are input features sampled over wide ranges, so a new vessel speed within those ranges requires no retraining at all, which conflicts with the premise in §1 that such a change is a change of distribution. The case that does need retraining, clustered targets where both policies are weakest, has not been tried. The time figure also inherits W2.
**Evidence Anchor**: `text: §6.1 "An operator can therefore retrain for a new vessel speed, drone endurance or service region within a day on a single workstation"`
**Why it matters**: The managerial insight and the "recurring operational expense" motivation rest on an extrapolation. The saving matters only if the recipe transfers without re-tuning, which is exactly what remains unknown.
**Suggestion**: Run the unchanged recipe on at least one shifted distribution (clustered targets are the obvious case), with the same selection rule and full cost accounting. Compare with G200 applied unchanged, to show when retraining pays. Otherwise present this as a hypothesis.
**Severity**: Major
**Confidence**: 4 — design logic; real operating settings are outside this seat

### W14: The lead quality figures rest on a non-held-out benchmark
**Problem**: The Data paragraph correctly says Real529 is not held out, and project notes show that matching G200 on it was a development target for the recipe. Training parameter ranges also derive from it. The Abstract and Conclusion nevertheless lead with +0.21% and "close to that baseline", without this caveat. Held-out data show a larger gap at matching sizes. Recomputed from the eight-view test500 file: J = 20 +0.46%, J = 40 +1.02%, J = 100 +1.39%, against Real529 +0.33% for J = 20–50 and +0.52% for J = 100.
**Evidence Anchor**: `text: Abstract "On 529 literature benchmark instances its mean per-instance gap to that baseline is +0.21"`
**Why it matters**: The instance distributions differ, so part of this difference may be real. But a reader cannot tell how much of the near-parity reflects development on the benchmark. Report order does affect how the result is read.
**Suggestion**: Lead with test500 under one common protocol, and mark Real529 as a development-era benchmark in the Abstract and Conclusion. A fresh held-out set from the Poikonen generator rules would give a clean literature-style test.
**Severity**: Major
**Confidence**: 3 — the extent of optimism cannot be measured from the available files

### W15: Causal language on transfer ability
**Problem**: The comparison has no same-network arm, and one seed per recipe. E133 is behind at every size without augmentation, recomputed as +2.21, +2.14, +1.23 and +0.94%. A shrinking gap may equally reflect how G200 degrades outside its training range under a J/4 training setting.
**Evidence Anchor**: `text: §5.3 "The results thus show that learning from the table did not cost additional transfer ability relative to learning from exact rewards."`
**Why it matters**: This is a causal null drawn from a confounded contrast, and it needs either a same-network arm or a neutral rewording.
**Suggestion**: Write "the gap to G200 was no larger beyond the training range than within it". Give the unaugmented table alongside the eight-view one.
**Severity**: Minor
**Confidence**: 5 — core expertise; recomputed from the result file

### W16: The CPU row mixes unlike quantities
**Problem**: G200's 2,580 core-hours uses 0.031 s per call. E133's own target solves, which are the same type of solve, averaged 61,582 s over 1,330,000 calls, or 0.046 s, under load across the training size range. Applying that gives about 3,850 core-hours and about 44-fold. E133's 86.7 h excludes 67.6 h of search; with it, the ratio is about 25-fold on the higher estimate and about 17-fold on the lower one. G200's 215.5 h extrapolates 64.7 minutes per epoch from epochs 184–188, and `params.json` lists 64 reward workers, not 128.
**Evidence Anchor**: `table: Table 10 — row Gurobi CPU (core-hours), about 2,580 (estimated) against 86.7, ratio about 30`
**Why it matters**: The paper already calls these secondary, so the core claim is unaffected, but the plausible range is about 17 to 44-fold rather than "about 30". E133 also takes more gradient steps (74,696 against about 39,000), so fewer instances does not mean less GPU training.
**Suggestion**: State a range and its assumptions, add a row for total CPU, and replace estimates with same-machine rates by size once the 24-hour run is done. Confirm the worker count.
**Severity**: Minor
**Confidence**: 4 — recomputed from `metrics.csv`; the 0.046 s rate was measured under different load from the 0.031 s rate

### W17: The SA row does not represent working SA
**Problem**: On Erd132 25–50, SA never improves on its initial solution in 95.8% of instances at its own 1× and 3× budgets, and in 77.8% at 10×. A note in the project already calls the stuck value suspicious. Table 7 shows the 10× row only, while the text says "all budgets". No tuning is reported for any metaheuristic, whereas E133's recipe had extended development.
**Evidence Anchor**: `table: Table 7 — row SA 10×, +5.15 on Erd132 25–50 and +3.14 overall`
**Why it matters**: The row reports a calibration failure as a method result. ILS and VNS, which carry the argument, are not affected.
**Suggestion**: Fix the initial temperature by calibration on a small tuning subset, or drop SA. Report tuning effort for all baselines.
**Severity**: Minor
**Confidence**: 5 — recomputed from `sa.jsonl`

### W18: The test-set disclosure is correct but imprecise
**Problem**: The "earlier checkpoint" is E150, which is later in training than E133. It was evaluated on test500 and Real529 mid-run on 21 October, and the per-epoch refinement over E131–E169 was then defined around it on 22 October. Eight-view test500 was then used to form the diversity explanation. Protocol text in `docs/progressive_recipe_20261020.md` had specified one final test only. S4 shows no sign of harm to selection.
**Evidence Anchor**: `text: §5.1 "We inspected test500 once for an earlier checkpoint before selecting E133"`
**Why it matters**: The primary evidence stays usable, as the validation gap (−2.23% relative to E133) agrees with test500's ratio of means, but the disclosure should be exact.
**Suggestion**: Name E150, state the order of events, and state that test500 informed the diversity analysis. Use a fresh test set for the final seeds.
**Severity**: Minor
**Confidence**: 5 — chronology from file timestamps and project notes

### W19: The learning-rate schedule is under-described
**Problem**: The "cosine schedule" is a 500-epoch cosine cut at epoch 200, inherited as a control from S500. The learning rate is still 8.4e-5 at E133 and 6.6e-5 at E200. Decline after E133 therefore happens without annealing, and the run is not a converged endpoint.
**Evidence Anchor**: `text: §5.1 "Adam with learning rate $10^{-4}$ and weight decay $10^{-6}$, and a cosine schedule"`
**Why it matters**: The text is not enough to reproduce E133, and a reader would assume a cosine ending at zero by epoch 200.
**Suggestion**: State the horizon and the values at E133 and E200.
**Severity**: Minor
**Confidence**: 5 — from `resolved_config.json` and `metrics.csv`

---

## Detailed Comments

### Title & Abstract
The Abstract's three gaps use different protocols: eight views on Real529, none on test500, eight views on the large set. See the Results comment below. "At the same test time" is verified.

### Introduction
The research question ("whether most of those calls are necessary") is answerable only at a stated quality level. As posed, it needs the curve in W3.

### Methodology / Research Design
Design type: a two-arm, one-seed comparison of complete pipelines, plus pending single-seed ablations. This fits a claim that one recipe was cheaper than another at a stated quality. It does not fit claims about why. Validation and test sets are disjoint, and test500's instance fingerprints are identical across the three result files checked.

**Pending experiments and what a wrong-way result would do**

| Pending item | Can it answer its question? | If it goes the wrong way |
|---|---|---|
| Group A (matching budget) | Yes for recipe against recipe at matching budget; no for attribution (W1). The 1.33M-instance point should be reachable; equal quality probably not within 24 h | If POMO at E133's instance budget matches E133, "15 times fewer instances" loses its quality meaning. The equal-solves point already exists and favours E133, on one run of each |
| Group B | No, as designed (W10) | If B matches C, the screening argument weakens to a time trade-off |
| Group D | No, as designed (W11) | If D matches C, diversity of search starts has no measured value, and nothing follows for route diversity |
| Two further seeds | Partly. Three seeds give a crude standard deviation; G200-recipe seeds are also needed (W4) | A seed spread near the size of the gaps removes the size pattern and the Real529 near-parity as findings |
| 24-hour curve | Yes for same-hardware time, if the proposed-method curve is added and its selection cost shown | If curves cross early on wall time, the headline holds only at long budgets |
| Sensitivity analysis | Only if it includes retraining, not only testing (W13) | If a shift needs re-tuning, the operator claim falls |

### Results / Findings
**Statistical reporting criterion** (`references/statistical_reporting_standards.md`, sections on effect size, intervals, significance and power, applied as far as a computational study allows): PARTLY_MEETS.
- Effect sizes and intervals: MEETS at instance level. Relative gaps are natural effect sizes.
- Power: DOES_NOT_MEET for ablations, since no smallest effect or seed count is justified.
- Multiplicity: not corrected across about 20 intervals. This is unimportant here, since the intervals are descriptive and most are far from zero.
- Non-independence: Real529 mixes five families (72, 60, 72, 175 and 150 instances), so the "All" row depends on the benchmark's composition. Report by family as the primary view.
- Red flags: no p-hacking pattern in the reported tests, since none are reported. Selective reporting exists in the sense of omitted informative budgets (W5).

**Arithmetic recompute**: `no_recomputable_statistics`: the manuscript reports no test statistics, p-values, degrees of freedom or integer-scale means, so none of the four bounded procedures applies. Ad hoc checks: all table sums and ratios verified, with 215.5 h being 215.7 h from the stated rate, a rounding-level difference.

**Eight-view test500 exists.** The placeholder in §5.2 says it was not run, but `outputs/pomo_gurobi_equal_budget_20261027/eval_equal_solves_test500_aug8/results.json` holds complete E133 results. Recomputed: +1.13% [+1.02, +1.25], 66/51/383 at tolerance 1e-3. By size: +0.46, +1.02, +1.32, +1.48 and +1.39%. The authors should verify this and use it, which also resolves the protocol mismatch in the Abstract.

### Discussion
§6.1, §6.3 and §6.4 exceed the evidence, as W5, W7, W8 and W13 set out. Limitations list one seed and hand-set parameters, which is good. They omit the unequal training starts and the selection cost.

### Conclusion
"It does not" should be conditional on a quality level. "Appears to be the main source" should be softened until W7 is addressed.

### Reproducibility
Method detail is strong in most respects: tolerances, budgets and search parameters are stated, and exact counts allow audit. Gaps are W6 and W19. Data and code availability is a placeholder and is not faulted here.

### Methodological Fallacies Detected
- Confounding read as causation: W1, W7, W15.
- Interpreting a null from an ineffective manipulation: W7, W8.
- Survivorship in checkpoint accounting: W2.
- Pooled result hiding a subgroup reversal, in the Simpson's-paradox family: W5.
- Overfitting to a development benchmark, disclosed but not carried into headlines: W14.
- Not found: p-hacking, leakage of test data into model selection, reverse causation.

No instruction-like text aimed at reviewers was found in the manuscript.

---

## Questions for Authors

1. Was G200 trained with J/4 starts throughout, and what do the original run configuration and logs show for workers and learning rate?
2. Would you pre-commit to a stopping and selection rule that can be applied without seeing later epochs? What budget does it give across seeds?
3. Can intermediate G200-recipe checkpoints (epochs 20–50) be produced, so that cost to reach E133's quality can be measured and not inferred?
4. What is the intended claim: that this recipe is cheaper at a given quality, or that the certified bound causes the saving? The answer determines whether the arms in W1 are required.

---

## Minor Issues

### Figures and Tables
- Table 7 uses 100(UB−BKS)/UB, whereas §5.1 defines gaps with the reference in the denominator. Add a note.
- Add the 1× column to Table 7, and add the unaugmented columns to Table 5.
- §4.7 says "Validation added 266,000 calls". State that this is 1,000 instances for two policies over 133 epochs and excludes selection on validation2000.

### Metrics
- The tie tolerance of 10^-3 is absolute, on costs of about 17 and about 330. A relative tolerance is cleaner. Summary files use another tolerance and report slightly different counts (124/159/246 against 123/160/246).
- §3.2's "about 0.03 s per call" differs from the 0.046 s measured for target solves.
- Real529 uses min(J,100) starts at J = 200, whereas the synthetic set uses 200 starts, so the J = 200 contrast in §5.4 crosses protocols.

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | `references/quality_rubrics.md` | NOT_ASSESSED | — | Outside this seat | Other seats | No |
| Methodological Rigor | `references/quality_rubrics.md` Dimension 2; Card #2 | PARTLY_MEETS | W1, W2, W4, W10, W11, W12; S2, S3, S5 | Measurement is rigorous; the design cannot answer the attribution question and the ablations cannot answer theirs | Pending seeds could change the weight of W4 | Yes: attribution and ablation design decide the revision |
| Evidence Sufficiency | `references/quality_rubrics.md` Dimension 3 | PARTLY_MEETS | W3, W5, W7, W8, W9, W14 | Descriptive results are well supported; explanatory and managerial claims are not | Some diagnostics concern predecessor checkpoints | Yes: several claims need new evidence or narrowing |
| Argument Coherence | `references/quality_rubrics.md` Dimension 4 | PARTLY_MEETS | W7, W8, W13, W15 | The recipe framing in §5.1 is abandoned by causal conclusions in §5.3, §6 and §7 | Rewording repairs part of this | Yes: conclusions exceed the design |
| Writing Quality | `references/quality_rubrics.md` Dimension 5 | NOT_ASSESSED | — | Outside this seat | Other seats | No |
| Literature Integration | `references/quality_rubrics.md` | NOT_ASSESSED | — | Reviewer 2's remit | Other seats | No |
| Significance & Impact | `references/quality_rubrics.md` | NOT_ASSESSED | — | Reviewer 3's remit | Other seats | No |

The recommendation follows from one repairable design gap (W1) and from unresolved decision-bearing criteria on accounting (W2, W3), seed uncertainty (W4), the practitioner conclusion (W5) and explanatory claims (W7 to W9, W13). All can be repaired by new arms, re-accounting or narrowing claims. Strengths in measurement do not offset them.
