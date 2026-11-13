# Issue index, guided review round 1

Working index for the dialogue. It groups findings from the five reports and is not a synthesis decision. Every item traces to the named reports. Status of each item is updated as the dialogue proceeds.

All five seats signal Major Revision. Severity is the highest given by any seat. No DA CRITICAL. `NOT_CALIBRATED`, `criteria_binding_unavailable`.

## Checked by the orchestrating session against raw files (2026-11-06)

| Claim | Result | Source |
|---|---|---|
| E133 8-view test500 exists | Confirmed. +1.13 [+1.02, +1.25], 66/51/383, ratio of means +1.05. By J: +0.46, +1.02, +1.32, +1.48, +1.39. Dated 2026-10-28 | `outputs/pomo_gurobi_equal_budget_20261027/eval_equal_solves_test500_aug8/results.json` |
| G200 used J/4 starts to epoch 200 | Confirmed from calls per instance (15.68 at mean J 62.73 at epoch 1; 1.0 x J after epoch 200). Mechanism still inferred; `params.json` shows null | `outputs/train__cvtsp_random_n100_nearest_quarter/metrics.csv` |
| Selection cost 5,175,733 unique solves | Confirmed (2,235,364 + 1,498,832 + 1,441,537). As-run total about 11.8M, ratio about 25 | `select_recipe_validation2000`, `refine_select_a`, `refine_select_b` live logs |
| Gap to reference by size | Confirmed. J=100: E133 +4.50, G200 +4.02, TSIH +0.51. J=200: +5.91, +7.09, +0.46. All: +1.35, +1.14, +0.30 at TSIH mean 20.2 s | `outputs/bks_comparison_20261023/`, `final_external_e133/real529/results.json` |
| Equal-solves arm | Confirmed. POMO at 3.84M solves is +4.44% against E133 on validation2000 and +5.67% on Real529 | `outputs/pomo_gurobi_equal_budget_20261027/equal_solves_results.md` |
| no_il run | Exists, 43 rows, queue log says `STOPPED_BY_USER`. DA's 21.80 vs 18.80 not re-derived | `outputs/ablation_20261028/` |

Not checked here: Erdoğan §4.2 quotes (R2 read the preprint), the reversal-symmetry numbers (R2's own small test; the mathematics is short and should be proved in the paper), repository diagnostics cited by R1 and R3.

## A. Factual corrections, no decision needed
- A1. G200 start rule is wrong in §5.1; add learning rate, schedule (R1 W6, DA M9). Also the wrong claim that overlap compares like with like.
- A2. §5.2 placeholder is wrong: E133 8-view test500 exists (R1).
- A3. Generalisation set uses 100 starts, not J starts (DA m2).
- A4. New-best sentence fails: rounding, LBBD provenance, Example_260 (R2 W6).
- A5. Remark 1 "only" is false; per-target recombination is valid (R2 W9, R3 W13).
- A6. Garone 2014 role, origin citation, Ren several-carriers, Vieira "all instances", depot specialisation (R2 W12 to W15, W17).
- A7. Tolerance wording, "falsify", cosine horizon, near-miss quota, test-set disclosure names E150 (R2 W10, R3 minor, R1 W18, W19).
- A8. Table 7 lacks a 1x row; §6.1 3x claim fails on clustered (R1 W5, R2 W11, DA M12).

## B. Positioning and claims, author decisions
- B1. Retraining premise conflicts with parameter inputs and is untested; the only reason to prefer E133 to G200 (EIC W1, R1 W13, R3 W10, DA M10).
  - Dialogue 2026-11-09, checked in code: both models use the same generator (`compute_random_instance_stats` on `instance/Data`) and the same 6 features incl. speed ratio and reach; speed ratio range 0.2 to 0.5. Real529 has only two parameter sets, both at the range edges (325 at ratio 0.5, MT 20; 204 at ratio 0.2, MT 1). Author confirmed: no experiment varies parameters beyond the range or retrains on a shifted distribution. §7 sentence on retraining for new speed or endurance must change.
  - Author decision 2026-11-09: options 2 and 3, no new experiment. (2) Training cost as the entry barrier when the problem itself changes (new constraints or variants mean training from scratch for any method). (3) Algorithmic contribution: one exact solution yields a valid training signal. Rewrite §1 motivation and §7; drop the retraining-for-new-parameters claim. Consequences: B3 accounting becomes load-bearing; option 2 needs the condition under which a variant keeps the table structure (links to C5 transfer sentence); B7 venue framing to revisit.
- B2. What is the claim: recipe cheaper at a quality, or the bound causes the saving? No arm isolates the bound (R1 W1, EIC W9, DA M1).
  - Dialogue 2026-11-09: author named Group B as the arm. Checked in code: `exact_search` swaps only the Expert search (`run_exact_self_improvement_task`): one start, first-improvement 2-opt, at most 100 exact solves per instance, LKH and refresh off. The anchor table is still built (`anchor_upper_bound_matrix`, ExactSearchAblation.py:130) and the RL reward stays `hybrid_pomo` (one exact target plus table bounds). So B isolates the table inside search, not the table as RL signal, and also changes search breadth (1 vs 30 starts, no LKH). B trained to E100, finished 2026-11-07 16:16 (queue line `EXIT 4B` has no exit code); wall 175,020 s vs 39,774 s for the main run to E100 (about 4.4x). Search solves go to `backend_diagnostics["exact_calls"]` and are not in metrics.csv, so B's true solve count is unknown (at most 100 per instance). No validation2000 result for B or D found yet.
  - Author decision 2026-11-09: core contribution is an outcome claim at recipe level: few exact solves give good quality. Not a causal claim that the bound alone causes the saving. Main evidence: equal-solves arm and the crossover bound. The table is described as the enabling mechanism; ablations A to D report component effects and are not offered as proof of attribution. Wording to avoid: "the bound causes", "the saving comes from the table". "Good" to be stated per evaluation set. Refines B1 option (3).
- B3. 78x accounting: unequal quality, checkpoint hindsight, selection cost, factors 5.2 x 15, core-hours (EIC W3, W5, R1 W2, W3, R3 W8, W9, DA M8).
  - Dialogue 2026-11-09, checked: `outputs/data_efficiency_20261027/data_efficiency.csv`. G200-recipe POMO from scratch (J/4 starts) on validation2000: 3.84M solves 18.488 (+4.44% vs E133 17.703); 13.22M solves (epoch 9) 18.089, still +2.2% behind E133. So the G200 recipe needs more than 13.2M solves (over 3.4x E133) to match E133; upper bound 299M (G200 E200 17.309). Group A (24 h, `pomo_divisor` 4) restarted 2026-11-09 10:22 after a failed start on 11-07 (`timeout` rejected `24h10m`); due about 11-10, will narrow the crossover.
  - Author decision 2026-11-09: combine (a) and (b). Lead with equal-budget quality (val2000 +4.44%, Real529 +5.67%, 354/110/65) and the crossover bound; report 78x as training-only at E200 with the 15x and 5.2x factors and the J/4 start rule, plus about 25x as-run including selection. Author view: performance close on Real529 (+0.21%); caveat that Real529 is not held out and test500/val2000 gaps are larger (+1.13% aug8, -2.23% val2000), so state "close" per set.
- B4. TSIH absent; learned models are 4 to 7% from reference at J >= 100 where TSIH is about 0.5% (R2 W4, W5).
- B5. Closest prior art is Erdoğan's retained-coordinate ILS, not Li (R2 W1). Gap table columns undefined and partly wrong (R2 W7, EIC W15).
- B6. G1 boundary fails: mothership problem is the same problem; three unread papers sit on it (R2 W8).
- B7. Logistics value: algorithm paper or managerial paper; break-even; seconds setting; "pays once" (EIC W2, W4, W10, W11, DA unexamined premise).

## C. Explanations that exceed the evidence
- C1. §6.3 loose bound: 0.32% dose; bias not only variance; "accurate nearby" contradicted (all seats).
- C2. §6.4 diversity as main cause; repository diagnostics point elsewhere (R1 W7, R3 W6, W7, DA M5 to M7, EIC W14).
- C3. Reversal symmetry not used in the bound (R2 W2). May tighten bound and bear on diversity.
- C4. Certificate rarely does the deciding work: acceptance mostly via near-miss and refresh (R3 W5, DA M2).
- C5. Transfer-ability causal sentence (all seats, minor).

## D. Experiment design, before the queue closes
- D1. Group B and D do not test what §5.7 and §6.3 say (R1 W10, W11, R3 W4).
  - Checked 2026-11-09 from `outputs/ablation_20261028/exact_search/train/exact_progress.jsonl` (1,000,000 lines): every instance exhausted the 100-solve budget, never reaching a 2-opt optimum; search solves 100,000,000, total 101,762,944 to E100, against 2,304,553 training solves for the main run to E100 (about 44x). Mean Expert improvement over anchor 2.34%. Planned evaluation: `scripts/run_ablation_test500_after_A.sh` runs test500 eight-view for B and D after Group A; no validation2000 evaluation for B or D is scripted, unlike §5.7 text.
- D2. Ablations at E100 with one seed and no decision rule (R1 W12).
- D3. Seeds for both recipes (R1 W4).
  - Dialogue 2026-11-09: author confirmed one seed only for every run. Indirect stability evidence: E133-run validation2000 over E120 to E150 spans 17.703 to 17.734 (0.18%), against the 4.44% equal-solves gap; this is checkpoint fluctuation, not seed variance. Equal-solves POMO run took about 1.4 h for its first two epochs (started 2026-10-27 11:42), so one more seed of that arm costs about 2 h; a second E133 seed costs about 28 h plus selection. Author decision 2026-11-09: write as a limitation; no extra seed. Limitations must say every model is one training run with one seed and that intervals cover instance variation only; claim strength scaled to gap size (large gaps stated firmly, Real529 +0.21% and generalisation stated as close).
- D4. SA row is a calibration failure (R1 W17).
- D5. Proposed new arms: same network exact reward; no dense reward; uncertified surrogate; shifted-distribution retrain; table search at test time; rank and gradient diagnostics on E133.

## Existing evidence the paper does not use
- Equal-solves arm favours E133 (see above). Strongest answer to B3.
- Random-subset study (R3 W3, DA M7).
- 188 proven-optimal instances agree with the model (R2 S2).

## Author decisions, batch of 2026-11-09 (answers 1a 2a 3a 4a 5a)
- D2 (1a): ablations compared with C at E100 on test500 eight-view, as scripted; text changed from validation2000. Rule fixed before results are seen: a component has an effect only if the paired difference to C has a 95% CI excluding zero, reported by J; conclusions limited to E100. Report B's 101.8M calls and 4.4x wall time.
- B5, B6 (2a): Erdoğan's ILS credited as origin of retained-coordinate evaluation in the CVTSP; novelty narrowed to use as training signal and label filter. Gap table: column definitions, rows for Erdoğan ILS and Gambella RBA, n/a where a property cannot apply. G1 ("no learning method for the CVTSP") dropped; rely on G3. Author has no full texts. Yan and Lee abstracts read 2026-11-09 (see roadmap R-6): no G3 collision at abstract level. Liu abstract not obtainable.
- B7 (3a): methods contribution stated openly. §6 renamed "Discussion and implications for practice"; §6.2 placeholder removed; break-even table from existing numbers, including the region where E133 is not preferred. No new experiment.
- C1, C3 (4a): §6.3 narrowed; general lesson on expensive objectives removed; 0.32% reported as a weak test; hybrid-reward bias stated (R1 W9, DA M5). Reversal symmetry as a remark plus future work.
- C2 (5a): diversity rewritten as one of several hypotheses (route diversity, anchor pull of a pessimistic bound, fewer instances, network, G200 J/4 training), with evidence for and against; Conclusion sentence changed.
