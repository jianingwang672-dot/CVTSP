# Revision Roadmap, guided review round 1

Manuscript: `paper/full_20261104/phase4_draft/main.tex` (unchanged by the review). Date: 2026-11-09.
Sources: the five seat reports in this folder and `issue_index.md`, which records each author decision and each fact checked against raw files. Seat IDs: EIC (Journal-Fit Reviewer), R1 (methodology), R2 (domain), R3 (multi-fidelity perspective), DA (Devil's Advocate).

Verification: numbers described as checked, and those in R-1, R-3, R-5 and A2, were re-derived from raw files in this session. Numbers taken from a seat report (A9, A11, A12, the 251/180 and clustered ILS figures in R-7, the 14.6% excess in R-8, the diagnostic values in R-8 and R-9) have not been re-derived and must be checked before they enter the manuscript.

Status: all five seats signalled Major Revision. DA raised no CRITICAL item. `NOT_CALIBRATED`, `criteria_binding_unavailable`. All seats share one model family, which also drafted the manuscript.

Section numbers: §1 Introduction, §2.1 to §2.4 Related work (2.4 Research gap, Table `tab:gap`), §3 Problem, §4.1 to §4.7 Methodology (4.2 bound and Remark 1, 4.3 network, 4.4 anchor and search, 4.5 learning, 4.6 progressive search, 4.7 budget), §5.1 to §5.7 Experiments (5.1 setup, 5.2 synthetic, 5.3 generalisation, 5.4 literature benchmark, 5.5 exact solver, 5.6 training efficiency, 5.7 ablation), §6.1 to §6.4 Discussion, §7 Conclusion.

---

## Part 1. Positioning decided by the author

### R-1. Value proposition (B1; EIC W1, R1 W13, R3 W10, DA M10)
- Decision: the value rests on (2) training cost as the entry barrier when the problem itself changes, and (3) an algorithmic contribution. No retraining experiment.
- Fact checked in code: both models share one generator and the same six features, including speed ratio and reach; speed ratio 0.2 to 0.5. Real529 has two parameter sets, both at the range edges (325 at ratio 0.5, endurance 20; 204 at ratio 0.2, endurance 1).
- Actions: rewrite the motivation in §1 (remove "a change of vessel, drone model or service region is a change of distribution"). Delete the §6.1 sentence on retraining for a new vessel speed or drone endurance within a day. State that inside the training parameter range neither model needs retraining. Say what changes would require training from scratch (new constraints, new variants), without claiming a test.

### R-2. Core claim (B2; R1 W1, EIC W9, DA M1, m1)
- Decision: the claim is at recipe level: few exact solves give good quality. The table is the enabling mechanism. No sentence says the bound causes the saving.
- Actions: Abstract, §1 contributions, §7. Replace "We use this bound as the main training signal" and similar with wording about the recipe. Remove "the saving comes from the table" and "learning from the table did not cost transfer ability" (§5.3, also C5). The ablations report component effects and are not offered as attribution.

### R-3. Solver accounting (B3; EIC W3, W5, W17, R1 W2, W3, R3 W8, W9, DA M8, m3, m5)
- Decision: combine quality at equal budget with the ratio.
- Lead evidence (all checked): equal-solves POMO (3.84M solves, G200 recipe, J/4 starts) is +4.44% against E133 on validation2000 (E133 better on 1,406 of 2,000) and +5.67% on Real529 (354/110/65). The G200 recipe at 13.22M solves (epoch 9) is still +2.2% behind, so it needs more than 3.4 times E133's solves to match it. Upper bound 299M (G200 E200, 17.309). Source `outputs/data_efficiency_20261027/data_efficiency.csv`, `outputs/pomo_gurobi_equal_budget_20261027/equal_solves_results.md`.
- Ratio reporting: 78x as training-only at E200, factored into 15x instances and 5.2x calls per instance, with the J/4 start rule of G200 stated. Also give the as-run figure: about 11.8M including 5,175,733 selection solves and training to E200, ratio about 25. Wall time as run: about 28.3 h to E200 plus about 4 h selection (DA m5). Fix the CPU row (DA m3).
- Add the data-efficiency figure (`scripts/plot_data_efficiency.py`). Group A will narrow the crossover (see Part 4).
- "Close" is stated per set: Real529 +0.21% (not held out), generalisation +0.72%, test500 +1.13% with eight views and +2.60% without, validation2000 -2.23% for G200 (EIC W6, R1 W14, DA M13). Report unaugmented Real529 as well.

### R-4. Single seed (D3; R1 W4)
- Decision: limitation, no extra seed.
- Actions: §6.4 states that every model is one training run with one seed and that intervals cover instance variation only. Claim strength follows gap size. Optional supporting fact: E133-run validation spans 17.703 to 17.734 over E120 to E150 (0.18%), which is checkpoint fluctuation and not seed variance.

### R-5. Ablations (D1, D2; R1 W10 to W12, R3 W4)
- Decision (1a): compare B and D with C at E100 on test500 eight-view, as `scripts/run_ablation_test500_after_A.sh` does. Change the §5.7 text from validation2000. Rule fixed before results: a component has an effect only if the paired difference to C has a 95% CI excluding zero, reported by J. Conclusions limited to E100.
- Describe B as it is: Expert search with exact evaluation, one start, first-improvement 2-opt, at most 100 solves per instance, no LKH or refresh; the RL reward keeps the table. Report its 101,762,944 solves to E100 (all instances hit the budget) against 2,304,553 for C, about 44x, and 4.4x wall time (175,020 s against 39,774 s).
- Describe D as varying diversity of expert search starts, not POMO route diversity; use distinct terms.
- Remove "Group B of the ablation will quantify this claim" from §6.3.
- Check the queue line `EXIT 4B` with no exit code.
- Group A wording: retrain under the G200 recipe, with J/4 starts.

### R-6. Prior art and the gap (B5, B6; R2 W1, W7, W8, EIC W15, DA m8)
- Decision (2a). Credit Erdoğan and Yıldırım's ILS as the origin of retained-coordinate evaluation in the CVTSP. Restrict novelty to use as a training signal and as a label filter; state that solver-free local search adapts an established idea. Rewrite the §1 "closest work is Li" sentence and the §2.1 observation.
- Table `tab:gap`: define every column in a note; rename the bound column to a primal upper bound on every order from one solved solution; use n/a where no training exists; add rows for Erdoğan's ILS and Gambella's RBA; correct Vieira, Li, Kim, Pirnay and Luo cells.
- Drop G1 (no learning method for the CVTSP). Rely on G3. Remove §2.2 "does not extend to mothership routing"; note that the single-drone mothership problem is the same problem.
- Author has no full texts (2026-11-09). Abstracts read via Crossref and OpenAlex: Yan et al. (SSRN 2026) study one truck with two drones, moving targets and battery management, a continuous-space model and CAMS-MAPPO; Lee et al. (AIAA 2023) study a moving depot bound for a destination with several drones, cumulative distance, and sequential cooperative multi-agent RL. Neither abstract mentions a solver or a bound inside the reward, so neither threatens G3 at abstract level; both are learning methods for mothership problems, which confirms dropping G1. §2.2 wording may rest on these abstracts and should say so. Liu et al. (TR-E 2026): no abstract obtainable (publisher 403; none in Crossref, OpenAlex or Semantic Scholar); obtain through a library before submission. Still to run: Scopus and Web of Science with USV-UAV and carrier-rendezvous terms.

### R-7. Journal framing (B7; EIC W2, W4, W10, W11, DA unexamined premise, M12)
- Decision (3a): methods contribution stated openly.
- Actions: rename §6 to "Discussion and implications for practice"; delete §6.2 and its placeholder. Add a break-even table from existing numbers (training compute, per-plan compute, per-plan quality for G200, E133 and a metaheuristic), including the region where E133 is not preferred. Fix §6.1: ILS at 3x wins more instances than it loses (251 against 180), and on Poikonen clustered ILS at 1x is already ahead (DA M12, R2 W11). Replace "seconds" with the tested regime: single-core budgets against a 16-worker GPU policy. Rephrase "pays once" (EIC W10).

### R-8. §6.3 loose bound (C1, C3; all seats, R1 W8, W9, R3 W1 to W3, W11, W14, DA M3 to M5)
- Decision (4a). Retitle §6.3; remove the general lesson on expensive objectives. Report the 0.32% tightening as a weak test against a 33.6% excess. Delete "accurate nearby" (near-miss routes carry a 14.6% mean excess at E133). State the bias of the hybrid reward in §4.5: the one solved route is always the top advantage. Cite the existing diagnostic values with the checkpoint they come from (S500 E100 and E365, not E133).
- Reversal symmetry (R2 W2): add a remark that f(reverse(q)) = f(q) when origin equals destination and that the table does not use it; future work. State the identical origin and destination as a restriction (R2 W17).

### R-9. §6.4 diversity (C2; R1 W7, R3 W6, W7, EIC W14, DA M6, M7)
- Decision (5a). Present route diversity as one of several hypotheses: diversity; anchor pull of a pessimistic bound (R3's link to conservative estimation in offline RL is a hypothesis); fewer training instances; the network difference; G200's J/4 training. Give evidence on both sides, including the size pattern (overlap difference shrinks with J while the gap grows) and the from-scratch random-subset run that did move overlap. Replace "we traced" in §4.6 with "we observed". Change §7 "appears to be the main source".

---

## Part 2. Factual corrections (no decision needed)

- A1. §5.1: G200 used J/4 starts to epoch 200 and J after; add its learning rate and schedule. Remove the claim of the same start rule and the claim that overlap compares like with like (R1 W6, DA M9).
- A2. §5.2 placeholder: E133 eight-view test500 exists. +1.13% [+1.02, +1.25], 66/51/383, ratio of means +1.05; by J +0.46, +1.02, +1.32, +1.48, +1.39. Source `outputs/pomo_gurobi_equal_budget_20261027/eval_equal_solves_test500_aug8/results.json`.
- A3. §5.1: generalisation set uses 100 starts, not J (DA m2).
- A4. §5.4: new-best sentence fails as written: rounding, LBBD provenance, Example_260 (R2 W6, W16). Rename "BKS" to a reference value.
- A5. Remark 1: "only" is false; per-target recombination gives a valid tighter bound (R2 W9, R3 W13).
- A6. Citations: Garone 2014 vehicle, origin of the problem, Ren has one vessel with several UAVs, Vieira's "all instances" qualification (R2 W12 to W15).
- A7. Tolerance wording, "falsify", cosine horizon, near-miss quota, test-set disclosure (name E150) (R2 W10, R1 W18, W19).
- A8. Table `tab:meta` lacks a 1x row (R1 W5).
- A9. Certificate role (C4; R3 W5, DA M2): through E133, first-round certified 128,154, near-miss 159,866, refresh 162,888 acceptances; at E133, 659 of 10,000 pass the first-round test. Restate §4.4 screening.
- A10. Transfer sentence in §5.3 (C5; R1 W15, EIC W13, DA m1): remove the causal reading.
- A11. Operator comparison: G200 recipe ran epochs 1 to 3 in 39 to 45 minutes on this workstation; qualify "more than a week" (DA m4).
- A12. LKH alone was worse than swap plus 2-opt from E15; final recipe drops swap (DA m6).
- A13. Extension claim in §7 states a validity condition, not a usefulness condition (R3 W12, EIC W16, DA m7).

## Part 3. Defaults applied from earlier author decisions

- B4 TSIH (R2 W4, W5, EIC W8): no own run. Cite TSIH paper numbers and report gaps by size, including J = 100 (E133 +4.50, G200 +4.02, TSIH +0.51 at 45.9 s) and J = 200 (+5.91, +7.09, +0.46 at 164 s); all sizes +1.35, +1.14, +0.30 at mean 20.2 s.
- D4 SA row (R1 W17): mark it as a calibration failure or remove it.
- D5 new arms proposed by reviewers (same network with exact reward, no dense reward, uncertified surrogate, shifted-distribution retrain, test-time table search, rank diagnostics on E133, DA alternative paths): not run; list the most relevant ones as limitations or future work. Test-time table search is a natural future-work item.
- Title kept. Optional: consider whether "Deep Reinforcement Learning" still fits the recipe-level claim (DA M1).

## Part 4. Pending data and placeholders

| Item | Status | Where it goes |
|---|---|---|
| Group A, 24 h G200-recipe retrain | Running since 2026-11-09 10:22, due about 2026-11-10 | §5.6 crossover, §5.7 |
| B and D test500 eight-view | Queued after Group A | §5.7, with the rule in R-5 |
| Full Gurobi 3600 s on Real529 and test500 | Finished | §5.5 placeholder |
| Vieira 2026 values merged into the reference | Not done | §5.4, Table `tab:meta`, new-best counts |
| Liu full text (Yan, Lee abstracts read); Scopus, WoS | Not done | §2.2, Table `tab:gap` |
| 24 h curve for the proposed method | Not decided | §5.6 placeholder; or remove |
| Author list, funding, COI | Not provided | Front matter |

## Part 5. Items the author did not adopt, with reasons

- New retraining experiment on a shifted distribution: declined; value proposition changed instead (R-1).
- Extra seeds: declined; limitation (R-4).
- New ablation groups and reviewer-proposed arms: declined; the four groups A to D stay.
- Direct TSIH runs: declined; paper numbers cited (Part 3).

## Next steps
1. `/ars-revision` applies Parts 1 to 3 to a copy of the manuscript; Part 4 items stay as placeholders until data arrive.
2. Remaining `full` phases: 5a citation audit, 5b abstract, 7 formatting.
3. `/ars-reviewer re-review` after the placeholders are filled.
