# Peer Review Report

## Manuscript Information
- **Title**: Deep Reinforcement Learning with Sparse Exact Feedback for the Carrier--Vehicle Traveling Salesman Problem
- **Manuscript ID**: full_20261104, phase4_draft/main.tex
- **Review Date**: 2026-11-06
- **Review Round**: Round 1 (guided mode; no sprint contract)

---

## Reviewer Information

### Reviewer Role *
Peer Reviewer 2 (Domain)

### Reviewer Identity *
Researcher in the CVTSP and mothership-drone routing, specialising in mixed-integer SOCP, exact algorithms and decomposition heuristics. Declared blind spot: RL gradients and network architecture.

### Review Focus *
1. Correctness of the model and Proposition 1, and consistency with the definitions of Garone, Gambella and Li.
2. Reliability of the comparison with Li 2025 and Vieira 2026, and of the handling of best known solutions.
3. Accuracy of the research-gap table, missing key literature, and the risk from the three unread papers.

### Basis of this review
- **Read in the relevant sections (preprint versions, not the journal versions):** Erdoğan and Yıldırım, Optimization Online preprint of January 2020 (introduction, parts of §2, §4, §5.2, §6); Vieira et al., CIRRELT-2025-22 (introduction, model, computational results, conclusions).
- **Known only from an abstract or a search summary:** Li et al. (Crossref abstract; INFORMS full text and supplement returned HTTP 403); Masone et al. 2022; Salama and Srinivas 2022; the KAIST thesis record; arXiv:2112.12545.
- **Not obtained at all:** Lee 2023 (Crossref metadata and reference list only, no abstract), Liu 2026 (same), Yan 2026, Ren 2026, Gambella 2018, Poikonen and Golden 2020, Garone 2008/2010/2011/2014. What I say about these comes from how Erdoğan and Vieira describe them, or from the manuscript.
- **Li's matrix mechanism:** I could not check it against Li's text and take it from the Phase 1 report.
- **Files outside the manuscript (read-only):** `CVTSP_SOCP.py`, `src/CertifiedUpperBound.py`, the stop checks in `src/JointSelfImprovement.py`, the Phase 1 report, `TSIH results/TSIH-results.xlsx`, `outputs/bks_comparison_20261023/`, `outputs/metaheuristics_real529_20261026/summary.md` and `instance/Data/`. An ordinary referee would not see these files, so figures from them are marked "repository".
- **Own computation:** one light test (four J=30 random instances, one thread, 12 perturbed routes each) run from my scratchpad. It is an illustration, not a benchmark.

---

## Overall Assessment *

### Recommendation *
- [ ] **Accept**
- [ ] **Minor Revision**
- [x] **Major Revision** — Substantial revisions needed, re-review required after revision
- [ ] **Reject**

### Confidence Score *
4 — Core expertise for the model, proposition and CVTSP literature. Partial for the RL-related inferences in W3. Li's full text was not accessible.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Calibration Status
`NOT_CALIBRATED`

### Summary Assessment *
The paper replaces most exact SOCP rewards in a POMO-style policy for the CVTSP with a table built from one optimal fixed-order solution. It reports 78 times fewer solver calls than an exact-reward baseline, at a quality cost. Proposition 1 is correct, including the equality at q, and the model agrees with the Erdoğan and Vieira definitions; repository results confirm it empirically on 188 proven-optimal benchmark instances. The domain positioning around that result is weaker. The closest prior art in the CVTSP literature is not Li et al.: Erdoğan and Yıldırım's ILS already retains take-off and landing coordinates to evaluate sequence moves without a solver, and also swaps them on reversed sections. That symmetry is missing from the table, so the bound is needlessly loose on reversed segments, and the conclusion that tightness is not limiting does not follow. The published TSIH on the same 529 instances is left out of the comparison and is far stronger at J>=100, where the gap to the reference is 4 to 7 percent for the learned models, not the subset figures in Table 7. The new-best claim does not stand as written. The core economics claim against G200 survives. I recommend Major Revision because these repairs need rewriting, new tables and probably one new experiment.

---

## Strengths *

### S1: Proposition 1 is correct, in both its inequality and its equality
I proved both parts independently (see "Proof audit of Proposition 1" under Detailed Comments). The proof in the manuscript is valid. It keeps every sortie variable, resets only the leg times, and observes that the leg constraints bind at an optimum. The side comment that a merely feasible anchor still gives a valid bound but may lose equality at q is also correct. The code agrees: `_build_directed_carrier_time` implements the table and `from_solution` relabels position-indexed solver arrays by target before use.
**Evidence Anchor**: `text: §4.2, proof of Proposition 1 "Constraints \eqref{eq:out}--\eqref{eq:end} involve only these sortie variables and do not depend on the order"`

### S2: The model is consistent with the literature definition and with benchmark values
Eqs. (1) to (7) describe the fixed-order CVP used by Erdoğan and Yıldırım and by Vieira et al. Inequalities in the sync and leg constraints allow waiting, while Erdoğan's structural result (cited in Vieira's model section) shows optima where neither vehicle waits, so optimal values coincide with the synchronous form. The repository gives empirical support. On the 188 instances flagged proven optimal, neither model undercuts the optimum by more than 0.0054, which is the size of the two-decimal rounding, and no value falls below any reported lower bound. G200 matches 145 of those 188 within rounding. The manuscript does not report this check, and it should, because it is the strongest evidence that model and benchmark agree.
**Evidence Anchor**: `equation: Eqs. (1)-(7), fixed-order problem, §3.1`

### S3: The novelty claim for the proposition is appropriately modest
The authors place the contribution in the training use of the bound and not in the proposition. This framing is what allows the core contribution to survive W1.
**Evidence Anchor**: `text: §2.4 "we do not claim Proposition~\ref{prop:bound} as a new idea in itself"`

### S4: Adverse results are stated plainly
The manuscript reports its shortfall against G200 and against long-budget metaheuristics in the introduction, abstract and conclusion, instead of confining them to a table. That candour is less common than it should be in learned-solver papers.
**Evidence Anchor**: `text: §1 "the method does not match the baseline on instances from its own training distribution"`

---

## Weaknesses *

### W1: The closest CVTSP prior art is misidentified; Erdoğan's ILS already evaluates orders with retained sortie points
**Problem**: The manuscript names Li et al. as the closest work and says most heuristics must solve a candidate order to learn its cost. Erdoğan and Yıldırım's ILS, cited here only as "an iterated local search", does neither thing that claim implies. Their §4.2 says 1-OPT and 2-EXCHANGE "retain the take-off and landing coordinates of each target point, effectively preserving feasibility", 2-OPT swaps those points in the reversed section, and the points are determined "at the beginning of each local search iteration". The authors' z(σ,s) is therefore a solver-free feasible value for a new order, which is Proposition 1's inequality in operational form, used for search. Li's TSIH also avoids SOCP solves during its sequence search. The Phase 1 report lists Erdoğan at reading level B (abstract only), which explains the miss.
**Evidence Anchor**: `text: §1 "The closest work is \citet{li2025}, whose heuristic also derives an edge matrix from solved subproblems" and §2.1 "a candidate order has to be solved before its true cost is known"`
**Why it matters**: A CVTSP referee will know this ILS. Solver-free local search on fixed sortie points is presented in contribution 2 as part of what the table newly enables, but it has a direct precedent on this exact problem. The distinctive parts that remain are the dense reward for policy routes and screening for imitation labels.
**Suggestion**: Read Erdoğan §4 in full. Cite it as the origin of retained-coordinate evaluation in the CVTSP, add a row to Table 1, rewrite the §2.1 observation, and restrict the novelty claim to use as a training signal and label filter. State that the solver-free local search adapts an established idea.
**Severity**: Major
**Confidence**: 5 — core expertise; preprint text read directly, journal version not read

### W2: The bound ignores the reversal symmetry of the fixed-order problem
**Problem**: With origin equal to destination, as in all 529 benchmark files (repository: J+1 points per file) and in training, f(reverse(q)) = f(q). This follows because swapping s_i with l_i and t1_i with t2_i preserves all of Eqs. (2) to (5). The table does not use this. U_q(reverse(q)) is well above f(q), even though a solution of equal cost for that order is available from the same anchor. Each target may independently keep or swap its orientation, and the best choice for a given route takes a two-state dynamic programme, O(J) per route, with no solver. This is the same observation Erdoğan uses for 2-OPT. In my test on four J=30 instances, f(reverse(q)) matched f(q) to 1e-8, U_q(reverse(q)) exceeded it by 22% to 157%, and the orientation-aware value recovered it to 1e-8. On routes made by two random segment reversals of q, mean excess fell from 8.9/30.2/50.7/29.2% to 1.7/12.9/19.0/13.6%, with no violated bound. These routes favour the orientation fix and are not policy routes, so the size of the benefit on real training routes is unknown.
**Evidence Anchor**: `text: §4.2 "The table is asymmetric, so local search must account for the reversal of internal arcs."`
**Why it matters**: The manuscript treats asymmetry as a property to be handled, when part of it is an artefact of a fixed orientation. POMO's value rests on symmetric optima (Kwon et al.), and reversal is the obvious symmetry of a closed tour. A signal that penalises reversed segments relative to the anchor is a plausible contributor to the diversity loss the authors identify as their main deficit. That link is my hypothesis and is untested, and it lies at the edge of my competence.
**Suggestion**: State the symmetry formally, as a remark or as a tightened proposition, and report bound excess with and without orientation on actual policy routes. If the effect is material, rerun the diversity diagnostic and at least a pilot training run. Note that a route-level orientation evaluation is no longer a plain table sum, and that the 2-opt delta needs adapting.
**Severity**: Major
**Confidence**: 4 — the mathematics is certain; the magnitude on policy routes and the link to diversity are not

### W3: "A loose bound suffices" is not supported by the evidence given
**Problem**: The conclusion rests on a continuation experiment that tightened the bound by 0.32% against an excess of 33.6%. A treatment that small cannot show that tightness is unimportant; it shows only that adding tables by route-wise minimum barely moves the bound. In my test, orientation handling reduced excess far more than a second table did (W2), so "not easy to obtain from nearby solutions" is too strong. The sentence that looseness affects only ranking and variance also understates the issue. REINFORCE acts on that ranking, and looseness is not random noise: the manuscript itself reads the fall in Table 9's excess as routes moving closer to anchors, which suggests it is structured by distance from the anchor. Moreover, by the manuscript's definition one route per instance receives its exact cost while the others receive inflated bounds. This section also sits uneasily with Limitations, where the quality gap is attributed to falling diversity without any test of whether the bound contributes to it.
**Evidence Anchor**: `text: §6.3 "A much tighter bound is thus not easy to obtain from nearby solutions, and on the evidence so far it is not the limiting factor."`
**Why it matters**: The wider design lesson offered in this section is one of the paper's more quotable claims, and the method is still 2.6% behind G200 inside the training range. The reader cannot tell whether the bound is harmless or a cause.
**Suggestion**: Downgrade the claim to a statement about route-wise minima, or test a materially tighter signal. The Group B ablation will measure what the table saves. It will not measure whether tightness is limiting, so it does not by itself answer this question.
**Severity**: Major
**Confidence**: 3 — bound structure is core expertise; the reward-variance reasoning is outside it and belongs to the RL reviewer's check

### W4: The published TSIH is missing from the comparison on its own benchmark
**Problem**: Table 7 compares E133 with authors' implementations of ILS, VNS and SA, each of which scores every order with an exact SOCP. The CVTSP literature has moved away from exactly that design: Erdoğan retains coordinates, and TSIH uses an approximate matrix. Table 7 has no TSIH row and no published ILS row, although Li's per-instance UB and time are in the authors' repository. Repository figures: TSIH's UB column gives a mean gap of +0.30% to the manuscript's reference at a mean 20.2 s, against +1.35% for E133 at about 10.2 s of single-core Gurobi time. By size, TSIH is at +0.54% (46.9 s) against E133's +2.33% (39.5 s) on Regular J=100, +0.49% (44.8 s) against +6.67% (40.7 s) on Clustered J=100, and +0.46% (164 s) against +5.91% (79.3 s) on Regular J=200. Hardware differs and the TSIH times are not normalised, and E133 parallelises to about 2.8 s and 5.6 s of wall time at those sizes.
**Evidence Anchor**: `absence: Table 7 (tab:meta) and §6.1 — expected a row or paragraph comparing E133 with published TSIH and Erdoğan ILS results and times; checked Abstract, §1, §2.1, §5.4, Table 7, §6.1, §7`
**Why it matters**: The abstract says metaheuristics win when "given ten times the computing time". The strongest published heuristic appears to win at about twice the single-core time on average, and at about equal to twice that time at J>=100. The closing recommendation that the method suits settings needing quick solutions therefore has no state-of-the-art comparator. Norm evidence: comparison with published heuristics on the shared benchmark is standard in this subfield (Vieira preprint Tables 5, 7, 8, 9 and 10; Li's abstract reports outperforming state-of-the-art heuristics).
**Suggestion**: Add TSIH and published ILS results with the authors' stated hardware, and normalise or bracket time carefully. Report both wall and single-core time, because the learned method's advantage is parallel wall time. Rewrite the practical positioning accordingly, and if the honest niche is sub-second replanning on many workers or repeated retraining economics, say so.
**Severity**: Major
**Confidence**: 4 — core expertise; the workbook columns are interpreted from their headers and Li's text was not available to confirm run settings

### W5: Subset aggregation hides the size at which learned policies fall far from the reference
**Problem**: Poi325 pools J=10 to 200, so the Regular and Cluster columns average easy small instances with hard large ones. Computing the same gap by size from repository results gives E133 +4.50% and G200 +4.02% at J=100, and E133 +5.91% and G200 +7.09% at J=200. These are roughly two to four times the subset figures. Table 6's headline that E133 is 1.25% better than G200 at J=200 is true, but both are 6% to 7% above the reference, and the manuscript never says so.
**Evidence Anchor**: `table: Table 7 (tab:meta), E133 row, Poi325 Regular +1.75 and Poi325 Cluster +2.10`
**Why it matters**: The paper's practical case for a learned solver is strongest at scale, which is where its distance from the reference is largest. The result is also relevant to §5.2's reading of transfer to large sizes, which is measured only against G200 and not against any strong solution. Norm evidence: Vieira Tables 9 and 10 and Erdoğan Tables 12 to 19 report this benchmark by size.
**Suggestion**: Add a by-family, by-size gap table to the reference, and state J=100 and J=200 values in the text.
**Severity**: Major
**Confidence**: 5 — direct recomputation; the overall +1.35% reproduces the manuscript, which validates the calculation

### W6: The new-best claim does not stand as written
**Problem**: Three parts of this sentence and the next fail under the manuscript's own stated reference, the best upper bound over all methods in Li et al.
1. Rounding. Reference values have two decimals. E133 is strictly below the reference on 85 instances and G200 on 113, but 71 and 86 of those are proven optima, so these are rounding artefacts. Beyond 0.0056, E133 has 3 and G200 has 8, or 10 distinct instances. None of the 10 is proven optimal.
2. Provenance. All 10 list LBBD among their finders, an exact method stopped at 7,200 s. The reference therefore does not come from a heuristic in these cases. The sentence appears to be a leftover from an earlier comparison with the `BKS_sheet` value.
3. The single recheck. The repository README identifies the recomputed case as Example_260, with G200 at 266.512. The stated reference there is 266.26, so under the manuscript's definition it is not a new best. It beats only the older sheet value of 269.80.
**Evidence Anchor**: `text: §5.4 "both models also find new best solutions on a number of Poikonen instances, for which the reference values come from a heuristic"`
**Why it matters**: Claims of new best solutions are the first thing this community checks. Li's and Vieira's numbers are public, and an overstated count would damage the credibility of the whole paper. The placeholder for the counts is not at fault; the prose around it asserts the result already.
**Suggestion**: Define a threshold above rounding, and rebuild the list after merging Vieira. Recompute all surviving cases, not one, by a route-independent feasibility check with solver tolerances excluded. Publish the routes and sortie points. Remove the provenance clause unless verified. Decide whether a value beating a time-limited LBBD incumbent merits being called a new best, given that the instances may already be improved in Vieira's data or in later work.
**Severity**: Major
**Confidence**: 5 — direct recomputation from the authors' reference file

### W7: Several cells and columns of the research-gap table are wrong or undefined
**Problem**: Table 1's columns are not defined, and this produces errors.
- "Bound on all orders" is No for Vieira, whose algorithm "provides a valid lower bound" on the whole problem, which is a bound over all orders. It is also No for Li, whose abstract describes an analytical cut built on valid bounds of travel time between consecutive nodes. Only *upper* bounds from a primal solution are absent.
- "Exact feedback in training" is No for Kim et al., although the point of symmetric replay is to save exact evaluations of an expensive objective that training does use. It is No for Pirnay and Luo, whose rewards are exact tour lengths. For Li, Vieira and Ren there is no training at all, so the correct entry is "n/a".
- Gambella's RBA is absent. Per Erdoğan's account it ranks Hamiltonian paths by a lower bound computed per path and solves fixed-order problems in that order. That is the exact-method precedent for screening orders with a cheap bound.
**Evidence Anchor**: `table: Table 1 (tab:gap), columns "Exact feedback in training" and "Bound on all orders", rows Li, Vieira and Kim`
**Why it matters**: This table is the formal statement of the gap and exactly what a domain referee will audit. The final row looks unique partly because the columns are unspecified.
**Suggestion**: Define each column in a note. Rename the bound column to something like "primal upper bound on every order from a solved solution". Use "n/a" where a property cannot apply. Add Erdoğan's ILS and Gambella's RBA. The gap survives in this narrower form.
**Severity**: Major
**Confidence**: 4 — Vieira read directly; Li from the abstract only; Gambella through Erdoğan's description

### W8: The fence around the "no learning method" finding fails, and all three unread papers sit on it
**Problem**: The finding is limited to the CVTSP and said not to extend to mothership routing. That boundary does not exist in the literature. Erdoğan calls Poikonen and Golden's problem "essentially the same problem as the CVTSP", and Vieira lists it as another name for it. A learning method for single-mothership, single-drone routing would therefore be a learning method for the CVTSP. Lee et al. 2023, Yan et al. 2026 and Liu et al. 2026 are exactly the papers that could cross this line, and they are unread. I could not close the risk either: Crossref returned only metadata for Lee and Liu, and I could not locate Yan's abstract. The KAIST thesis record, which appears related to Lee 2023, describes a mothership as a moving depot with multi-agent RL (search summary only; the link to Lee 2023 is unconfirmed). Phase 1 also records that Scopus and Web of Science were not queried, and says this should be done before submission. Liu et al. is in the target journal.
**Evidence Anchor**: `text: §2.2 "That finding is bounded by our search and does not extend to mothership routing in general."`
**Why it matters**: This is one of the two claims my card singles out. The primary G3 claim about reusing one primal solution is much safer than G1, but G1 is stated in the text and depends on papers no one has read. Risk assessment for the placeholder: if Lee's problem has continuous launch and recovery with one vehicle, the G1 sentence is false as a matter of definition. G3 would still survive unless its reward uses a similar bound, which seems unlikely but is unverified.
**Suggestion**: Obtain all three full texts and the KAIST thesis before submission. State the G1 finding for "the CVTSP, also known as the single-drone mothership routing problem", or drop G1 and rely on G3. Run Scopus and Web of Science searches and report them in the search statement, and extend the search to USV-UAV and carrier-rendezvous terms.
**Severity**: Major
**Confidence**: 4 — the definitional point is verified from two sources; the content of the three papers is unknown

### W9: Remark 1 is false as stated
**Problem**: Tables can be combined at finer granularity than complete routes. Sortie constraints are per-target, so for a given route any target may take its sortie from any solved anchor. Per-target choice is safe; arc-wise choice is not. The right test is that each target uses one sortie throughout. A dynamic programme with K states per target finds the best such choice for a route in O(JK^2), and its value can only be equal to or below min_q U_q(π). In my test, with two anchors, it was strictly below on 47 of 48 routes (mean excess 4.9/16.6/31.9/21.4% against 6.8/26.8/43.0/26.5%), and it never violated the bound. The second half of the remark, on entry-wise minima, is correct.
**Evidence Anchor**: `text: Remark 1 "Several tables can be combined only by minimising over complete routes"`
**Why it matters**: It is a wrong formal statement in a paper whose only theorem is elementary, and it narrows the design space that was then tested in §6.3.
**Suggestion**: Replace "only by minimising over complete routes" with a statement that any combination must assign each target one sortie, and note both the route-wise and the per-target forms.
**Severity**: Minor
**Confidence**: 5 — short proof, confirmed numerically

### W10: "Holds up to that tolerance" misstates how numerical error accumulates
**Problem**: The audit limits each constraint's violation to 1e-5 in absolute terms. A violation in a sortie must be repaired by extending times, so the worst-case effect on the bound accumulates over J targets and is amplified by 1/v_c, roughly J·δ/v_c, and more where endurance binds. At J=100 and v_c=0.2 that is on the order of 5e-3, not 1e-5. The code itself uses other values: 2e-5 for the stop test and up to 1e-4 for anchor consistency. In practice errors are far smaller; I measured |U_q(q)-f(q)| of at most 2.3e-9 in four cases. A related wording problem is that a run stop cannot "falsify" a proved proposition. It would indicate a numerical or implementation fault.
**Evidence Anchor**: `text: §4.2 "the bound therefore holds up to that tolerance"`
**Why it matters**: It is a small imprecision, but "certified" is a strong word and the title-level claim rests on it.
**Suggestion**: State per-constraint tolerances and the stop tolerance actually used, report the largest observed violation, and describe the bound as valid up to solver feasibility tolerances accumulated over the J sorties.
**Severity**: Minor
**Confidence**: 5 — code read directly; the accumulation estimate is order-of-magnitude

### W11: The managerial claim about the three-times budget contradicts Table 7 on clustered instances
**Problem**: On Cluster, ILS and VNS at 3× are at +0.36% against +2.10% for E133. Repository data also show ILS at 1× at +1.30% on Cluster, so it beats E133 there at equal single-core time. The 1× row exists and is not reported.
**Evidence Anchor**: `text: §6.1 "ILS and VNS at three times the single-core time of E133 are behind it."`
**Why it matters**: Clustered targets are a realistic maritime pattern, so the unqualified statement would mislead a planner. The manuscript notes the cluster weakness at 10× but not here.
**Suggestion**: Qualify by instance family and add the 1× budget.
**Severity**: Minor
**Confidence**: 5 — direct table reading

### W12: Origin of the problem is misattributed
**Problem**: Erdoğan states the CVTSP was introduced by Garone et al. (2010b), and the fixed-order CVP by Garone et al. (2008). Both are in the Phase 1 report but not in `refs.bib`. Garone 2011 is at reading level C.
**Evidence Anchor**: `text: §1 "introduced by \citet{garone2011}, models this operation"`
**Why it matters**: Small but visible to anyone in the subfield, and it is the first citation of the paper.
**Suggestion**: Cite Garone et al. 2008 and 2010 for the origin, after reading them.
**Severity**: Minor
**Confidence**: 4 — secondary source only; the originals were not read

### W13: Garone 2014 is described with the wrong vehicle
**Problem**: In the generalised problem it is the Vehicle that visits several targets between take-off and landing (Erdoğan's account). The carrier does not.
**Evidence Anchor**: `text: §2.1 "extended it to carriers that visit several targets per sortie"`
**Why it matters**: It suggests the source was not read (level C) and confuses the two roles on which the paper depends.
**Suggestion**: Replace with "to a vehicle that visits several targets per sortie".
**Severity**: Minor
**Confidence**: 4 — secondary source only

### W14: Ren et al. is cited for several carriers, but it has one vessel with several UAVs
**Problem**: The manuscript itself describes Ren et al. as a vessel carrying several UAVs in §1 and §2.1, and the title agrees. The conclusion then cites it as a several-carrier extension. Several UAVs on one vessel raises a different separability question, because sorties may overlap in time and leg structure changes.
**Evidence Anchor**: `text: §7 "Extending the approach to several carriers, as in the vessel routing model of"`
**Why it matters**: It is an internal inconsistency in the future-work claim about how far the structure reaches.
**Suggestion**: Cite Fahradyan et al. 2020 or Li et al. 2024 for several carriers, and treat the multi-UAV case separately, with its overlap issue.
**Severity**: Minor
**Confidence**: 5 for the inconsistency; Ren's full text was not read

### W15: Vieira's "all benchmark instances" claim is repeated without its qualification
**Problem**: The sentence follows Vieira's abstract, but the same preprint's Tables 10 and 11 give gaps of 0.37% to 0.42% at clustered J=15 and 7.90% at clustered J=20 on the Poikonen set. Its text says optimality was not proved for all instances of that group. The claim therefore cannot cover every Poi325 instance up to 25. Only the CIRRELT preprint was read; the journal version is unread and could differ.
**Evidence Anchor**: `text: §2.1 "solves all benchmark instances with up to 25 customers"`
**Why it matters**: Real529 is the benchmark used here, so a reader will assume these 529 instances are all closed up to 25.
**Suggestion**: Read the journal version and specify which instance sets are closed.
**Severity**: Minor
**Confidence**: 4 — preprint read directly; final version unread

### W16: "BKS" does not describe the reference, and the Vieira merge carries a specific practical risk
**Problem**: Negative gaps in a best-known column show the reference is not best known. Vieira's own figures show why, on the Poikonen set. Size-mean UB is lower than the reference mean at Regular J=30 (290.59 vs 290.64) and J=50 (354.77 vs 355.33), and at Clustered J=15 (252.73 vs 252.97) and J=20 (260.07 vs 260.34). For J>=100 and Clustered J>=30 Li's reference is better. On Erdoğan's 25 to 50 set the reference is safe: of the 69 instances I could parse from Vieira's Table 8, 62 tie, 7 are worse and none is better (repository reference). Per the placeholder rule I do not fault the missing merge. The risk is that the preprint reports Poi325 only as means per size, so the per-instance merge cannot be done from it. The expected change in the overall gaps is small, but the signs in the Cluster column may change.
**Evidence Anchor**: `table: Table 7 (tab:meta), Poi325 Cluster column, ILS 10× -0.02 and VNS -0.03`
**Why it matters**: The quoted VNS result of -0.03% may be an artefact of a stale reference at clustered J=15 and J=20.
**Suggestion**: Obtain per-instance values from Vieira's data or the authors. Rename the reference ("best value in Li et al.") until merged, and then recompute with all methods in this paper included so that no gap is negative.
**Severity**: Minor
**Confidence**: 4 — means compared directly; instance-level values not available to me

### W17: The restriction to identical origin and destination is not stated as a restriction
**Problem**: Erdoğan's definition has a destination that "may, in general, be different from the origin", and Vieira's model also has both. The manuscript has a depot only and does not note that this specialises the general problem. Benchmark files all have J+1 points, so results are unaffected. Proposition 1 holds without change in the general case. The reversal symmetry in W2 does not.
**Evidence Anchor**: `text: §3 "Let $d \in \mathbb{R}^{2}$ be the depot"`
**Why it matters**: It affects the accuracy of the claim of following the standard definition and the scope of any symmetry result.
**Suggestion**: Add one sentence stating the specialisation and that the benchmark satisfies it.
**Severity**: Minor
**Confidence**: 5 — Erdoğan text and benchmark files checked; Gambella's and Li's own definitions not read

---

## Detailed Comments *

### Proof audit of Proposition 1

**Verdict: correct as stated, for any v_c > 0.** It uses no assumption beyond this. v_d > v_c and convexity are unused, and optimality of q is used only for the equality.

Fix π and let x^π keep (s*, l*, t1*, t2*, τ*) by target, with T_k = C^q at (π_k, π_{k+1}).

| Constraint | Treatment | Result |
|---|---|---|
| (2) outbound | same s*_i, t1*_i, p_i; no π in the constraint | holds |
| (3) inbound | same | holds |
| (4) sync | same | holds |
| (5) τ = t1 + t2, τ ≤ E | same; no π in the constraint | holds |
| (6) leg, k = 0 and k = J | with the depot conventions, C_{0j} and C_{i0} match the constraint | holds with equality |
| (6) leg, 1 ≤ k < J | C_{ij} = norm(s*_j − l*_i)/v_c | holds with equality |
| (7) non-negativity | norm divided by v_c > 0; the others unchanged | holds |

The objective at x^π is F^q + ΣC = U_q(π), so f(π) ≤ U_q(π).

**Equality.** The manuscript's argument (T*_k is at its lower bound since the objective strictly increases in T_k) is valid. There is a shorter route that needs no monotonicity: (6) gives T*_k ≥ C^q_{q_k q_{k+1}} directly, so f(q) = F^q + ΣT*_k ≥ U_q(q), and the first part gives f(q) ≤ U_q(q).

**Two notes.** (i) U_q depends on which optimum is chosen when the SOCP has several. The statement holds for each, but the notation hides the dependence. (ii) f is attained: the set is non-empty (s_i = l_i = p_i with zero flight), and sublevel sets are compact since τ ≤ E confines each sortie.

**Code check.** `CVTSP_SOCP.py` (active code from line 209) implements Eqs. (1) to (7). `_route_objective` implements U_q. `from_solution` requires Gurobi status 2, as the text says. Tolerance wording is W10.

**Later statements that rest on the proposition.** The screening rule, U_q(π) < f(q) implies f(π) < f(q), is correct. The sentence that a merely feasible anchor retains validity is correct. The entry-wise warning in Remark 1 is correct, but its "only" is not (W9). The extension statement in Limitations ("only if sortie constraints remain independent of the order") is a reasonable sufficient condition, stated rather too firmly as necessary.

### Title & Abstract
- The title is accurate. "Certified" needs the W10 qualification.
- The abstract reports gaps only against G200 and says nothing about distance to reference values. Given W4 and W5, a reader will overestimate quality on the benchmark.

### Introduction
- Motivation is clear. The claim that the bound "says something about every route" is right, but W1 removes its appearance of being a new structural observation.
- The search statement names its sources honestly. See W8 for what it omits.

### Literature Review / Theoretical Framework
- **Coverage**: Good breadth in neural combinatorial optimisation. Thin on the literature nearest to the mathematics: the CVTSP originals, Erdoğan's structural results and ILS mechanism, Klaučo et al., the TSPN and close-enough TSP alternation literature, and flexible launch-and-recovery work in drone routing. Reading levels in Phase 1 show that foundational CVTSP papers were cited from level B or C; W1, W12 and W13 are the direct consequences.
- **Integration quality**: §2.3 is real synthesis. §2.1 is closer to a list until its last paragraph, and that paragraph's first observation is the one W1 disputes.
- **Research gap argument**: G3, in the narrow form of a primal upper bound used as the main DRL training signal and label screen, stands in my reading. I found no counter-example, within the limits of W8. G1 is weak. The move from "closest work is Li" to "closest work is Erdoğan's ILS for the bound, Li for the matrix" makes the gap more honest and no smaller in its training component.
- **Framework**: The bilevel reading (policy over orders, SOCP over sortie points) is appropriate and deeply applied. An alternative framing that the authors could use is alternating minimisation: f(π') ≤ U_q(π') ≤ U_q(q) = f(q) gives a monotone descent, which explains refresh rounds and connects directly to two-stage methods.

### Methodology / Research Design
Within my remit only. Benchmark handling is covered by W4 to W6 and W16. Two matters are for Reviewer 1, but I note them because they affect the comparison I was asked to check. Budgets are counted to epoch 133 although selection used validation up to epoch 200, which is relevant to the 15× and 78× ratios. Also, G200's 299,437,760 calls over 20,000,000 instances is about 15 calls per instance, which should be reconciled with "one call per route", given that POMO routes scale with J.

### Results / Findings
I checked table arithmetic: means and win/tie/loss totals in Tables 4, 5 and 6, the call total in Table 8, the budget ratios in Table 10, and the Table 7 "All" column, which reproduces from size-weighted subset values. All are consistent, and E133's +1.35% reproduces from the repository.

### Discussion
§6.1 and §6.3 are the places where domain claims go beyond the evidence (W4, W11, W3). The statement on retraining economics is the paper's strongest contribution and stands if budgets stand.

### Conclusion
The first paragraph is fair except for the unqualified suitability claim (W4). "All it requires" is well put. The several-carrier remark needs correcting (W14).

### References
- `refs.bib` has no entries for the Garone 2008 and 2010 papers.
- Coverage of very recent CVTSP work is good. Li 2025 and Vieira 2026 are both present and used.

### Theoretical Framework
- **Appropriateness**: Decomposition into order and sortie levels is the field's standard and suits the question.
- **Application depth**: Deep; the bound is the method's mechanism and not a label.
- **Alternative frameworks**: The alternating-minimisation view above, and per-target recombination (W9). Erdoğan's single-target CVTSP-B subproblem is a third option: a small SOCP that re-optimises one sortie between fixed neighbours and could tighten a route's value at a fraction of a full solve.

### Academic Argument Quality
- **Factual accuracy**: W6, W7, W12 to W15.
- **Argument logic**: The chief leap is W3.
- **Terminology precision**: "BKS" (W16); "certified" (W10); "bound" without a direction, in Table 1 (W7).

### Contribution to the Field
- **Incremental contribution**: A demonstration that a CVTSP policy can be trained with about 2.9 exact solves per instance, using a primal bound as a dense signal plus verified labels. This is a real, methodological contribution, at the level of training economics rather than solution quality.
- **Positioning**: Correctly placed against G200. Incompletely placed against the CVTSP heuristic literature.
- **Overclaiming**: Risk in the "solutions needed quickly" conclusion, the loose-bound lesson, and new bests.

### Missing Key References
Attested from sources I read, or from search results as noted:
- Erdoğan and Yıldırım §4.2, already in the bibliography; it needs to be used, not added.
- Garone et al. 2008 (CDC, CVP) and Garone et al. 2010 (CDC): the origin of the problem. Metadata from Erdoğan's reference list.
- Klaučo et al. 2014: MISOCP path planning for the generalised problem. Metadata from Erdoğan's reference list.
- Coutinho et al. 2016: branch-and-bound for the close-enough TSP, the sister problem with fixed-order SOCP. Metadata from Erdoğan's reference list.
- Masone, Poikonen and Golden 2022: alternation of discrete and continuous improvement with edge launches. Abstract-level only.
- Salama and Srinivas 2022: flexible launch and recovery sites, in the target journal. Search summary only.
- Amorosi, Puerto and Valverde 2023: mothership with multiple drones and arc routing. Metadata from Vieira's reference list.
- arXiv:2112.12545: DRL for the TSP with drone, a coordination case with solver-free reward. Search summary only; authors not confirmed by me.

Search leads, all `[UNVERIFIED]`: literature on alternating TSP and continuous tuning in the TSP with neighbourhoods and close-enough TSP, which may contain the earliest statement of the fixed-points argument; recent USV-UAV collaborative routing with attention or DRL; Li's GitHub or INFORMS repository for the exact TSIH run settings; and neural methods with a convex or continuous lower-level solve.

~~~yaml suggested-references
- title: "Cooperative path planning for a class of carrier-vehicle systems"
  authors: ["Garone, E.", "Naldi, R.", "Casavola, A.", "Frazzoli, E."]
  year: "2008"
  doi: null
  arxiv_id: null
- title: "Cooperative mission planning for a class of carrier-vehicle systems"
  authors: ["Garone, E.", "Naldi, R.", "Casavola, A.", "Frazzoli, E."]
  year: "2010"
  doi: null
  arxiv_id: null
- title: "Mixed-integer SOCP formulation of the path planning problem for heterogeneous multi-vehicle systems"
  authors: ["Klauco, M.", "Blazek, S.", "Kvasnica, M.", "Fikar, M."]
  year: "2014"
  doi: null
  arxiv_id: null
- title: "A branch-and-bound algorithm for the close-enough traveling salesman problem"
  authors: ["Coutinho, W. P.", "do Nascimento, R. Q.", "Pessoa, A. A.", "Subramanian, A."]
  year: "2016"
  doi: "10.1287/ijoc.2016.0711"
  arxiv_id: null
- title: "The multivisit drone routing problem with edge launches: An iterative approach with discrete and continuous improvements"
  authors: ["Masone, A.", "Poikonen, S.", "Golden, B."]
  year: "2022"
  doi: "10.1002/net.22087"
  arxiv_id: null
- title: "Collaborative truck multi-drone routing and scheduling problem: Package delivery with flexible launch and recovery sites"
  authors: ["Salama, M. R.", "Srinivas, S."]
  year: "2022"
  doi: "10.1016/j.tre.2022.102788"
  arxiv_id: null
- title: "A multiple-drone arc routing and mothership coordination problem"
  authors: ["Amorosi, L.", "Puerto, J.", "Valverde, C."]
  year: "2023"
  doi: null
  arxiv_id: null
- title: "A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone"
  authors: []
  year: "2021"
  doi: null
  arxiv_id: "2112.12545"
~~~

---

## Questions for Authors *

1. Does the reference used in Table 7 remain the minimum over all methods in Li et al., and if so, which instances count as new bests once the threshold exceeds two-decimal rounding and Vieira's values are merged? Is the Example_260 recheck still relevant?
2. On actual policy routes, how much of the 33.6% bound excess disappears with orientation-aware evaluation? Does overlap between POMO routes change if the reward uses it?
3. Why is TSIH not in Table 7, given that per-instance values and times are available? On what hardware were they produced, and what time basis would you defend for the comparison?
4. Do Lee 2023 or the related KAIST thesis, Yan 2026 or Liu 2026 allow continuous take-off and landing points with one drone? If so, how does the G1 sentence change?

---

## Minor Issues

### Language / Grammar
- §4.2: "Three conditions apply in practice", but only the first is a condition; the other two describe checks.

### Citation Format
- Key and year differ in several entries, so printed years will not match keys: `kool2019` (2018), `hottung2022` (2021), `kim2024` (2023), `ferber2023` (2022), `li2025` (2026), `ren2026` (2027). The hand-typed years in the Table 1 caption and in placeholders should be checked against the printed form.
- Many neural combinatorial optimisation entries are `@misc` arXiv records. Check for published versions at Phase 5.

### Figures and Tables
- Table 7 uses 100(UB-BKS)/UB, while the metrics paragraph uses the baseline as denominator. State the reason or align them.
- Table 1's five properties need a table note (W7).

### Layout
- Algorithm box labelled as a table: acceptable, though `alg:train` may be better as a float of its own type.

---

## Criterion-Bound Judgements *

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Reviewer card focus 3; rubric Dimension 1 | PARTLY_MEETS | `text: §1 "The closest work is \citet{li2025}"` | The training use of a primal bound appears new within the search; solver-free local search has CVTSP precedent | Three unread papers | Yes: positioning must change |
| Methodological Rigor | Rubric Dimension 2, domain subset only | PARTLY_MEETS | `table: Table 7 (tab:meta)` | Comparator choice and reference handling are weak; proposition and code are sound | Experimental design left to Reviewer 1 | Yes: W4 to W6 |
| Evidence Sufficiency | Rubric Dimension 3 | PARTLY_MEETS | `text: §6.3 "not the limiting factor"` | Economics claim is supported; the loose-bound claim and new bests are not | RL inference partly outside my expertise | Yes: W3, W6 |
| Argument Coherence | Rubric Dimension 4 | PARTLY_MEETS | `text: §7 "several carriers"` | Main line is traceable; the speed-based conclusion lacks its comparator | none identified | Yes: W4 |
| Writing Quality | Rubric Dimension 5 | NOT_ASSESSED | — | Outside this seat's remit except for terminology | Left to other seats | No |
| Literature Integration | Reviewer card focus 3; rubric | PARTLY_MEETS | `text: §2.2 "does not extend to mothership routing in general"` | NCO synthesis is good; CVTSP foundations were not read closely enough | Li's full text inaccessible to me | Yes: W1, W7, W8 |
| Significance & Impact | Rubric; reviewer card | PARTLY_MEETS | `text: §1 "a change in what a learned CVTSP solver costs to obtain"` | Real contribution on training cost; practical value relative to TSIH is unproven | Time comparison across hardware is uncertain | Yes: W4, W5 |

The unresolved decision-bearing items are prior-art positioning (W1), the reversal symmetry and its consequence for the bound claims (W2, W3), the state-of-the-art comparison (W4, W5), the new-best claim (W6), the gap table (W7) and the G1 boundary (W8). All appear repairable. W2 and W3 may need one new experiment, and W4 needs careful treatment of time. Proposition 1 and the economics claim against G200 are not in question from this seat.
