# Peer Review Report

## Manuscript Information
- **Title**: Deep Reinforcement Learning with Sparse Exact Feedback for the Carrier–Vehicle Traveling Salesman Problem
- **Manuscript ID**: none assigned
- **Review Date**: 2026-11-06
- **Review Round**: Round 1 (guided mode)

---

## Reviewer Information

### Reviewer Role *
Journal-Fit Reviewer (`eic` seat). This card gives a recommendation signal only. It is not an editorial decision.

### Reviewer Identity *
Senior editorial-board perspective for logistics and transportation operations research journals, working on drone and carrier routing and its operational decisions.

### Review Focus *
Whether this is a logistics management paper or an algorithm paper sent to a logistics journal; whether the managerial insights follow from the results; and whether a contribution placed in the use of a simple proposition can carry a journal paper. I test throughout whether the selling point survives a 2.6% deficit to the baseline and better solutions from metaheuristics at ten times the time.

### Venue criteria status
`criteria_binding_unavailable`. The author declares Transportation Research Part E, but no criteria were bound. I make no claim about alignment with that journal's rules or scope. "Fit" below means fit as a reader of logistics and transportation OR would reason about it.

Red `\placeholder{...}` blocks are not faulted for missing content. Their bearing on the argument is assessed in "Dependence on pending results".

---

## Overall Assessment *

### Recommendation *
- [ ] **Accept**
- [ ] **Minor Revision**
- [x] **Major Revision** — Substantial revisions needed, re-review required after revision
- [ ] **Reject**

### Confidence Score *
4. Mostly within my area. Proof details and network training details are outside my focus and I did not assess them.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Summary Assessment *
The paper shows that one optimal fixed-order solution bounds the cost of every visiting order in the CVTSP, and builds a DRL training method on that bound. It reports 15 times fewer instances and 78 times fewer solver calls than POMO on exact rewards, at gaps of +0.21% to +2.60% to that baseline. As an algorithmic study it is candid and carefully counted. As a logistics paper it is not yet argued. The bound is sound and the accounting of calls is exact, and the authors state their deficits openly. The central weakness is the practical case. The value of cheap training rests on recurring retraining, yet the paper's own network is built to avoid retraining for two of the three changes it names, and no retraining experiment appears. Cost ratios compare unequal quality, stop at a checkpoint chosen after a longer run, and leave a one-off compute saving unweighed against a recurring loss in mission time. The managerial section therefore reads as attached to the results, not derived from them. All of this is repairable, and one experiment on a shifted distribution would address much of it. I signal Major Revision.

---

## Strengths *

### S1: Deficits are stated where a reader will see them
The abstract reports the +2.60% gap and the metaheuristic result, and the Introduction repeats both. Few papers in this genre lead with the conditions under which they lose.
**Evidence Anchor**: `text: Introduction, final paragraph before the outline "the method does not match the baseline on instances from its own training distribution"`

### S2: A short structural result with a usable consequence
Proposition 1 turns one solve into a one-sided signal on every order, and the screening rule follows in one line. The authors also audit anchors and stop on violation, so "certified" is backed by an operational check.
**Evidence Anchor**: `text: §4.2 "every anchor solution passes an audit of all constraints with a tolerance of"`

### S3: Evidence contamination is disclosed
The authors state that test500 was inspected once and that Real529 is not held out. This is more than most comparative computational studies report.
**Evidence Anchor**: `text: §5.1 Data "Real529 is not a held-out set, because earlier checkpoints of this project were evaluated on it during method development."`

### S4: Novelty claims are bounded and modest
The authors disclaim the proposition as a new idea and bound the "no learning method" finding by their search. That leaves a narrow claim they can defend.
**Evidence Anchor**: `text: §2.4 "we do not claim Proposition"`

### S5: Solver calls are accounted for line by line
Table 8 sums exactly (1,330,000 + 1,330,000 + 493,128 + 686,842 = 3,839,970), and the per-instance figure and the 15 and 78 ratios recompute correctly. I also recomputed the win/tie/loss totals and mean rows of Tables 4 to 6 without finding an error.
**Evidence Anchor**: `table: Table 8, Total row — 3,839,970 calls, 2.89 per instance`

### S6: Metaheuristics are compared on a per-instance time basis with the same exact scorer
Budgets scale with E133's time on each instance and every order is scored by the same SOCP, so the comparison is not distorted by a different evaluator. Reporting the loss at 10× was a choice the authors could have avoided by stopping at 3×.
**Evidence Anchor**: `table: Table 7, rows ILS 10× and VNS 10× — +0.73 and +0.78 against +1.35 for E133`

---

## Weaknesses *

### W1: The retraining premise is undercut by the paper's own design and is never tested
**Problem**: The practical value of cheap training depends on an operator having to retrain. The Introduction calls training cost a recurring expense because a change of vessel, drone model or service region is a change of distribution. §6.1 repeats that an operator can retrain for a new vessel speed, endurance or region within a day. But §4.3 gives the network speed-ratio and reach features so that one network serves different parameters, and §5.1 trains across $v_d \in [0.4,1]$ and $E \in [0.05,1]$. Within those ranges no retraining is needed, for G200 either. Results on Real529 and at $J$ up to 500 also show both policies transferring without retraining. The paper contains no experiment in which any model is retrained for a changed setting.
**Evidence Anchor**: `text: §6.1 "An operator can therefore retrain for a new vessel speed, drone endurance or service region within a day on a single workstation" and §4.3 "The speed and endurance features let one network serve instances with different operating parameters."`
**Why it matters**: Inside the training ranges an operator who already holds G200 gets better plans at the same test time, and E133 offers nothing. Outside them the paper has no evidence. The conclusion that the method suits settings where "retraining has to be affordable" therefore has no supporting result, and this is the only ground on which E133 is preferred to G200.
**Suggestion**: State the parameter ranges in which a trained network can be reused, then run the case that actually needs retraining. The natural one is a shifted distribution: for example, train the recipe on clustered instances within a day, and compare with G200 applied unchanged and with ILS or VNS at the same time, on Poi325 Cluster. If the retrained policy closes the +2.10% gap, the practical claim becomes a result. If not, drop the retraining claim and present the work as a method paper.
**Severity**: Critical
**Confidence**: 4 — core expertise: operational reading of computational studies; the contradiction is internal to the manuscript

### W2: The paper's question and answer are algorithmic, and the managerial section contains no operational result
**Problem**: The stated question concerns solver calls in training, and the Conclusion answers that question. The title carries no logistics decision. No experiment varies an operating parameter, models a cost, or reports a mission outcome other than the gap. "Managerial insights" in §6 concern when to spend CPU time, which is advice about solvers rather than operations. The only subsection that would be operational (§6.2) is a placeholder that offers its own removal. The maritime and humanitarian framing appears in the Introduction and then does no further work.
**Evidence Anchor**: `text: §1 "This paper asks whether most of those calls are necessary."`
**Why it matters**: A logistics reader will ask what a planner learns about operating a carrier–drone system. As drafted, the answer is nothing beyond method choice. On focus 1 of my card, this is an algorithm paper with a logistics introduction.
**Suggestion**: Either keep §6.2 and make it decide something (how speed ratio, endurance and clustering change mission time, and where policy, metaheuristic or retraining should be used), or be open that this is a methods contribution and remove the "managerial" label. A planning question should appear in the Introduction and be answered in the Conclusion.
**Severity**: Major
**Confidence**: 4 — core expertise: drone and carrier routing as an operations topic

### W3: Cost ratios compare two different quality levels
**Problem**: The headline ratios set a policy that is 2.60% worse on the primary evidence set against a baseline at its full 200 epochs. A ratio of costs at unequal quality does not measure efficiency, because the last percentage points of a learning curve commonly take much of the budget. The abstract and Contribution 3 state the ratios unconditionally. Group A, as designed, measures quality at E133's budget, but nothing in the paper states the converse: the budget at which POMO on exact rewards reaches E133's quality.
**Evidence Anchor**: `text: §1 Contribution 3 "The method uses 15 times fewer training instances and 78 times fewer solver calls, with no change in test time."`
**Why it matters**: The 78× figure is the paper's selling point. Without a matched-quality reading, it cannot be told apart from stopping a learner early. This is the point at which the 2.6% deficit bears directly on the claim.
**Suggestion**: Report cost to reach a stated quality, in calls and in wall time on the same hardware. Read G200's own validation history for the epoch at which it first matched E133, if those logs exist. State the ratios in the abstract only at matched quality, or qualify them with the quality difference in the same sentence.
**Severity**: Major
**Confidence**: 4 — core expertise: comparative computational studies

### W4: A one-off compute saving is never weighed against a recurring loss in plan quality
**Problem**: §6.1 names the training budget as the relevant cost for the operator, but the objective of the problem is mission completion time, incurred on every mission. The saving is paid once; the deficit recurs on every plan. The paper says a planner needs all three results and then gives no way to combine them. The numbers it does supply are not favourable. By my estimate from reported figures, E133 used 154.3 CPU hours in training (86.7 Gurobi plus 67.6 local search). A 10× metaheuristic costs about 93 s more per instance on a single core, so about 6,000 instances are needed per training run before the policy recovers its own training CPU, while the metaheuristic gives the better plans on Real529 overall. This excludes GPU and elapsed time and treats training CPU and planning CPU as equal in value.
**Evidence Anchor**: `text: §6.1 "The relevant cost for the operator is the training budget"`
**Why it matters**: On focus 2 of my card, this is the main test of whether the insight follows from the results. A vessel-time reader would ask why a mission-time loss on every plan should be traded for hours of computing that could run overnight once.
**Suggestion**: Give an explicit break-even: plans per training run, mission-time loss per plan and compute per plan, for each of G200, E133 and a metaheuristic. Let the recommendation follow from that table, including the region where E133 is not preferred.
**Severity**: Major
**Confidence**: 4 — applying standard cost reasoning to the manuscript's own figures; the 6,000 figure is a rough reviewer estimate

### W5: The training budget stops at a checkpoint that was chosen after a longer run
**Problem**: The run went to 200 epochs, and E133 was selected from it by validation. The budget and "less than a day" claims count only epochs 1 to 133 while G200 is counted to epoch 200. No stopping rule is stated that would have ended the run at 133. On a like-for-like basis the instance ratio is 10, not 15. By my estimate calls are at least 6.1 million, giving about 49× or less, because refresh exceeds one call per instance late in the run. Wall time scales to at least 28.9 h, more than a day. Separately, the settings came from earlier development runs whose cost is uncounted. The ratio chosen for the abstract is also the largest available: Table 10 gives about 30× for CPU and about 11× for wall time, and the CPU row omits the 67.6 h of local search, which would bring it to about 17×. §5.6 says the wall-time ratio shows the order of magnitude only, yet §6.1 builds its operator argument on that comparison.
**Evidence Anchor**: `text: §5.1 "Training ran for 200 epochs of 10,000 instances each" and §1 "Training on a single workstation finishes in less than a day"`
**Why it matters**: An operator cannot stop at epoch 133 without having seen epoch 200. A reader who notices this will discount every efficiency figure. The saving remains large on any accounting, so overstating it gains nothing.
**Suggestion**: Report the full-run cost as primary, or pre-specify a stopping rule and apply it. Report development cost at least in order of magnitude. Show calls, total CPU and same-hardware wall time together. If the abstract keeps one number, it should be the one an operator pays.
**Severity**: Major
**Confidence**: 4 — arithmetic on manuscript figures; the late-epoch rate is an inference from §5.6 and I have no epoch-level data

### W6: "Close on the benchmark, behind on synthetic" mixes protocols and instance sizes
**Problem**: The Conclusion contrasts +0.21% and +0.72% (eight views) with +2.60% (no augmentation), as though the difference were about the instance sets. §5.3 itself warns that the tables are not directly comparable. The abstract gives no protocol for the Real529 figure. Of Real529, 86% of instances have $J \le 50$, including the 220-instance band with 127 ties, so the +0.21% partly reflects composition. The abstract also leads with Real529, which the authors disclose is not held out, whereas §5.1 calls test500 the primary evidence.
**Evidence Anchor**: `text: §7 "Its quality is close to that baseline on the literature benchmark and on large instances"`
**Why it matters**: This sentence is what most readers will carry away. The available comparison at the same protocol on the same size range has not been reported. The result on the primary set should determine the headline, not the order of presentation.
**Suggestion**: Report all three sets under one protocol, and lead with the primary set. State the Real529 size mix, and give the gap for $J \ge 20$ next to the overall figure. Reserve "close" for a stated threshold.
**Severity**: Major
**Confidence**: 5 — direct reading of the manuscript's protocols

### W7: The aggregate metaheuristic comparison hides differences by instance class in both directions, and "long-budget" misdescribes the budget
**Problem**: §5.4 and §6.1 read Table 7 through its All column, which Poi325 dominates (325 of 529 instances). By subset, the picture differs in both directions. At 3×, ILS and VNS are already ahead of E133 by about 1.7 points on Cluster. At 10×, E133 is still ahead of both on Gam72 and on both Erd132 subsets, and behind only on Poi325. So neither "the policy wins at 3×" nor "metaheuristics win at 10×" is a general statement, and the abstract's sentence understates the method on two of the three sources. The "long-budget" label covers a mean of about 103 s on a single core (10× of 10.3 s). In every column of Table 7 G200 is better than E133, so the "fast" half of the selling point belongs to learned policies generally and not to this method. §1 uses an offshore installation as its example; if such targets cluster, as I would expect but the paper does not say, the relevant column is the unfavourable one.
**Evidence Anchor**: `table: Table 7, Poi325 Cluster column — E133 +2.10, ILS 3× +0.36, VNS 3× +0.36, G200 +1.76`
**Why it matters**: On my card's central question, the selling point does not survive as a general claim about speed against metaheuristics. It survives as a class-specific claim, conditional on W1. The advice in §6.1 ("minutes per instance" favours a metaheuristic, "seconds" favours the policy) is also too coarse, because instance class matters at least as much as the time budget.
**Suggestion**: Report the crossover by subset, adding a 1× row and intermediate budgets, and separate size from distribution, since the subsets mix both. Replace "long-budget" with the mean seconds. State the recommendation by instance class. Say explicitly that test-time preference over metaheuristics is shared with G200.
**Severity**: Major
**Confidence**: 5 — directly from Table 7 and §5.4; the clustering of offshore targets is my expectation and is posed as a question below

### W8: The most relevant competitors are absent
**Problem**: The comparison uses three in-house generic metaheuristics that call the SOCP on every order. Two rivals a field reader will ask about are missing. One is the two-stage heuristic of Li et al., which supplies the reference values and against which E133 is +1.35%, with no time comparison given. The other is a learning-free search using the paper's own table: solve one order, run LKH on the table, re-solve, repeat. §2.1 says the table "can be read as one such alternation step". §7 defers warm starts to future work, but that defers only the combination, not the test of whether DRL is needed at all.
**Evidence Anchor**: `absence: §5.4 and Table 7 — expected a time-matched comparison with the Li et al. heuristic and with a learning-free search on the table; checked §1, §2.1, §2.4, §5.4, §6.1, §7`
**Why it matters**: If the contribution is Proposition 1, the first question is whether the proposition alone beats a trained policy at equal time, without training at all. A generic metaheuristic that pays one SOCP per order is a weak opponent, and even that one wins in aggregate at 10×.
**Suggestion**: Add a table-based iterated search as a baseline on Real529 at the same budgets, and report Li et al.'s values with their published times, noting hardware differences. A learning-free win would be an important result and would change what the paper should claim.
**Severity**: Major
**Confidence**: 4 — field expertise in CVTSP heuristics; I have not run or seen any such comparison, so the outcome is open

### W9: No experiment varies the certificate, although the certificate is the claimed contribution
**Problem**: The paper's central contrast is that a bound with a known direction of error is better suited than a surrogate (§2.3). The planned ablations do not test that. Group B replaces table-based search with exact move evaluation. Group D alters expert diversity. No variant keeps the architecture and swaps the table for a cheap uncertified signal such as a closed-form proxy or an estimate matrix of the kind in Li et al. Nor does any variant remove the dense table reward while keeping imitation, so the abstract's "main training signal" is not separated from verified imitation labels.
**Evidence Anchor**: `absence: §5.7 Ablation study — expected a variant replacing the table with an uncertified cheap surrogate, and one removing the table from the reward; checked §2.3, §4.2, §4.5, §5.7, §6.3`
**Why it matters**: On focus 3 of my card, a one-paragraph proposition can carry a paper only if the empirical work shows the proposition is what delivers the result. With a bound excess still 33.6% at epoch 133, a reader may suspect that imitation does the work and that a cheaper, unguaranteed signal would do as well.
**Suggestion**: Add one ablation that keeps the verification step and replaces the certified table with an uncertified proxy, and one that removes the table term from the reward. Either outcome is reportable, and the second would redirect the claim to the screening rule.
**Severity**: Major
**Confidence**: 4 — design-level judgement; the outcome is open, and detailed design belongs to the methodology seat

### W10: The "pays once" framing is contradicted by the test-time evidence
**Problem**: §2.1 contrasts per-instance effort in existing methods with a trained policy that pays once. But 10.2 s of the 10.3 s single-core test time is Gurobi, about 99%, because every distinct route of up to eight views is solved exactly. The deployed system is therefore a candidate generator plus hundreds of SOCP solves per instance, needing a solver and 16 workers to reach seconds. At $J = 200$ the reported 5.6 s wall time implies over a minute on one core by my estimate, which is the regime in which §6.1 prefers a metaheuristic. The proposition is not used at test time at all, and "no change in test time" is partly a consequence of that.
**Evidence Anchor**: `text: §2.1 "whereas a trained policy pays once during training" and §5.6 "it is dominated by exact solves of the POMO routes"`
**Why it matters**: It weakens the "solutions quickly" half of the conclusion and leaves an obvious question unanswered: why does a method built on a solver-free bound not use it to reduce solves at deployment?
**Suggestion**: Correct the framing, state the deployed resource need, and test table-based screening of routes at inference (one anchor solve, then a few verifications) with its quality loss. This could become a true test-time advantage over G200, which the paper currently lacks.
**Severity**: Major
**Confidence**: 4 — arithmetic on reported times; the $J = 200$ single-core figure scales the mean parallel speed-up and is approximate

### W11: No operating setting is shown in which seconds matter
**Problem**: The recommendation depends on plans being needed in seconds and on frequent replanning, but no operational scenario, planning interval or decision time is given or sourced. The model is a static single-carrier, single-drone tour from depot to depot with all targets known. A replan during a mission starts at the carrier's current position with a subset of targets, so it is a different problem from the one trained and tested, and "fleet operator" goes beyond a one-carrier model. Launch and recovery times, weather and any stochastic element are absent. Yet those are what would drive replanning.
**Evidence Anchor**: `text: §6.1 "For a fleet operator that replans repeatedly, this favours a trained policy when the number of decisions is large or the replanning interval is short."`
**Why it matters**: If a vessel mission lasts hours and a plan can be computed in the minutes beforehand, the case for seconds disappears, and a longer metaheuristic run may simply be the better choice. The claim is plausible for some settings, but the paper does not identify any.
**Suggestion**: Identify one concrete planning setting with a cited decision time, or remove the claim. If replanning is kept, test instances that begin mid-mission from a non-depot position.
**Severity**: Major
**Confidence**: 3 — operational judgement; I have no data on the decision times of such operators

### W12: The one transferable insight rests on a negligible treatment
**Problem**: §6.3 concludes that tightness has low marginal value and offers this as a wider implication for learning with expensive objectives. The only evidence is a continuation run in which the bound tightened by 0.32% against an excess of 33.6%. A null result from so small a change says almost nothing about tightness. "Accurate nearby" is asserted without reported data, and the Group B result that "will quantify this claim" is pending while the prose already concludes. The word "substitute" is also strong given that the accurate signal remains 2.6% ahead.
**Evidence Anchor**: `text: §6.3 "on the evidence so far it is not the limiting factor" and "can substitute for an accurate one"`
**Why it matters**: A paper whose proposition is admitted to be simple needs a generalisable lesson, and this is its main candidate. As stated it does not follow from the experiment.
**Suggestion**: Test a materially tighter signal (for example, more tables from diverse solved routes) and show bound quality against distance from the anchor. Until then, phrase it as a hypothesis, and describe the "region we tested" numerically.
**Severity**: Major
**Confidence**: 4 — reasoning about the inference from the reported numbers; training-level causes are outside my focus

### W13: Transfer ability is attributed to the training signal despite an architecture difference
**Problem**: §5.1 says the models differ in architecture as well as signal and that the comparison is between recipes. §5.3 nonetheless concludes about learning from the table. The statement also rests on a positive gap on one seed.
**Evidence Anchor**: `text: §5.3 "learning from the table did not cost additional transfer ability"`
**Why it matters**: The claim is a causal one that the design cannot support, though the descriptive result stands.
**Suggestion**: Rephrase in terms of recipes, or support it with a run of the same architecture under both signals.
**Severity**: Minor
**Confidence**: 4 — direct reading

### W14: Route diversity is given as the main cause on thin evidence
**Problem**: The Conclusion directs future work at diversity on the basis of one overlap comparison (0.62 against 0.53) at $J = 100$, from epoch 163 rather than E133, and without control for the architecture difference. §4.6 and §6.4 read as settled.
**Evidence Anchor**: `text: §7 "which appears to be the main source of the remaining gap"`
**Why it matters**: A reader may take it as a finding. The alternatives (network size, a reward with a 34% excess, imitation pressure) are not excluded.
**Suggestion**: Keep it as a hypothesis, list the rival explanations, and report the overlap for E133 at all sizes.
**Severity**: Minor
**Confidence**: 3 — adjacent to my focus

### W15: The gap table column "Exact feedback in training" is not well defined
**Problem**: Table 1 marks works with closed-form objectives "No", though their feedback is exact by construction, and marks methods with no training at all "No" as well. Only this paper can then score "Yes" in all columns. §2.3 describes the Kim et al. method as reusing evaluated solutions, which is exact feedback on those solutions.
**Evidence Anchor**: `table: Table 1, column "Exact feedback in training" — rows Kim et al., Pirnay et al./Luo et al., Li et al.`
**Why it matters**: The claimed gap is real, and a table that appears tailored makes it look weaker than it is.
**Suggestion**: Use "n/a" for non-learning work and redefine the column, for example as "training evaluation requires a solver per route".
**Severity**: Minor
**Confidence**: 4 — direct reading; the description of Kim et al. relies on the manuscript's own wording

### W16: The generality claim in the Conclusion has no worked case
**Problem**: The Conclusion says the idea carries over to other routing problems, whereas §6.4 notes it fails if time windows couple the sorties. No second problem is shown.
**Evidence Anchor**: `text: §7 "The idea carries over to other routing problems"`
**Why it matters**: The contribution's scope is a key part of its significance, and at present it is asserted.
**Suggestion**: Name at least one variant and show that its sortie constraints separate, or phrase the sentence conditionally, matching §6.4.
**Severity**: Minor
**Confidence**: 4 — logical reading; the proof-level extension is outside my focus

### W17: The baseline's route count in training is not stated, and the 78× depends on it
**Problem**: Table 10 implies about 15 solver calls per training instance for G200. With $J$ averaging 60 and "one call per route", a reader expects about 60. The manuscript nowhere states how many routes G200 or the proposed method use in training. The 78× is a function of this free baseline setting, so a baseline with fewer routes per instance would shrink the ratio.
**Evidence Anchor**: `table: Table 10, row "Gurobi calls in training" — 299,437,760 calls on 20,000,000 instances`
**Why it matters**: The headline ratio cannot be interpreted without it, although the fix is one sentence plus a remark on whether the baseline's setting is cost-efficient.
**Suggestion**: State $P$ as a function of $J$ for both models in §5.1, and note the sensitivity of the ratio to it.
**Severity**: Minor
**Confidence**: 4 — arithmetic on manuscript figures

---

## Detailed Comments *

### Journal Fit (field reader; no venue-alignment claim)
The topic is drone and carrier routing, and the problem is one this readership follows. The treatment is algorithm-first: the research question, title, contributions and main table all concern training cost (W2). Sentences such as those noting that the line of work is "reaching transport journals" gesture at an audience without giving it anything specific to use. Fit as logistics scholarship depends on W1, W2 and W4.

### Originality
The source of originality is a new combination, not a new theorem, as the authors say. The narrow claim (a certified bound from one primal solution, used for training rewards and expert search) looks defensible on the evidence shown, subject to the unread works below. The claim is narrow by construction, being a conjunction of features, so its worth depends on what the combination delivers. On focus 3: a simple proposition can carry a paper if the empirical part shows it is the cause of a result that matters. Here the causal link is untested (W9) and the learning-free alternative is unexamined (W8).

### Significance
Direct answer to my card's question. The 2.6% deficit is survivable only if the claim is about the cost of obtaining a policy where a new policy is really needed, and at matched quality (W1, W3). The metaheuristic result makes the test-time preference class-specific: E133 already loses on Cluster at 3× and on Poi325 Regular at 10×, holds on the Gambella and Erdoğan sets even at 10×, and is behind G200 in every column (W7). So as drafted the selling point survives as "one can get a somewhat worse learned solver far more cheaply". That is a useful methodological finding for neural combinatorial optimisation with expensive objectives. It is not yet a finding for operators. Impact would be sub-field level unless the retraining case or a test-time use of the bound succeeds.

### Structural Coherence
Title to Conclusion is coherent as an algorithm paper: the question posed in §1 is answered in §7. The break is between that question and the operational conclusion ("suits settings where plans are needed quickly and where retraining has to be affordable"), which no section tests. §6.1 also over-promises relative to §5.6, as noted in W5.

### Title & Abstract
The title is accurate and names the method and problem, but no logistics decision. The abstract is unusually honest. Its problems are a missing protocol for Real529, the order of evidence, and unqualified ratios (W3, W6). The bound's looseness is not mentioned next to "certified", although 33.6% excess is relevant to anyone reading "main training signal".

### Conclusion
It answers the stated question and restates the deficits. Its last sentence of the first paragraph ("so the method suits...") and the generality claim go further than the evidence (W1, W16).

### Dependence on pending results
| Pending item | Does the argument depend on it? | If it goes the wrong way |
|---|---|---|
| Group A and 24-hour curve (§5.6, §5.7) | Yes. This is the most decision-relevant pending work (W3). | If POMO on exact rewards at about E133's wall time on the same hardware matches or nearly matches E133, the efficiency claim collapses and the paper has no selling point. A clearly worse result gives the ratios real meaning. |
| E133 eight views on test500 | Yes for abstract and conclusion (W6). | A gap near the +1.07% of E150 helps; the abstract must then be rewritten. A gap that stays large confirms an in-range weakness and removes the "augmentation narrows it" explanation. The text must not be written before the number exists. |
| Group B | Yes for §6.3 and for "what the table saves". | If exact move evaluation at 100 solves per instance costs little more time or is much better, the table is not the source of the savings, and §6.3 fails. |
| Group D | Partly. It speaks to expert diversity, while the diagnosed issue is POMO route diversity. | A null result leaves the diversity story untested rather than refuted, and W14 stands. |
| Two further seeds | Yes for any gap near zero. | Between-seed spread comparable to +0.21% would remove "close to baseline" as a claim. Ratios are little affected. G200 is also a single run. |
| Sensitivity (§6.2) | Yes for the logistics reading (W2). | If removed as the placeholder allows, the paper has no operational result left. |
| Liu, Yan, Lee full texts | Partly. | Could narrow Table 1 and the "no learning method" sentence. The certified-bound claim likely survives unless one of them reuses a primal solution in this way, which is not known. |
| Vieira BKS merge, new-best counts | Little. | Gaps to BKS can only rise or hold. The new-best statement could shrink, and the negative Cluster gaps are worth a check against model differences. |
| Full MISOCP run | No. It sets scale only. | No effect on the core claims. |

---

## Questions for Authors *

1. Which concrete operator changes require retraining, given that speed ratio and endurance are network inputs, and outside what parameter range or instance distribution does a trained network fail?
2. At what stage of G200's own training was E133's validation quality first reached, and what had it cost by then?
3. What rule, fixed in advance, would have stopped the proposed run, and what does a full run cost in calls, CPU and wall time?
4. Are the targets in the intended maritime application uniform or clustered, and how long does a planner actually have to produce a plan?

---

## Minor Issues

### Language / Grammar
- §5.4: "This holds for 25 instances from one source" follows the sentence on ties, so its antecedent reads wrongly. It refers to the $J = 200$ result.
- Contribution 2 lists anchor, policy and expert solves but omits refresh, which is 17.9% of calls in Table 8.
- §5.1: "It was not retrained for this paper" has an unclear subject after the sentence on model comparison.

### Citation Format
- Several classic works are cited as arXiv items, though to my knowledge conference versions exist (POMO, the attention model, pointer networks, Nazari et al., Sym-NCO, efficient active search). Please check and cite the archival versions.
- Keys and years differ: `li2025` (2026), `ren2026` (2027), `kool2019` (2018), `hottung2022` (2021), `kim2024` (2023), `ferber2023` (2022). The keys are harmless, but the `ren2026` year should be checked against the issue before submission.
- The Table 1 caption hard-codes author–year strings that should follow the citation style.

### Figures and Tables
- Table 7 defines its gap with UB in the denominator, while all other gaps use the baseline. A one-line note would prevent confusion.
- Table 10 is discussed first in §5.6 but appears after Tables 8 and 9.
- Table 10 labels 0.78 s for both models. Please note that this is rounded, since the networks and distinct-route counts differ.
- §4.4 gates on a validation set of 1,000 instances, §5.1 selects on 2,000, and §4.7 reports 266,000 validation calls, which equals 2,000 × 133. Please say whether the gate is a subset and whether its calls are counted.
- Table 10's estimated G200 CPU implies 0.031 s per call, while E133's measured figures imply 0.081 s. The estimate at $J \approx 60$ may be low, and the difference deserves a sentence.

### Layout
- Table 3 is an algorithm set as a table; an algorithm environment would be clearer.

---

## Criterion-Bound Judgements *

Calibration status: `NOT_CALIBRATED`

Criterion source below is `references/quality_rubrics.md` (dimension named) or Card #1 focus items. No venue criterion is applied (`criteria_binding_unavailable`).

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Logistics contribution | Card #1 focus 1 | DOES_NOT_MEET | W2; W11 | Question, results and advice concern solver budgets | Could change if §6.2 is written with substance | Yes: decides whether this is a logistics paper |
| Managerial insight | Card #1 focus 2 | DOES_NOT_MEET | W1; W4 | Advice is not derived from an experiment or a trade-off | Repairable with one shifted-distribution experiment and a break-even | Yes: the main unresolved criterion |
| Contribution beyond the proposition | Card #1 focus 3 | PARTLY_MEETS | W8; W9; S4 | Combination is new as far as shown, but the proposition's causal role is untested | Depends on Groups A and B | Yes |
| Originality | Rubric Dimension 1 | PARTLY_MEETS | S4; W15 | Narrow gap is defensible and modestly claimed; three close works unread | Reliance on the authors' search; domain seat to verify | Yes, moderately |
| Methodological Rigor | Rubric Dimension 2 | NOT_ASSESSED | — | Outside this seat's remit, except budget accounting in W5 | Methodology seat | No |
| Evidence Sufficiency | Rubric Dimension 3 | PARTLY_MEETS | S5; W3; W5; W6 | Quality and call data are solid; practical claims lack evidence of the right type | Several core results pending; one seed | Yes |
| Argument Coherence | Rubric Dimension 4 | PARTLY_MEETS | W1; W10; W12 | Algorithmic argument is traceable; operational argument contains internal contradictions | none identified | Yes |
| Writing Quality | Rubric Dimension 5 | MEETS | S1 | Clear, plain and candid prose; minor items only | First draft | No |
| Literature Integration | Rubric Dimension 6 | PARTLY_MEETS | W8; W15 | Good coverage of closest work, but rivals not brought into the experiments | Detailed coverage belongs to the domain seat | Partly |
| Significance & Impact | Rubric Dimension 7 | PARTLY_MEETS | W1; W7 | Useful to expensive-objective NCO; operator value unproven | Could rise with a retraining result or test-time use of the bound | Yes |

The Major Revision signal rests on these unresolved decision-bearing criteria: the operator case for cheap training (W1, repairable by a new experiment or by repositioning the claim), managerial derivation (W2, W4, W11), and cost evidence at matched quality with honest accounting (W3, W5). The sound proposition and exact call accounting do not offset these. If the shifted-distribution experiment fails and the matched-quality comparison is unfavourable, the paper should be repositioned as a methods paper with a narrower claim.
