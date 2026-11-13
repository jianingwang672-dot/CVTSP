   Deep Reinforcement Learning with Sparse Exact
 Feedback for the Carrier–Vehicle Traveling Salesman
                      Problem
           [Author names, affiliations and corresponding author]




Abstract
In the carrier–vehicle traveling salesman problem (CVTSP), a carrier launches
and recovers a drone at continuous points while the drone visits a set of
targets. Evaluating one visiting order requires solving a second-order cone
program (SOCP), so training a deep reinforcement learning (DRL) policy on
exact rewards consumes hundreds of millions of solver calls. We show that
one optimal SOCP solution yields a certified upper bound on the cost of ev-
ery visiting order, because fixed take-off and landing points remain feasible
when the order changes. We use this bound as the main training signal: it
supplies dense rewards, guides solver-free local search and screens candidate
expert routes, while exact solves are reserved for a few routes per instance.
Compared with POMO trained on exact rewards for every route, the pro-
posed method uses 15 times fewer training instances and 78 times fewer solver
calls at the same test time. On 529 literature benchmark instances its mean
per-instance gap to that baseline is +0.21%. The gap is +2.60% on 500 syn-
thetic instances without augmentation and +0.72% on instances with 150 to
500 targets with eightfold augmentation. Long-budget metaheuristics obtain
better solutions when given ten times the computing time. The method is
therefore most useful when training budgets are limited and solutions are
needed quickly.
Keywords: Carrier–vehicle routing, Drone logistics, Deep reinforcement
learning, Second-order cone programming, Certified upper bound


1. Introduction
    A drone with a short flight range can serve targets far beyond that range
if a long-range platform carries it. A ship can launch an unmanned aerial
vehicle (UAV) towards an offshore installation, keep moving, and recover the
UAV further along its course. The carrier–vehicle traveling salesman problem
(CVTSP), introduced by Garone et al. (2011), models this operation. A
carrier leaves a depot, releases a drone before each target, recovers it after
the visit and returns to the depot once all targets have been served. The
planner chooses the visiting order and the take-off and landing point of each
sortie in order to minimise mission completion time.
    The problem combines two kinds of decisions. The visiting order is a
permutation, and the sortie points lie in the continuous plane. Once the
order is fixed, the remaining problem is convex and can be written as an
SOCP (Lobo et al., 1998). The joint problem is a mixed-integer SOCP.
Gambella et al. (2018) and Erdoğan and Yıldırım (2021) solved it exactly
for small instances, and Vieira et al. (2026) report optimal solutions for all
benchmark instances with up to 25 customers and for one instance with 35
customers. Larger instances rely on heuristics such as the two-stage iterative
heuristic of Li et al. (2026). Ren et al. (2027) extend the model to a vessel
that carries several UAVs, which shows that the setting remains an active
research problem in maritime logistics.
    Deep reinforcement learning has become a practical alternative for routing
problems when solutions must be produced quickly. A trained policy pro-
duces a route in a single forward pass, and methods such as POMO (Kwon
et al., 2020) reach small gaps on the traveling salesman problem. These
methods rest on a property that the CVTSP lacks: the cost of a route is a
sum of known edge lengths. In the CVTSP, the cost of a route is the optimal
value of an SOCP. Our reference baseline, POMO trained on an exact solver
reward for every sampled route, made about 3 × 108 solver calls over 200
epochs. That cost must be paid again whenever the instance distribution or
the operating constraints change. For a maritime or humanitarian operator,
a change of vessel, drone model or service region is a change of distribution,
so training cost is a recurring operational expense and not a one-off research
cost.
    This paper asks whether most of those calls are necessary. The answer
rests on one structural observation. In the fixed-order SOCP, the constraints
on a sortie concern one target only, and the visiting order enters only through
the carrier legs between sorties. If the sortie points from one optimal solution
are kept and the order is changed, every sortie remains feasible and only the
carrier legs change. The resulting value is an upper bound on the optimal cost
of the new order, and it can be computed by adding up entries of a table. One

                                       2
exact solve thus says something about every route of the instance, including
the routes that will never be solved.
    In the literature we searched (OpenAlex, arXiv, Crossref and web sources,
last searched 4 November 2026), we found no DRL method for routing prob-
lems with a continuous-optimisation route cost that reuses one exact primal
solution as a certified upper bound on all visiting orders, for both training re-
wards and expert search. The closest work is Li et al. (2026), whose heuristic
also derives an edge matrix from solved subproblems. Their matrix updates
traversed arcs only, keeps estimates elsewhere and carries no bound. We
return to this comparison in Section 2.4.
    The paper makes three contributions.
  1. We state and prove that one optimal fixed-order solution yields a table
     whose route sums bound the exact cost of every visiting order from
     above and match it at the solved order. The proof is short because it
     uses the structure of the CVTSP directly. The contribution lies in how
     this bound is put to work.
  2. We build a training method around the table. Dense rewards, local
     search and expert screening use the table without calling a solver, and
     exact solves are reserved for the anchor route, for one policy route and
     for expert verification. The method uses 2.89 solver calls per training
     instance on average.
  3. We compare the method with POMO trained on exact rewards for
     every route, and we report the results by instance size. The method
     uses 15 times fewer training instances and 78 times fewer solver calls,
     with no change in test time. It remains behind the baseline on synthetic
     instances, and we quantify where and by how much.
    The practical result is a change in what a learned CVTSP solver costs
to obtain. Training on a single workstation finishes in less than a day, com-
pared with more than a week for the baseline on a larger worker pool. The
paper also reports where this comes at a price: the method does not match
the baseline on instances from its own training distribution, and metaheuris-
tics with long time budgets obtain better solutions. We report both results
alongside the savings, because a planner choosing a method needs all three.
    Section 2 reviews related work. Section 3 defines the problem and Sec-
tion 4 presents the method. Section 5 reports the experiments, Section 6
discusses practical implications and Section 7 concludes.


                                       3
2. Related work
  Three strands of work bear on this paper: solution methods for the
CVTSP, DRL for routing, and learning under expensive objectives.

2.1. The CVTSP and its extensions
    Garone et al. (2011) introduced the problem and proposed a two-step ap-
proach that first fixes an order and then optimises the sortie points. Garone
et al. (2014) extended it to carriers that visit several targets per sortie. Gam-
bella et al. (2018) gave the first exact mixed-integer SOCP approach, solving
instances with up to 15 targets. Erdoğan and Yıldırım (2021) tightened the
formulation and added an iterated local search, and Poikonen and Golden
(2020) studied the closely related mothership and drone routing problem with
instances of up to 200 targets. Fahradyan et al. (2020) considered several
carrier–vehicle pairs, and Li et al. (2024) studied a joint vessel–UAV routing
variant with a tabu search heuristic.
    Recent work has moved the exact and heuristic frontiers. Li et al. (2026)
developed exact methods and a two-stage iterative heuristic, and their results
supply the reference values used in this paper. Vieira et al. (2026) propose a
linearisation-based refinement algorithm that solves all benchmark instances
with up to 25 customers. Ren et al. (2027) address a vessel with multiple
UAVs through a mixed-integer SOCP and an adaptive large neighbourhood
search. CVTSP research is one branch of a wider literature on drone-assisted
routing (Murray and Chu, 2015; Agatz et al., 2018; Otto et al., 2018; Macrina
et al., 2020; Chung et al., 2020), and transport journals have recently paid
attention to operational uncertainty in such systems (Zhao et al., 2022). All
of the CVTSP methods above solve each instance from scratch. They spend
their computing time per instance, whereas a trained policy pays once during
training.
    Two observations from this literature shape our design. The first is that
most of the heuristics we reviewed rest on the fixed-order problem: a can-
didate order has to be solved before its true cost is known. The second is
that the order and the sortie points are strongly coupled, since a good order
depends on where the carrier will be, and the carrier’s position depends on
the order. Two-stage methods handle this coupling by alternating between
the two levels. Our table can be read as one such alternation step, frozen
and reused across a whole search.



                                       4
2.2. Deep reinforcement learning for routing
    Learning to construct routes began with pointer networks (Vinyals et al.,
2015) and was extended to vehicle routing by Nazari et al. (2018). The
attention model of Kool et al. (2018) established the encoder–decoder archi-
tecture that most later work adopts. POMO (Kwon et al., 2020) exploits
the multiple symmetric starts of a route, and Sym-NCO (Kim et al., 2022)
and efficient active search (Hottung et al., 2021) add symmetry training and
instance-level adaptation. Improvement-based policies (Wu et al., 2022) and
heavy-decoder models (Luo et al., 2023) target better quality and larger sizes.
Surveys by Bengio et al. (2021) and Mazyavkina et al. (2021) cover the field,
and Wu et al. (2026) show that this line of work is reaching transport jour-
nals through clustered vehicle routing. Across this body of work, the reward
signal is rarely a concern, since an additional route evaluation costs a few
floating-point operations.
    Most of these solvers evaluate a route in closed form, as a sum of distances
with a feasibility check. Work on routing in continuous space comes closer to
our setting. Fan et al. (2025) address the close-enough TSP, in which a tour
only needs to reach a neighbourhood of each target. They discretise candidate
waypoints, so that the reward remains a tour length and the training loop
avoids solving a continuous subproblem. Liu et al. (2026) apply reinforcement
learning to eVTOL-and-drone routing in continuous space, Yan et al. (2026)
study truck-based mothership delivery to moving targets with multi-agent
reinforcement learning, and Lee et al. (2023) apply reinforcement learning to
mothership routing. [Confirm the methods of Liu, Yan and Lee from their
full texts; the wording here rests on titles and search summaries.] Within the
sources we searched, we found no learning method stated for the CVTSP as
defined by Garone et al. (2011). That finding is bounded by our search and
does not extend to mothership routing in general. DRL has clearly reached
related problems, so we do not argue that this class of problem is untouched.
Our argument concerns the cost of training when each reward requires a
solver.

2.3. Expensive objectives and self-improvement
    Expensive objectives have received direct attention in neural combinato-
rial optimisation. Kim et al. (2023) propose symmetric replay training, which
reuses solutions that have already been evaluated through their symmetries
and thereby saves objective evaluations. It offers no estimate for a route that


                                       5
was never evaluated. Sun et al. (2026) train a neural network offline to esti-
mate drone flight times from trajectory-optimisation data and use it inside
a heuristic for truck and drone delivery. SurCo (Ferber et al., 2022) learns
linear surrogate costs for nonlinear combinatorial problems. These methods
learn or reuse a cost signal, and none of those we read provides a guarantee
on routes outside the evaluated set. A learned surrogate can be accurate on
average, but it can also be wrong in a direction that is hard to detect, and
its training set has to be produced in advance. A bound derived from the
solution of the instance at hand needs no training data and has a known
direction of error, which is what makes it usable as an acceptance test.
    Self-improvement offers a second route to lower training cost, in which
a policy trains on labels that it produces itself. Luo et al. (2023) train by
supervised learning on small instances, Pirnay and Grimm (2024) and Luo
et al. (2024) generate pseudo-labels from the current model, and Luttmann
and Xie (2025) extend the idea to multi-action settings. On the problems
these papers study, comparing two candidate solutions is cheap, so labels
can be accepted or rejected by direct evaluation. In the CVTSP, the same
test costs one SOCP per candidate. Our method keeps the self-improvement
loop and uses the bound to decide which candidates deserve that cost. This
changes the economics of the loop. Generating a hundred candidates is cheap,
but verifying a hundred candidates is not, so the screening step becomes the
part of the method that determines its budget.

2.4. Research gap
    Table 1 compares the closest work on five properties. Among the sources
we read, none combines a continuous route-cost subproblem, a training signal
for routes that were never solved, and a guarantee on that signal. The
contrast with Li et al. (2026) is the sharpest. Their approximate matrix is
initialised in closed form and, after a fixed-order problem is solved, updated
only on traversed arcs; the other entries keep their prior values. The matrix
guides a heuristic and is regenerated for diversification. Our table is built
from one primal solution in a single step, assigns a valid value to every arc
and bounds the cost of every order. Fixing part of a solution and reoptimising
the rest is an established pattern in routing, so we do not claim Proposition 1
as a new idea in itself. Our claim concerns its use as the main training signal
for a DRL policy.



                                      6
Table 1: Closest work compared on properties relevant to this paper. Liu et al. (2026),
Yan et al. (2026) and Lee et al. (2023) are omitted because their full texts have not been
read.
                                             Continuous     Exact feedback       Value for      Bound on     DRL builds
Work
                                             subproblem       in training     unsolved routes   all orders    the route
Li et al. (2026)                                Yes                 No          Estimate           No           No
Vieira et al. (2026)                            Yes                 No             No              No           No
Ren et al. (2027)                               Yes                 No             No              No           No
Kim et al. (2023)                                No                 No             No              No           Yes
Sun et al. (2026)                               Yes            Offline data     Estimate           No           No
Fan et al. (2025)                            Discretised            No         Closed form         No           Yes
Pirnay and Grimm (2024), Luo et al. (2024)       No                 No         Closed form         No           Yes
This paper                                      Yes            Yes, sparse      Certified          Yes          Yes



3. Problem formulation
    Let d ∈ R2 be the depot and p1 , . . . , pJ ∈ R2 the targets. The carrier
moves at speed vc and the drone at speed vd > vc . The drone can fly for
at most E time units per sortie. Each target must be visited in exactly one
sortie. For target i, the drone leaves the carrier at take-off point si , flies to
pi and returns to the carrier at landing point li . The carrier travels from si
to li during the sortie.

3.1. The fixed-order problem
    Let π = (π1 , . . . , πJ ) be a visiting order, with the conventions that lπ0
and sπJ+1 denote the depot. Let t1i and t2i be the outbound and inbound
flight times, τi the sortie duration, and Tk the carrier travel time on the leg
between consecutive sorties. The carrier–vehicle problem for order π is
                                  J
                                  X              J
                                                 X
             f (π) = min                  τi +         Tk                                                             (1)
                                   i=1           k=0
                          s.t.    ksi − pi k ≤ vd t1i ,                              i = 1, . . . , J,                (2)
                                  kli − pi k ≤ vd t2i ,                              i = 1, . . . , J,                (3)
                                  kli − si k ≤ vc τi ,                               i = 1, . . . , J,                (4)
                                  τi = t1i + t2i , τi ≤ E,                           i = 1, . . . , J,                (5)
                                     sπk+1 − lπk ≤ vc Tk ,                           k = 0, . . . , J,                (6)
                                  t1i ,
                                      t2i , τi , Tk ≥ 0.                                                              (7)

Every constraint is a second-order cone or a linear constraint, so the problem
is an SOCP (Lobo et al., 1998). The CVTSP asks for minπ f (π). Mission

                                                           7
                                   Table 2: Main notation.
     Symbol           Meaning
     J                Number of targets
     d, pi            Depot and target locations
     vc , vd          Carrier and drone speeds
     E                Maximum sortie duration
     π                Visiting order, with π0 = πJ+1 = 0 denoting the depot
     si , li          Take-off and landing points of the sortie for target i
     t1i , t2i , τi   Outbound time, inbound time and duration of a sortie
     Tk               Carrier travel time on leg k
     f (π)            Optimal value of the fixed-order problem for π
     q                Anchor route, solved exactly
     Cq, F q          Directed table and total sortie time from the solution for q
     Uq (π)           Certified upper bound on f (π)


completion time is the objective in (1), because the carrier is always either
on a leg or within a sortie. Table 2 lists the main notation.

3.2. Cost structure
    Two features of this formulation drive the method. First, constraints
(2)–(5) involve the variables of a single target and make no reference to
π. Second, the order appears only in (6), which links the landing point
of one target to the take-off point of the next. Any solution that fixes all
sortie variables therefore induces a closed-form cost for every order, since
the leg times are the only remaining quantities. Evaluating a fixed-order
problem with J targets takes tens of milliseconds at moderate sizes in our
implementation, about 0.03 s per call at J ≈ 60, while evaluating a table
sum takes microseconds. The number of orders that can be scored in one
solve time differs by several orders of magnitude, which is the leverage the
method relies on. This is the structure that the next section exploits. The
policy in Section 4 selects π, and the SOCP determines the sortie points, so
the method follows the natural decomposition into a discrete upper level and
a continuous lower level. Only the upper level is learned.




                                             8
        Table 3: One training step for a batch of instances. Exact solves are marked.
For each instance in the batch:
   1. Sample 200 routes from the frozen reference policy.
   2. Rank them with the edge predictor and choose the anchor q.
   3. Exact solve: solve the SOCP for q and build the table C q .
   4. Score all sampled routes with Uq and select 30 starts.
   5. Run 2-opt on C q from each start, and LKH from a share of the starts when
      J ≥ 60.
   6. Exact solve: if the best route has Uq below f (q), solve it. Otherwise, with
      the near-miss probability, solve the closest route.
   7. Exact solves: for J ≥ 60 after epoch 60, run up to two refresh rounds of
      the search, with one solve per round.
For the batch as a whole:
   8. Decode the POMO routes of the learning policy and score them with the
      tables.
   9. Exact solve: solve the best-scoring route of each instance.
  10. Update the policy by REINFORCE on the hybrid reward.
  11. Update the policy by cross-entropy on the verified expert routes.
  12. Update the edge predictor on the tables.


4. Methodology
4.1. Overview
    Table 3 summarises one training step for a batch of instances. A frozen
reference policy samples 200 candidate routes per instance. A small edge
predictor selects one of them as the anchor, and the anchor is solved exactly.
The solution yields a table C that scores every candidate. Local search on
the table produces an expert route, which is verified exactly before it can
serve as an imitation label. The learning policy then receives two updates: a
reinforcement step on rewards drawn mainly from the table, and an imitation
step on the verified experts. The number of exact solves per instance is
independent of the number of POMO routes. This independence is the source
of the savings. In POMO with an exact reward, a batch of B instances with
P routes costs BP solves. Here it costs about 3B, plus a refresh term for
large instances, regardless of P .



                                             9
4.2. A certified upper bound on all visiting orders
    Let q be a visiting order and let (s∗ , l∗ , τ ∗ , T ∗ ) be an optimal solution of
the fixed-order
              P problem   for q, with each sortie variable indexed by target.
Write F q =       τ
                 i i
                    ∗
                      and define a directed table on the nodes {0, 1, . . . , J},
with node 0 the depot:
                   s∗j − d                   kd − li∗ k                  s∗j − li∗
           q
          C0j =            ,        q
                                   Ci0 =                ,       Cijq =             .   (8)
                      vc                        vc                           vc
For any order π, define the route value
                                   J
                                   X
                  Uq (π) = F + q
                                         Cπqk πk+1 ,        π0 = πJ+1 = 0.             (9)
                                   k=0

Proposition 1. For every visiting order π, f (π) ≤ Uq (π). Moreover, f (q) =
Uq (q).
Proof. Fix π. Keep s∗i , li∗ , t1∗
                                i , ti and τi for each target and set Tk = Cπk πk+1 .
                                     2∗     ∗                               q

Constraints (2)–(5) involve only these sortie variables and do not depend on
the order, so they hold because they hold for q. Constraint (6) holds with
equality by construction, and non-negativity is immediate. The point is thus
feasible for the fixed-order problem of π, with objective value Uq (π). Since
f (π) is the minimum over feasible points, f (π) ≤ Uq (π). For π = q, each
Tk∗ is at its lower bound at an optimal solution because the objective strictly
increases in Tk , which gives Uq (q) = f (q).
    Three conditions apply in practice. The anchor must be solved to opti-
mality; a merely feasible solution would still give a valid upper bound, but
the equality at q could fail. In the implementation, every anchor solution
passes an audit of all constraints with a tolerance of 10−5 before its table is
used, and the bound therefore holds up to that tolerance. The training code
also checks the inequality whenever an expert, near-miss or refresh route is
solved exactly, and stops on a violation. The final run completed without a
stop.
Remark 1. Several tables can be combined only by minimising over complete
routes, minq Uq (π). An entry-wise minimum of tables would splice together
sortie points from different solutions and loses the guarantee, because the im-
plied take-off and landing point of one target would differ between its incoming
and outgoing arcs.

                                              10
    The table is asymmetric, so local search must account for the reversal
of internal arcs. With prefix sums of the differences between forward and
reverse arc costs, a 2-opt move is scored in constant time, without any solver
call.

4.3. Policy network
    The learning policy is an attention encoder–decoder of the POMO type
(Kwon et al., 2020), with 4 encoder layers, embedding dimension 256, 16 at-
tention heads and feed-forward dimension 512. Node features are the depot-
relative coordinates scaled by instance size, the distance to the depot, a
depot indicator, the speed ratio vc /vd and the drone reach vd E relative to
the instance scale. The speed and endurance features let one network serve
instances with different operating parameters. The policy decodes one route
per POMO start, with starts taken as the targets nearest to the depot.

4.4. Anchor selection and solver-free expert search
    Two policies interact during training. The learning policy is updated
at every step. A frozen reference policy generates search material and is
replaced only when the learning policy obtains a strictly better exact mean
on a fixed validation set of 1,000 instances. This gate ensures that the expert
search starts from routes of validated quality.
    An edge predictor maps the detached encoder output to a non-negative
directed table of size (J +1)×(J +1). It is trained with a Huber loss towards
tables built from exact solutions, and its loss does not reach the policy. Its
only role is to rank the 200 reference routes and choose the anchor, so a poor
prediction costs search quality and cannot affect validity.
    The anchor table scores all sampled routes. We select 30 starts: the 20
routes with the lowest bound, and 10 further routes chosen for diversity from
a pool of the 60 next-best routes within a relative gap of 5%. From each
start, directed 2-opt runs on the table for at most 30 moves and accepts only
strict decreases of Uq . Local search on the table is complementary to LKH
(Lin and Kernighan, 1973; Helsgaun, 2000): on large instances, a share of the
starts is also passed to LKH as an asymmetric TSP on C q . In every case the
search minimises a surrogate that can only overestimate the true cost. The
LKH time budget for all participating starts is deterministic, 1.5 × 10−4 J 2
seconds scaled by the LKH share, so that the search cost of a batch can be
predicted in advance.


                                      11
    Proposition 1 gives the screening rule. If a route satisfies Uq (π) < f (q),
then f (π) < f (q) holds without any solve, so the route is a certified im-
provement. Its exact cost is then computed once, and the route becomes
the expert. If no route clears the threshold, the anchor remains the expert.
Because the bound is conservative, some truly better routes fail this test.
With a small probability, the near-miss rule therefore solves the route with
the smallest relative gap above the threshold and accepts it if it is exactly
better. The probability adapts to training progress: for small instances it
moves between 1% and 5% as the imitation accuracy and the share of policy
routes that beat the expert rise. Each instance has a budget of one such
verification, and the solve is charged whether or not the route is accepted. If
the exact cost of a near-miss route exceeds its bound, the run stops, because
that would falsify Proposition 1.
4.5. Learning from sparse exact feedback
    Each instance yields three kinds of cost information for the POMO routes
π , . . . , π P of the learning policy. The anchor table and the expert table
  1

bound every route, and the code takes the route-wise minimum of the two.
The route with the lowest bound is then solved exactly. Writing cp for the
exact cost of that route and for the bound of every other route, the dense
advantage is
                                             c̄ − cp
                               ap =                         ,            (10)
                                     max{σ(c), 10−6 }
where c̄ and σ(c) are the mean and standard deviation over the POMO routes
of the instance. Standardising within each instance prevents hard instances
from dominating the gradient. A second term compares the exactly solved
policy route π sel with the verified expert π E ,
                                    f (π E ) − f (π sel )
                                                                
                          ex
                         a = clip                         , −1, 1 .      (11)
                                         |f (π E )|
The reinforcement loss combines the two terms using REINFORCE (Williams,
1992):
                      1 X p                    w X ex
            LRL = −         ab log pθ (πbp ) −     a log pθ (πbsel ), (12)
                    BP b,p                     B b b
where B is the batch size. The weight w rises linearly from 0 to 1 as the
imitation accuracy of the previous epoch moves from 0.50 to 0.85, so the com-
parison with the expert begins only when the policy can produce competitive
routes.

                                      12
    A separate optimiser step performs teacher-forced cross-entropy on the
expert sequences. Each batch mixes current experts with a replay buffer of
50,000 experts, which supplies 30% of the batch. Only verified experts enter
this update, so an error in the bound can waste a solve but can never become
a label. Imitation runs for five inner epochs at each update on batches of
1,000 sequences, split into micro-batches of 128. The two updates use a
common network and take separate optimiser steps.

4.6. Progressive search
     Search effort is scheduled by instance size and training stage, because a
strong expert helps large instances and hurts small ones. Earlier runs showed
that small instances reach their best validation value early and then deteri-
orate, a pattern we traced to falling route diversity in the POMO output.
Instances with J < 60 therefore keep the plain 2-opt search throughout. For
J ≥ 60, a ramp r = clip((e − 40)/60, 0, 1) at epoch e raises the fraction
of LKH starts to r/2 and raises the near-miss probability towards 1. After
epoch 60, large instances also receive two refresh rounds: the most recently
solved route supplies a new table, which is tight around that route, and the
search continues on it. Each round solves at most one additional route.
     The quantity of imitation labels is normalised at the same time. An im-
proved expert is repeated in the imitation batch so that the expected number
of samples per instance stays near 1.44, with the repeat weight clipped to
[1, 5]. Without this step, the imitation pressure would grow as the search
improves.
     The values of these settings came from earlier development runs. We
compared the expert operators on a frozen policy and found that swap, 2-
opt and LKH were complementary. Those runs also showed that the benefit
of a stronger expert concentrates at J ≥ 60, which set the size threshold.
The ramp over epochs 40 to 100 was chosen so that the policy first learns
from the cheapest search and receives the stronger expert only once it has
stabilised. We did not run a formal search over these values, and Section 6
returns to this point.

4.7. Solver budget
    Per instance, the base loop uses one anchor solve, one target solve and at
most one expert solve, and refresh rounds add at most two solves for large
instances. The training run made 3,839,970 calls for 1,330,000 instances, or
2.89 calls per instance. Validation added 266,000 calls. By contrast, POMO

                                     13
with an exact reward makes one call per route per instance, so its cost grows
with J. The composition of these calls is reported in Section 5, and the
training loop is otherwise unconstrained by the solver: local search on the
tables took 67.6 hours of summed worker time, against 86.7 hours of Gurobi
time.

5. Computational experiments
5.1. Setup
Baseline.. G200 is POMO trained for 200 epochs with an exact Gurobi re-
ward on every sampled route, on 100,000 random instances per epoch. It uses
the same instance generator and POMO start rule as the proposed method,
a smaller network (6 encoder layers, embedding dimension 128, 8 heads, 1.27
million parameters, against 2.44 million for E133), and a batch size of 512.
The two models therefore differ in architecture as well as in training signal,
and the comparison in this paper is between complete training recipes. It
was not retrained for this paper. An independent retraining under the same
recipe is reported in Section 5.7.

Proposed model.. Training ran for 200 epochs of 10,000 instances each, with
a batch size of 64, Adam with learning rate 10−4 and weight decay 10−6 , and
a cosine schedule. We report the checkpoint at epoch 133 (E133), selected
by mean exact cost on a validation set of 2,000 instances, with 400 at each
of J ∈ {20, 40, 60, 80, 100}. The selection used no test data. One fixed-
order SOCP is solved by Gurobi with one thread per call. Experiments ran
on a workstation with an Intel i9-13900KF processor (32 hardware threads),
31 GB of memory and an NVIDIA RTX 4090 GPU. Over the selection range,
validation means were close: E133 reached 17.703 and E150 reached 17.705.

Data.. Training instances have J drawn uniformly from {20, . . . , 100} and
targets drawn uniformly from the unit square. Speeds and endurance are
sampled uniformly from ranges derived from the literature benchmark, with
vc = 0.2, vd ∈ [0.4, 1] and E ∈ [0.05, 1] in normalised units. We eval-
uate on three sets. Test500 contains 100 new instances at each of J ∈
{20, 40, 60, 80, 100}. The generalisation set contains 100 new instances at
each of J ∈ {150, 200, 300, 500}. Real529 is the literature benchmark of Gam-
bella et al. (2018), Erdoğan and Yıldırım (2021) and Poikonen and Golden
(2020), as compiled by Li et al. (2026), with 72, 132 and 325 instances. Its


                                     14
sizes include J from 10 to 19 and J = 200, which lie outside the training
range. Test500 serves as the primary evidence because it matches the training
distribution and allows a clean comparison at each size. We inspected test500
once for an earlier checkpoint before selecting E133, and the selection itself
relied on the validation set only. Real529 is not a held-out set, because earlier
checkpoints of this project were evaluated on it during method development.
Test protocol.. Both models decode greedily with POMO, with min(J, 100)
starts on Real529 and J starts on the synthetic sets. Under eight-view aug-
mentation, each geometric view is decoded separately. Every distinct route
is solved exactly and the best value is reported. Costs are therefore exact in
all tables.
Metrics.. For model A and baseline B on instance n, the relative gap is
      n − cn )/cn , so positive values mean that A is worse. We report the
100 (cA    B    B

mean over instances with a 95% normal-approximation interval, together
with the counts of wins, ties and losses at a tolerance of 10−3 . We prefer the
per-instance mean because a ratio of mean costs is dominated by the largest
instances. On Real529, for example, the ratio of means is +0.12%, whereas
the per-instance mean is +0.21%.

5.2. Results on synthetic instances
    Table 4 reports test500. Without augmentation, E133 is behind G200 at
every size. The gap is +1.08% at J = 20, rises to about +3.4% at J = 60 and
J = 80, and is +3.05% at J = 100; the mean over all sizes is +2.60%. G200
wins 407 of 500 instances. This is the largest gap in our evaluation, and it
marks the main quality cost of the proposed training scheme. The gap does
not grow steadily with size. It is smallest at J = 20 and stabilises between
J = 60 and J = 100, which is the range in which the progressive search is
active. The mean costs at J = 20 differ by about 0.12 units, and ties are
common there, whereas at J ≥ 60 nearly all instances are lost by a modest
margin. The picture is one of a consistent shortfall and not of occasional bad
instances: E133 wins only 10 to 14 instances out of 100 at the three largest
sizes.
    Eight-view augmentation narrows the gap in the runs available so far.
[E133 with eight views on test500 has not been run. Available result: the
earlier checkpoint E150 has a mean gap of +1.07% to G200 under eight views,
which must not be reported as E133. Replace this paragraph with the E133
result.]

                                       15
Table 4: Test500 without augmentation. Gap is E133 relative to G200 (positive means
E133 is worse).
          J    E133      G200    Gap (%)        95% interval      Win/Tie/Loss
         20   12.594   12.472       +1.08      [+0.72,   +1.43]       15/22/63
         40   15.923   15.633       +2.00      [+1.55,   +2.45]        18/1/81
         60   18.502   17.946       +3.39      [+2.77,   +4.01]        13/0/87
         80   20.079   19.453       +3.47      [+2.98,   +3.96]        10/0/90
        100   22.040   21.440       +3.05      [+2.43,   +3.68]        14/0/86
        All   17.828   17.389       +2.60      [+2.35,   +2.84]      70/23/407


      Table 5: Generalisation set with eight views. Gap is E133 relative to G200.
          J    E133      G200    Gap (%)        95% interval      Win/Tie/Loss
        150   26.003   25.795       +0.96      [+0.62,   +1.31]        23/0/77
        200   29.077   28.868       +0.97      [+0.56,   +1.37]        30/0/70
        300   36.945   36.875       +0.38      [−0.03,   +0.78]        42/0/58
        500   51.283   51.040       +0.56      [+0.23,   +0.90]        40/0/60
        All   35.827   35.644       +0.72      [+0.53,   +0.91]      135/0/265


5.3. Generalisation to larger instances
    Table 5 reports instances with 150 to 500 targets, which are beyond the
training range. Under eight views, the mean gap to G200 is +0.72%, and
at J = 300 the interval includes zero. These gaps are smaller than the
unaugmented test500 gaps, but the two protocols differ, so the tables are
not directly comparable. Without augmentation, the mean gap on the same
instances is +1.63%. Augmentation helps both models, and it helps E133
more: the unaugmented mean is 36.801 for E133 and 36.336 for G200, and
eight views reduce these to 35.827 and 35.644. Neither model was trained at
these sizes. The results thus show that learning from the table did not cost
additional transfer ability relative to learning from exact rewards.

5.4. Results on the literature benchmark
    Table 6 reports Real529 by size band. The mean gap to G200 is +0.21%,
with an interval that excludes zero. Gaps are below +0.53% in every band
up to J = 100. At J = 200, E133 is 1.25% better than G200, with 21 wins
out of 25. Ties are frequent at the smallest sizes, where 127 of 220 instances
end at the same value, which suggests that both models often reach the

                                          16
    Table 6: Real529 with eight views, by size band. Gap is E133 relative to G200.
 Band           n      E133      G200     Gap (%)       95% interval     Win/Tie/Loss
 J = 10–19    220   255.570    255.086        +0.19   [+0.12,   +0.26]      23/127/70
 J = 20–50    234   342.841    341.847        +0.33   [+0.21,   +0.45]      60/33/141
 J = 100       50   431.749    429.727        +0.52   [+0.13,   +0.92]        19/0/31
 J = 200       25   720.934    730.123        −1.25   [−1.78,   −0.72]         21/0/4
 All          529   332.819    332.421        +0.21   [+0.13,   +0.29]    123/160/246


same route. This holds for 25 instances from one source and lies outside the
training range, so we treat it as an observation and not as evidence of general
superiority: on the synthetic instances at the same size in Table 5, E133 is
behind.
    Table 7 compares both models with best known solutions and with three
metaheuristics that search over visiting orders and score every order with
the same exact SOCP. The reference value of each instance is the best upper
bound over all methods in Li et al. (2026). [Merge the best values of Vieira
et al. (2026) into the reference.] Each metaheuristic was given a multiple
of the single-core time that E133 needs on that instance. At three times
that budget, iterated local search (ILS) and variable neighbourhood search
(VNS) are worse than E133. At ten times the budget they are better than
both learned policies, at +0.73% and +0.78% against +1.35% for E133 and
+1.14% for G200. The difference is concentrated in the clustered instances
of Poikonen and Golden (2020), where VNS reaches −0.03% and E133 is
at +2.10%. Clustered instances differ from the uniform distribution used
for training, which is consistent with the weakness of both learned policies
there. Simulated annealing remains behind at all budgets, with the gap at
+3.14% at ten times the budget. On average, E133 and G200 need 0.78 s
of wall time per instance, or 10.3 s on a single core, on the same hardware.
The time grows with size, from under 0.1 s of wall time at J = 10 to about
5.6 s at J = 200, with 16 parallel solver workers. Relative to the best values
in Li et al. (2026), both models also find new best solutions on a number of
Poikonen instances, for which the reference values come from a heuristic. We
have recomputed one such case geometrically and found it feasible under our
model. [Report the counts of new best solutions once the reference includes
Vieira et al. (2026).]



                                         17
Table 7: Real529 gap to the best known value, 100 (UB − BKS)/UB, by subset. Budgets
are multiples of the single-core time of E133.
                           Erd132      Erd132     Poi325       Poi325
 Method         Gam72                                                        All
                           16–20       25–50      Regular      Cluster
 E133             +0.19      +0.24      +0.89        +1.75      +2.10     +1.35
 G200             +0.01      +0.08      +0.80        +1.59      +1.76     +1.14
 ILS, 3×          +0.71      +0.80      +3.08        +2.79      +0.36     +1.63
 VNS, 3×          +0.69      +0.76      +3.14        +2.84      +0.36     +1.65
 ILS, 10×         +0.33      +0.27      +1.90        +1.22      −0.02     +0.73
 VNS, 10×         +0.20      +0.34      +1.89        +1.40      −0.03     +0.78
 SA, 10×          +0.53      +1.27      +5.15        +5.16      +1.84     +3.14


5.5. Comparison with a general-purpose exact solver
    To set the scale of the problem, we also solved the full mixed-integer
SOCP (Model+ of Li et al. (2026)) with Gurobi, with a time limit of 3,600 s
per instance and one thread. [Report status, upper bound, lower bound, gap
and time by J for Real529 and test500. The Real529 run has finished 489 of
529 instances and the test500 run has not started.]

5.6. Training efficiency
    Table 10 compares training budgets. E133 used 15 times fewer training
instances and 78 times fewer Gurobi calls than G200, or 73 times fewer when
in-training validation calls are included. The test-time cost is unchanged
in our measurements because it is dominated by exact solves of the POMO
routes: on Real529, decoding takes 0.12 s per instance on the GPU and
Gurobi takes about 10.2 s of single-core time for either model. CPU time for
G200 is an estimate, obtained by multiplying its call count by a mean solve
time measured at about J = 60. Wall time for G200 comes from a different
machine with 128 reward workers, so the ratio of wall times shows the order
of magnitude only. Solver calls and instances are exact counts and form the
primary comparison.
    Table 8 breaks down the 3.84 million calls of E133. Anchor and target
solves contribute one call per instance each. Expert verification, including
near-miss checks, contributes 0.37 calls per instance. Refresh rounds add
0.52 calls per instance on average across all sizes, and more than one per

                                        18
           Table 8: Exact solver calls of E133, epochs 1 to 133, by purpose.
           Purpose                                    Calls   Per instance
           Anchor solve                          1,330,000             1.00
           Target reward solve                   1,330,000             1.00
           Expert and near-miss verification       493,128             0.37
           Refresh rounds                          686,842             0.52
           Total                                 3,839,970             2.89


Table 9: Training dynamics of the proposed method. Improved share is the fraction of
instances whose expert beats the anchor. Bound excess compares the lowest bound among
policy routes with the exact cost of that route.
  Epoch    Anchor cost    Expert cost    Improved share (%)       Bound excess (%)
      20         19.986         19.779                     12.1                92.4
      40         19.205         19.096                      9.0                70.7
      60         19.146         18.798                     24.4                57.5
      80         18.993         18.640                     33.0                57.9
     100         18.850         18.603                     34.9                44.2
     133         18.522         18.288                     37.9                33.6


instance at the end of the run, when more than half of the batches have
J ≥ 60 and refresh is active. A share of 23.7% of refresh solves produced a
better expert, so the refresh budget has a measurable return but also a large
waste. Near-miss checks were attempted 364,974 times.
    Table 9 shows how the loop behaves during training. In the first epoch,
almost any search improves on an untrained policy. By epoch 40 the plain
2-opt search improves only 9% of the anchors, by 0.38% on average. The
ramp then lifts the share of improved experts above one third, which is the
intended effect of the progressive search. Over the same period, the lowest-
bound policy route has a bound that falls from 92% to 34% above its exact
cost, which is consistent with policy routes moving closer to the anchors
as training proceeds. The bound remains loose at epoch 133, yet training
continues to make progress; Section 6 discusses why.
    [Add the 24-hour training curve: validation cost against wall time for
POMO with exact rewards and for the proposed method on the same hard-
ware. The POMO run is queued; confirm the plan for the proposed-method
curve.]

                                          19
                      Table 10: Training budgets of G200 and E133.
                                                         G200                 E133       Ratio
 Training instances                                  20,000,000           1,330,000         15
 Gurobi calls in training                           299,437,760           3,839,970         78
 Gurobi CPU (core-hours)                about 2,580 (estimated)                86.7   about 30
 Wall time (hours)                    about 215.5 (128 workers)   19.2 (32 workers)   about 11
 Test time per Real529 instance (s)                        0.78                0.78          1



5.7. Ablation study
    The ablations answer three questions about the design. Variants B and
D train from scratch to one million instances (epoch 100) and are compared
with the proposed method at the same point on the validation set of 2,000
instances; the results are then reported on test500 by size.
    • Group A retrains POMO with exact rewards from scratch under the
      G200 recipe, on the same hardware, for 24 hours. Its checkpoints at
      the instance and solver budgets of E133 measure the quality attainable
      at a matching budget.

    • Group B replaces table-based search with exact evaluation of candidate
      moves, under a budget of 100 exact solves per instance, to measure what
      the table saves in training time.

    • Group D removes expert diversity: one greedy route receives the full
      local-search budget of the 30 starts, to separate the effect of search
      diversity from that of search effort.
Group C is the proposed method at the same point in training. [Report
quality, wall time and exact calls for groups A, B, C and D. All are queued.]

6. Managerial insights and discussion
6.1. When the approach is worthwhile
    The method changes when computing effort is spent. A metaheuristic
spends it on every instance; a trained policy spends it once, and each later in-
stance takes seconds. For a fleet operator that replans repeatedly, this favours
a trained policy when the number of decisions is large or the replanning in-
terval is short. The relevant cost for the operator is the training budget,
and our results reduce the solver effort by nearly two orders of magnitude.

                                              20
An operator can therefore retrain for a new vessel speed, drone endurance
or service region within a day on a single workstation, where the baseline
recipe needed more than a week on a 128-worker setup. The method needs
no training labels from an external solver run and no precomputed instance
library, because it generates its own experts during training.
    The cost of this saving depends on the use case. Where an operator has
minutes per instance, the iterated local search results in Table 7 suggest that
a metaheuristic will give better plans, particularly when targets are clustered.
Where the plan must be available in seconds, the learned policy is the better
option at the time budgets we tested: ILS and VNS at three times the single-
core time of E133 are behind it. The two can also be combined, since a route
from the policy can start a metaheuristic. We did not test that combination.

6.2. Sensitivity to operating parameters
    [Sensitivity analysis on speed ratio, drone endurance and target distribu-
tion has not been run. Add results and practical implications here, or remove
this subsection.]

6.3. Why a loose bound suffices
    The bound is far from tight on routes that differ from the anchor: Table 9
shows an excess of 33.6% at epoch 133. In a continuation experiment from
E133, we took the minimum over all tables available for each route, which
used about two additional tables per instance on the affected batches and
tightened the bound by only 0.32%. It left validation quality unchanged. A
much tighter bound is thus not easy to obtain from nearby solutions, and on
the evidence so far it is not the limiting factor. Two properties of the design
explain why a loose bound can work. The bound is exact at the anchor
and accurate nearby, which is where local search operates. The labels that
reach imitation learning are exact, so the looseness affects only the ranking of
candidates and the variance of the reward. The relevant property of the table
is that it is cheap and valid, not that it is tight. Group B of the ablation will
quantify this claim. This observation has a wider implication for the design
of learning methods with expensive objectives: a signal with a guaranteed
direction of error and a verification step can substitute for an accurate one.
It is not an argument against tighter bounds. It says only that the marginal
value of tightness appears low in the region we tested.



                                       21
6.4. Limitations
    Five limitations apply. First, E133 remains behind G200 on synthetic
instances, by up to about 3.5% without augmentation. The evidence points
to route diversity. In diagnostics on a later checkpoint from the same method
(epoch 163), the overlap between POMO routes at J = 100 was 0.62, against
0.53 for G200. Attempts to recover diversity by changing the reward bound
or the imitation weight had little effect. Second, the results rest on one train-
ing seed. [Add two further seeds.] Third, the comparison on time depends
on hardware and implementation, which is why we treat solver calls as the
primary measure. Fourth, the settings of the progressive search were set by
development runs and not by a formal search, so the reported performance
may not be the best attainable for the method. Fifth, the bound depends on
the separable structure of the CVTSP. It extends to a variant only if sortie
constraints remain independent of the order, which should hold for several
sorties of fixed composition but fails if, for instance, target time windows
couple the sorties.

7. Conclusion
    This paper asked whether a DRL policy for the CVTSP needs an exact
solver reward for every route. It does not. One optimal fixed-order solution
provides an upper bound on all visiting orders, computed by summing table
entries, and this bound can supply most of the training signal. With 2.89
exact solves per training instance, the resulting policy uses 15 times fewer
instances and 78 times fewer solver calls than POMO trained on exact re-
wards, at the same test time. Its quality is close to that baseline on the
literature benchmark and on large instances, with mean gaps of +0.21% and
+0.72%, and it remains behind on synthetic instances within the training
range, with a mean gap of +2.60% without augmentation. Metaheuristics
given ten times the computing time of the policy on each instance obtain
better solutions on the literature benchmark, so the method suits settings
where plans are needed quickly and where retraining has to be affordable.
    The idea carries over to other routing problems in which a route cost
comes from a continuous subproblem with a separable structure, because
all it requires is that a solution for one route remain feasible for others.
Future work should target the loss of route diversity, which appears to be
the main source of the remaining gap, and should test whether the table
can provide fast warm starts for metaheuristics. Extending the approach to

                                       22
several carriers, as in the vessel routing model of Ren et al. (2027), would
test how far the separable structure reaches.

CRediT authorship contribution statement
   [To be completed by the authors.]

Declaration of competing interest
   [To be completed by the authors.]

Funding
   [To be completed by the authors.]

Data availability
   [State where the code, generated instances and per-instance results are
available. Real529 instances are from the cited sources.]

Ethics statement
   This study used computational experiments on synthetic and published
benchmark instances and involved no human participants or personal data.

Declaration of generative AI use
    [The authors should confirm this statement.] During the preparation of
this work the authors used Claude (Anthropic) to assist with literature search
and manuscript drafting. The authors reviewed and edited the content and
take full responsibility for the publication.




                                     23
References
Agatz, N., Bouman, P., Schmidt, M., 2018. Optimization approaches for the
  traveling salesman problem with drone. Transportation Science 52, 965–
  981. URL: http://dx.doi.org/10.1287/trsc.2017.0791, doi:10.1287/
  trsc.2017.0791.

Bengio, Y., Lodi, A., Prouvost, A., 2021. Machine learning for combinato-
  rial optimization: A methodological tour d’horizon. European Journal of
  Operational Research 290, 405–421. URL: http://dx.doi.org/10.1016/
  j.ejor.2020.07.063, doi:10.1016/j.ejor.2020.07.063.

Chung, S.H., Sah, B., Lee, J., 2020. Optimization for drone and drone-truck
  combined operations: A review of the state of the art and future directions.
  Computers & Operations Research 123, 105004. URL: http://dx.doi.
  org/10.1016/j.cor.2020.105004, doi:10.1016/j.cor.2020.105004.

Erdoğan, G., Yıldırım, E.A., 2021. Exact and heuristic algorithms for the
  carrier–vehicle traveling salesman problem. Transportation Science 55,
  101–121. URL: http://dx.doi.org/10.1287/trsc.2020.0999, doi:10.
  1287/trsc.2020.0999.

Fahradyan, T., Bono Rossello, N., Garone, E., 2020. Multiple carrier-
  vehicle travelling salesman problem, in: Modelling and Simulation for
  Autonomous Systems. Springer. Lecture Notes in Computer Science, pp.
  180–189. doi:10.1007/978-3-030-43890-6_14.

Fan, M., Cheng, J., Wu, Y., Zhang, Y., Yang, Y., Wu, G., Sartoretti, G.,
  2025. A Unified Deep Reinforcement Learning Approach for Close Enough
  Traveling Salesman Problem. arXiv:2510.03065.

Ferber, A., Huang, T., Zha, D., Schubert, M., Steiner, B., Dilkina, B., Tian,
  Y., 2022. SurCo: Learning Linear Surrogates For Combinatorial Nonlinear
  Optimization Problems. arXiv:2210.12547.

Gambella, C., Lodi, A., Vigo, D., 2018. Exact solutions for the carrier–
 vehicle traveling salesman problem. Transportation Science 52, 320–
 330. URL: http://dx.doi.org/10.1287/trsc.2017.0771, doi:10.1287/
 trsc.2017.0771.



                                     24
Garone, E., Determe, J.F., Naldi, R., 2014. Generalized traveling sales-
 man problem for carrier-vehicle systems. Journal of Guidance, Control,
 and Dynamics 37, 766–774. URL: http://dx.doi.org/10.2514/1.62126,
 doi:10.2514/1.62126.

Garone, E., Naldi, R., Casavola, A., 2011. Traveling salesman problem
 for a class of carrier-vehicle systems. Journal of Guidance, Control, and
 Dynamics 34, 1272–1276. URL: http://dx.doi.org/10.2514/1.50539,
 doi:10.2514/1.50539.

Helsgaun, K., 2000. An effective implementation of the lin–kernighan trav-
  eling salesman heuristic. European Journal of Operational Research 126,
  106–130. URL: http://dx.doi.org/10.1016/s0377-2217(99)00284-2,
  doi:10.1016/s0377-2217(99)00284-2.

Hottung, A., Kwon, Y.D., Tierney, K., 2021. Efficient Active Search for
  Combinatorial Optimization Problems. arXiv:2106.05126.

Kim, H., Kim, M., Ahn, S., Park, J., 2023. Symmetric Replay Training:
  Enhancing Sample Efficiency in Deep Reinforcement Learning for Combi-
  natorial Optimization. arXiv:2306.01276.

Kim, M., Park, J., Park, J., 2022. Sym-NCO: Leveraging Symmetricity for
  Neural Combinatorial Optimization. arXiv:2205.13209.

Kool, W., van Hoof, H., Welling, M., 2018. Attention, Learn to Solve Routing
 Problems! arXiv:1803.08475.

Kwon, Y.D., Choo, J., Kim, B., Yoon, I., Gwon, Y., Min, S., 2020. POMO:
 Policy Optimization with Multiple Optima for Reinforcement Learning.
 arXiv:2010.16011.

Lee, J., Park, J., Lee, D., Bang, H., 2023. Sequential cooperative rein-
  forcement learning of mothership routing problem, in: AIAA SCITECH
  2023 Forum, American Institute of Aeronautics and Astronautics. URL:
  http://dx.doi.org/10.2514/6.2023-2658, doi:10.2514/6.2023-2658.

Li, Y., Wang, S., Zhou, S., Wang, Z., 2024. A mathematical formulation and
  a tabu search heuristic for the joint vessel-uav routing problem. Computers
  & Operations Research 169, 106723. URL: http://dx.doi.org/10.1016/
  j.cor.2024.106723, doi:10.1016/j.cor.2024.106723.

                                     25
Li, Y., Zhou, S., Côté, J.F., 2026. Exact methods and a two-stage iterative
  heuristic for the carrier-vehicle traveling salesman problem. INFORMS
  Journal on Computing 38, 1611–1626. URL: http://dx.doi.org/10.
  1287/ijoc.2025.1140, doi:10.1287/ijoc.2025.1140.

Lin, S., Kernighan, B.W., 1973. An effective heuristic algorithm for the
  traveling-salesman problem. Operations Research 21, 498–516. URL:
  http://dx.doi.org/10.1287/opre.21.2.498, doi:10.1287/opre.21.2.
  498.

Liu, S., Yu, Y., Tian, Q., Sun, H., 2026. Routing optimization for an evtol-
  and-drone delivery system in continuous space with no-fly zones: A rein-
  forcement learning approach. Transportation Research Part E: Logistics
  and Transportation Review 209, 104741. URL: http://dx.doi.org/10.
  1016/j.tre.2026.104741, doi:10.1016/j.tre.2026.104741.

Lobo, M.S., Vandenberghe, L., Boyd, S., Lebret, H., 1998. Applications of
  second-order cone programming. Linear Algebra and its Applications 284,
  193–228. URL: http://dx.doi.org/10.1016/s0024-3795(98)10032-0,
  doi:10.1016/s0024-3795(98)10032-0.

Luo, F., Lin, X., Liu, F., Zhang, Q., Wang, Z., 2023. Neural combina-
  torial optimization with heavy decoder: Toward large scale generaliza-
  tion, in: Advances in Neural Information Processing Systems 36, Neural
  Information Processing Systems Foundation, Inc. (NeurIPS). pp. 8845–
  8864. URL: http://dx.doi.org/10.52202/075280-0387, doi:10.52202/
  075280-0387.

Luo, F., Lin, X., Wang, Z., Tong, X., Yuan, M., Zhang, Q., 2024.
  Self-Improved Learning for Scalable Neural Combinatorial Optimization.
  arXiv:2403.19561.

Luttmann, L., Xie, L., 2025. Multi-Action Self-Improvement for Neural Com-
  binatorial Optimization. arXiv:2510.12273.

Macrina, G., Di Puglia Pugliese, L., Guerriero, F., Laporte, G., 2020. Drone-
 aided routing: A literature review. Transportation Research Part C:
 Emerging Technologies 120, 102762. URL: http://dx.doi.org/10.1016/
 j.trc.2020.102762, doi:10.1016/j.trc.2020.102762.


                                     26
Mazyavkina, N., Sviridov, S., Ivanov, S., Burnaev, E., 2021. Reinforcement
 learning for combinatorial optimization: A survey. Computers & Opera-
 tions Research 134, 105400. URL: http://dx.doi.org/10.1016/j.cor.
 2021.105400, doi:10.1016/j.cor.2021.105400.
Murray, C.C., Chu, A.G., 2015. The flying sidekick traveling salesman prob-
 lem: Optimization of drone-assisted parcel delivery. Transportation Re-
 search Part C: Emerging Technologies 54, 86–109. URL: http://dx.doi.
 org/10.1016/j.trc.2015.03.005, doi:10.1016/j.trc.2015.03.005.
Nazari, M., Oroojlooy, A., Snyder, L.V., Takáč, M., 2018. Reinforcement
  Learning for Solving the Vehicle Routing Problem. arXiv:1802.04240.
Otto, A., Agatz, N., Campbell, J., Golden, B., Pesch, E., 2018. Optimization
  approaches for civil applications of unmanned aerial vehicles (uavs) or
  aerial drones: A survey. Networks 72, 411–458. URL: http://dx.doi.
  org/10.1002/net.21818, doi:10.1002/net.21818.
Pirnay, J., Grimm, D.G., 2024. Self-Improvement for Neural Combina-
  torial Optimization: Sample without Replacement, but Improvement.
  arXiv:2403.15180.
Poikonen, S., Golden, B., 2020. The mothership and drone routing problem.
  INFORMS Journal on Computing 32, 249–262. URL: http://dx.doi.
  org/10.1287/ijoc.2018.0879, doi:10.1287/ijoc.2018.0879.
Ren, B., Li, Y., Sun, H., Bai, D., Chen, K., 2027. A tractable model
  and heuristic for vessel routing with multiple unmanned aerial vehicles.
  Transportation Research Part E: Logistics and Transportation Review
  217, 105178. URL: http://dx.doi.org/10.1016/j.tre.2026.105178,
  doi:10.1016/j.tre.2026.105178.
Sun, Y., Fügenschuh, A., Vaze, V., 2026. Physics-aware truck and drone
  delivery planning using optimization & machine learning. Computers &
  Operations Research 196, 107665. URL: http://dx.doi.org/10.1016/
  j.cor.2026.107665, doi:10.1016/j.cor.2026.107665.
Vieira, B.S., Frazzon, E.M., Coelho, L.C., 2026. An exact linearization-based
  refinement algorithm to the carrier-vehicle traveling salesman problem.
  Transportation Science 60, 589–607. URL: http://dx.doi.org/10.1287/
  trsc.2025.0292, doi:10.1287/trsc.2025.0292.

                                     27
Vinyals, O., Fortunato, M., Jaitly, N., 2015.             Pointer Networks.
  arXiv:1506.03134.

Williams, R.J., 1992. Simple statistical gradient-following algorithms for
 connectionist reinforcement learning. Machine Learning 8, 229–256.
 doi:10.1007/BF00992696.

Wu, Y., Song, W., Cao, Z., Zhang, J., Lim, A., 2022. Learning improvement
 heuristics for solving routing problems. IEEE Transactions on Neural Net-
 works and Learning Systems 33, 5057–5069. URL: http://dx.doi.org/
 10.1109/tnnls.2021.3068828, doi:10.1109/tnnls.2021.3068828.

Wu, Y., Yu, Y., Wu, L., Feng, T., Zhang, L., Wang, Z., Gao, J., 2026.
 Deep reinforcement learning approach to solving clustered vehicle routing
 problems. Transportation Research Part E: Logistics and Transportation
 Review 209, 104742. URL: http://dx.doi.org/10.1016/j.tre.2026.
 104742, doi:10.1016/j.tre.2026.104742.

Yan, K., He, S., Song, R., Zhao, R., Cai, J., Yuan, S., 2026. Mothership-based
  truck-multi-drone cooperative delivery to moving targets: An event-driven
  multi-agent reinforcement learning approach. URL: http://dx.doi.org/
  10.2139/ssrn.7371806, doi:10.2139/ssrn.7371806.

Zhao, L., Bi, X., Li, G., Dong, Z., Xiao, N., Zhao, A., 2022. Robust traveling
  salesman problem with multiple drones: Parcel delivery under uncertain
  navigation environments. Transportation Research Part E: Logistics and
  Transportation Review 168, 102967. URL: http://dx.doi.org/10.1016/
  j.tre.2022.102967, doi:10.1016/j.tre.2022.102967.




                                     28
