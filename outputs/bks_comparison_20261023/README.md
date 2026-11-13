# Real529 vs paper BKS / TSIH (2026-10-23)

Source: `TSIH results/TSIH-results.xlsx` (copy of /home/marmot/wangjianing/CVTSP/TSIH results/),
row Instance N = `instance/Data/Example_N.txt`; J matches on all 529 rows. Parsed to `tsih_by_example.json`.
Model numbers: `outputs/progressive_recipe_20261020/final_external_e133/real529/results.json`
(greedy, POMO=min(J,100), 8x aug, exact min).

Mean UB* (best known) = 327.396, TSIH UB = 328.051, TSIH mean time 20.2 s/instance.
Gap of means to UB*: G200 +1.53%, E150 +1.60%, E133 +1.66%, S500-E100 +1.85%.

Anomaly: G200 < UB* on 117 instances; 83 by 0.001-0.005 (BKS rounded to 2 decimals), 34 by >0.005
(Regular 19, Clustered 13, MD 1, VLD 1; J20-50 mostly). Example_260 (Regular, J20, BKS 269.80):
G200 route re-solved with CVTSP_SOCP (carrier speed 1 = Vv, UAV speed 2 = Vd, endurance MT=20) gives 266.512,
and an independent geometric recomputation from the returned take-off/landing points gives 266.512, all sorties
<= 20. So the solution is feasible under our model; the paper's BKS is not optimal there (or its model differs).
