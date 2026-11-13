# Equal-budget POMO + exact Gurobi reward (3,840,896 solves, 264,512 instances) vs E133 and G200

Real529 (8x aug, greedy POMO=min(J,100), exact min):
| | Gam72 Δ | Erd 16-20 Δ | Erd 25-50 Δ | Poi Regular Δ | Poi Cluster Δ | Real529 Δ | mean UB |
|---|---|---|---|---|---|---|---|
| G200 (299M solves) | +0.01 | +0.08 | +0.80 | +1.47 | +1.45 | +1.02 | 332.42 |
| E133 (3.84M solves) | +0.19 | +0.24 | +0.89 | +1.64 | +1.80 | +1.23 | 332.82 |
| POMO+Gurobi equal solves | +0.61 | +1.54 | +6.28 | +6.30 | +4.52 | +4.48 | 351.69 |
POMO equal vs E133: +5.67% of means, 95% CI [+4.87, +6.54]; E133 better/tie/worse 354/110/65.
By J (mean UB, G200/E133/POMO-eq): J10 224.95/225.47/225.30; J20 282.38/283.28/285.25; J30 307.33/308.30/314.76;
J50 358.98/360.74/379.49; J100 429.73/431.75/481.69; J200 730.12/720.93/893.63.

validation2000 (full POMO=J, no aug): E133 17.7030; POMO-eq 18.4883 (+4.44%, E133 better on 1406/2000);
G200 17.3091 (-2.23% vs E133, G200 better on 1648/2000).
By J (E133/POMO-eq/G200): J20 12.456/12.557/12.321; J40 15.424/15.695/15.094; J60 18.336/18.954/17.909;
J80 20.311/21.517/19.806; J100 21.988/23.718/21.415.
