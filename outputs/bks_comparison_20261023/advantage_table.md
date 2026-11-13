# Training budget vs quality: G200 (Gurobi-reward POMO) vs E133 (self-improving, progressive recipe)

| | G200 | E133 (ours) | Ratio |
|---|---:|---:|---:|
| Training epochs | 200 | 133 | |
| Training instances | 20,000,000 (100k/epoch) | 1,330,000 (10k/epoch) | 15x fewer |
| Gurobi SOCP calls in training | 299,437,760 (metrics.csv, every POMO route) | 3,839,970 (+266,000 validation) | 78x fewer (73x incl. validation) |
| Gurobi CPU time in training | ~2,580 core-h (estimate: calls x 0.031 s/call measured here at J~60) | 86.7 core-h (measured) | ~30x less |
| Training wall time | ~215.5 h (~9 days): 64.7 min/epoch x 200, measured over epochs 184-188 with 128 reward workers (other machine; user's run.log record) | 19.2 h incl. in-training validation (32 workers, i9-13900KF + RTX 4090) | ~11x shorter with 4x fewer workers |
| Worker-hours (wall x workers) | ~27,600 | ~614 | ~45x fewer |
| Real529 mean makespan (8x aug) | 332.42 | 332.82 (+0.12%, 95% CI [+0.004, +0.229]) | |
| Real529 Δ to BKS | +1.02% | +1.23% | |
| Real529 J200 | 730.12 | 720.93 (-1.26%) | |
| Real529 test time / instance | 0.78 s wall, 10.3 s single-core | 0.78 s wall, 10.3 s single-core | same |
| test500 (no aug) | ref | +2.52% | |
