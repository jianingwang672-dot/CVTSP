# import numpy as np
# from gurobipy import Model, GRB, quicksum

# def gurobi_cvp_socp(
#     depot,
#     targets,
#     tour,          
#     Vc,
#     Vv,
#     endurance_a,
#     threads=None,
#     time_limit=None,
#     test=False
# ):
#     depot = np.asarray(depot, dtype=float).reshape(2,)
#     targets = np.asarray(targets, dtype=float)

#     n = len(tour)
#     assert targets.shape[0] == n
#     assert all(0 <= j < n for j in tour)

#     m = Model("CVP-fixed-tour-0based")
#     m.setParam("OutputFlag", 0)

#     if threads is not None:
#         m.setParam("Threads", int(threads))
#     if time_limit is not None:
#         m.setParam("TimeLimit", float(time_limit))

#     # ---------- 变量 ----------
#     sx = [m.addVar(lb=-GRB.INFINITY) for _ in range(n)]
#     sy = [m.addVar(lb=-GRB.INFINITY) for _ in range(n)]
#     lx = [m.addVar(lb=-GRB.INFINITY) for _ in range(n)]
#     ly = [m.addVar(lb=-GRB.INFINITY) for _ in range(n)]

#     t1  = [m.addVar(lb=0.0) for _ in range(n)]
#     t2  = [m.addVar(lb=0.0) for _ in range(n)]
#     tau = [m.addVar(lb=0.0) for _ in range(n)]

#     Tseg = [m.addVar(lb=0.0) for _ in range(n+1)]  # n+1 段

#     m.update()

#     # ---------- 目标 ----------
#     m.setObjective(
#         quicksum(tau[k] for k in range(n)) +
#         quicksum(Tseg[k] for k in range(n+1)),
#         GRB.MINIMIZE
#     )

#     # ---------- 约束 ----------
#     for k in range(n):

#         j = tour[k]              # 0-based
#         px, py = targets[j]

#         # tau = t1 + t2
#         m.addConstr(tau[k] == t1[k] + t2[k])

#         # endurance
#         m.addConstr(tau[k] <= endurance_a)

#         # ---- UAV outbound ----
#         dx = m.addVar(lb=-GRB.INFINITY)
#         dy = m.addVar(lb=-GRB.INFINITY)
#         d  = m.addVar(lb=0.0)

#         m.addConstr(dx == sx[k] - px)
#         m.addConstr(dy == sy[k] - py)
#         m.addGenConstrNorm(d, [dx, dy], 2)
#         m.addConstr(d <= Vv * t1[k])

#         # ---- UAV inbound ----
#         dx = m.addVar(lb=-GRB.INFINITY)
#         dy = m.addVar(lb=-GRB.INFINITY)
#         d  = m.addVar(lb=0.0)

#         m.addConstr(dx == lx[k] - px)
#         m.addConstr(dy == ly[k] - py)
#         m.addGenConstrNorm(d, [dx, dy], 2)
#         m.addConstr(d <= Vv * t2[k])

#         # ---- sync ----
#         dx = m.addVar(lb=-GRB.INFINITY)
#         dy = m.addVar(lb=-GRB.INFINITY)
#         d  = m.addVar(lb=0.0)

#         m.addConstr(dx == lx[k] - sx[k])
#         m.addConstr(dy == ly[k] - sy[k])
#         m.addGenConstrNorm(d, [dx, dy], 2)
#         m.addConstr(d <= Vc * tau[k])

#         # ---- carrier segment ----
#         dx = m.addVar(lb=-GRB.INFINITY)
#         dy = m.addVar(lb=-GRB.INFINITY)
#         d  = m.addVar(lb=0.0)

#         if k == 0:
#             m.addConstr(dx == sx[k] - depot[0])
#             m.addConstr(dy == sy[k] - depot[1])
#         else:
#             m.addConstr(dx == sx[k] - lx[k-1])
#             m.addConstr(dy == sy[k] - ly[k-1])

#         m.addGenConstrNorm(d, [dx, dy], 2)
#         m.addConstr(d <= Vc * Tseg[k])

#     # ---- final العودة depot ----
#     dx = m.addVar(lb=-GRB.INFINITY)
#     dy = m.addVar(lb=-GRB.INFINITY)
#     d  = m.addVar(lb=0.0)

#     m.addConstr(dx == depot[0] - lx[n-1])
#     m.addConstr(dy == depot[1] - ly[n-1])
#     m.addGenConstrNorm(d, [dx, dy], 2)
#     m.addConstr(d <= Vc * Tseg[n])

#     m.optimize()

#     if not test:
#         return m.objVal

#     return {
#         "obj": m.objVal,
#         "sx": np.array([[sx[k].X, sy[k].X] for k in range(n)]),
#         "lx": np.array([[lx[k].X, ly[k].X] for k in range(n)]),
#         "t1": np.array([t1[k].X for k in range(n)]),
#         "t2": np.array([t2[k].X for k in range(n)]),
#         "tau": np.array([tau[k].X for k in range(n)]),
#         "Tseg": np.array([Tseg[k].X for k in range(n+1)]),
#     }


# def check_cvp_solution(depot, targets, tour, Vc, Vv, a, sol, tol=1e-6):
   
#     depot = np.asarray(depot, dtype=float).reshape(2,)
#     targets = np.asarray(targets, dtype=float)
#     n = len(tour)

#     assert targets.shape[0] == n, "targets 行数应等于 tour 长度"
#     assert all(0 <= j < n for j in tour), "tour 必须是 0-based（元素在 0..n-1）"

#     sx = np.asarray(sol["sx"], dtype=float)   # (n,2)
#     lx = np.asarray(sol["lx"], dtype=float)   # (n,2)
#     t1 = np.asarray(sol["t1"], dtype=float)   # (n,)
#     t2 = np.asarray(sol["t2"], dtype=float)   # (n,)
#     tau = np.asarray(sol["tau"], dtype=float) # (n,)
#     Tseg = np.asarray(sol["Tseg"], dtype=float) # (n+1,)

#     assert sx.shape == (n, 2) and lx.shape == (n, 2), "sx/lx 形状应为 (n,2)"
#     assert t1.shape == (n,) and t2.shape == (n,) and tau.shape == (n,), "t1/t2/tau 形状应为 (n,)"
#     assert Tseg.shape == (n+1,), "Tseg 形状应为 (n+1,)"

#     violations = []

#     for k in range(n):
#         j = tour[k]      # 0..n-1
#         pj = targets[j]  # 目标点坐标

#         # (1) outbound: ||s_k - p_j|| <= Vv * t1_k
#         lhs = np.linalg.norm(sx[k] - pj)
#         rhs = Vv * t1[k]
#         if lhs > rhs + tol:
#             violations.append(("out", k, lhs, rhs))

#         # (2) inbound: ||l_k - p_j|| <= Vv * t2_k
#         lhs = np.linalg.norm(lx[k] - pj)
#         rhs = Vv * t2[k]
#         if lhs > rhs + tol:
#             violations.append(("in", k, lhs, rhs))

#         # (3) sync: ||l_k - s_k|| <= Vc * tau_k
#         lhs = np.linalg.norm(lx[k] - sx[k])
#         rhs = Vc * tau[k]
#         if lhs > rhs + tol:
#             violations.append(("sync", k, lhs, rhs))

#         # (4) endurance: tau_k <= a
#         lhs = tau[k]
#         rhs = a
#         if lhs > rhs + tol:
#             violations.append(("endurance", k, lhs, rhs))

#         # (5) tau definition: tau_k == t1_k + t2_k
#         lhs = abs(tau[k] - (t1[k] + t2[k]))
#         if lhs > 1e-5:
#             violations.append(("tau_def", k, lhs, 0.0))

#         # (6) carrier segment: ||s_k - prev|| <= Vc * Tseg[k]
#         #     prev = depot if k==0 else l_{k-1}
#         prev = depot if k == 0 else lx[k-1]
#         lhs = np.linalg.norm(sx[k] - prev)
#         rhs = Vc * Tseg[k]
#         if lhs > rhs + tol:
#             violations.append(("car_seg", k, lhs, rhs))

#     # (7) back to depot: ||depot - l_{n-1}|| <= Vc * Tseg[n]
#     lhs = np.linalg.norm(depot - lx[n-1])
#     rhs = Vc * Tseg[n]
#     if lhs > rhs + tol:
#         violations.append(("back", n, lhs, rhs))

#     ok = (len(violations) == 0)
#     return ok, violations
import numpy as np
from gurobipy import Model, GRB, quicksum


def gurobi_cvp_socp(
    depot,
    targets,
    tour,          # visit-order, 0-based indices into targets, length = n
    Vc,
    Vv,
    endurance_a,
    threads=None,
    time_limit=None,
    test=False,
    output_flag=0,
):
    """
    Fixed-tour CVP (SOCP/QCP via Gurobi general norm constraints).

    Variables (k = 0..n-1):
      s_k = (sx[k], sy[k])   take-off point
      l_k = (lx[k], ly[k])   landing point
      t1[k], t2[k], tau[k]   UAV outbound/inbound and total flight time
      Tseg[k]               carrier travel time from prev landing to current take-off
      Tseg[n]               last segment from last landing back to depot

    Constraints:
      tau[k] = t1[k] + t2[k]
      tau[k] <= a
      ||s_k - p_j|| <= Vv * t1[k]
      ||l_k - p_j|| <= Vv * t2[k]
      ||l_k - s_k|| <= Vc * tau[k]
      ||s_k - prev|| <= Vc * Tseg[k], prev = depot if k=0 else l_{k-1}
      ||depot - l_{n-1}|| <= Vc * Tseg[n]

    Objective:
      min sum_k tau[k] + sum_{k=0..n} Tseg[k]
    """

    depot = np.asarray(depot, dtype=float).reshape(2,)
    targets = np.asarray(targets, dtype=float)

    n = len(tour)
    N = targets.shape[0]

    # ✅ 修复：targets 不要求等于 n；只要求 tour 索引合法
    assert N >= 1, "targets must be non-empty"
    assert n >= 1, "tour must be non-empty"
    assert all(isinstance(j, (int, np.integer)) for j in tour), "tour indices must be integers"
    assert all(0 <= j < N for j in tour), f"tour must be 0-based indices within 0..{N-1}"
    # 可选：如果你期望每个目标访问一次（CVTSP 常见），加这个检查
    # assert len(set(tour)) == n, "tour contains duplicates"

    m = Model("CVP-fixed-tour-0based")
    m.setParam("OutputFlag", int(output_flag))
    if threads is not None:
        m.setParam("Threads", int(threads))
    if time_limit is not None:
        m.setParam("TimeLimit", float(time_limit))

    # ---------- variables ----------
    sx = [m.addVar(lb=-GRB.INFINITY, name=f"sx[{k}]") for k in range(n)]
    sy = [m.addVar(lb=-GRB.INFINITY, name=f"sy[{k}]") for k in range(n)]
    lx = [m.addVar(lb=-GRB.INFINITY, name=f"lx[{k}]") for k in range(n)]
    ly = [m.addVar(lb=-GRB.INFINITY, name=f"ly[{k}]") for k in range(n)]

    t1  = [m.addVar(lb=0.0, name=f"t1[{k}]") for k in range(n)]
    t2  = [m.addVar(lb=0.0, name=f"t2[{k}]") for k in range(n)]
    tau = [m.addVar(lb=0.0, name=f"tau[{k}]") for k in range(n)]

    # carrier segments: 0..n-1 between visits, n is the final return to depot
    Tseg = [m.addVar(lb=0.0, name=f"Tseg[{k}]") for k in range(n+1)]

    # store distances (for debugging / returning)
    d_out  = [m.addVar(lb=0.0, name=f"d_out[{k}]")  for k in range(n)]
    d_in   = [m.addVar(lb=0.0, name=f"d_in[{k}]")   for k in range(n)]
    d_sync = [m.addVar(lb=0.0, name=f"d_sync[{k}]") for k in range(n)]
    d_car  = [m.addVar(lb=0.0, name=f"d_car[{k}]")  for k in range(n)]
    d_back = m.addVar(lb=0.0, name="d_back")

    m.update()

    # ---------- objective ----------
    m.setObjective(
        quicksum(tau[k] for k in range(n)) +
        quicksum(Tseg[k] for k in range(n+1)),
        GRB.MINIMIZE
    )

    # ---------- constraints ----------
    for k in range(n):
        j = int(tour[k])
        px = float(targets[j, 0])
        py = float(targets[j, 1])

        # tau = t1 + t2
        m.addConstr(tau[k] == t1[k] + t2[k], name=f"tau_def[{k}]")

        # endurance
        m.addConstr(tau[k] <= float(endurance_a), name=f"endurance[{k}]")

        # ---- UAV outbound: ||s_k - p|| <= Vv * t1 ----
        out_x = m.addVar(lb=-GRB.INFINITY, name=f"out_x[{k}]")
        out_y = m.addVar(lb=-GRB.INFINITY, name=f"out_y[{k}]")
        m.addConstr(out_x == sx[k] - px, name=f"out_x_def[{k}]")
        m.addConstr(out_y == sy[k] - py, name=f"out_y_def[{k}]")
        m.addGenConstrNorm(d_out[k], [out_x, out_y], 2, name=f"norm_out[{k}]")
        m.addConstr(d_out[k] <= float(Vv) * t1[k], name=f"out[{k}]")

        # ---- UAV inbound: ||l_k - p|| <= Vv * t2 ----
        in_x = m.addVar(lb=-GRB.INFINITY, name=f"in_x[{k}]")
        in_y = m.addVar(lb=-GRB.INFINITY, name=f"in_y[{k}]")
        m.addConstr(in_x == lx[k] - px, name=f"in_x_def[{k}]")
        m.addConstr(in_y == ly[k] - py, name=f"in_y_def[{k}]")
        m.addGenConstrNorm(d_in[k], [in_x, in_y], 2, name=f"norm_in[{k}]")
        m.addConstr(d_in[k] <= float(Vv) * t2[k], name=f"in[{k}]")

        # ---- sync: ||l_k - s_k|| <= Vc * tau ----
        sync_x = m.addVar(lb=-GRB.INFINITY, name=f"sync_x[{k}]")
        sync_y = m.addVar(lb=-GRB.INFINITY, name=f"sync_y[{k}]")
        m.addConstr(sync_x == lx[k] - sx[k], name=f"sync_x_def[{k}]")
        m.addConstr(sync_y == ly[k] - sy[k], name=f"sync_y_def[{k}]")
        m.addGenConstrNorm(d_sync[k], [sync_x, sync_y], 2, name=f"norm_sync[{k}]")
        m.addConstr(d_sync[k] <= float(Vc) * tau[k], name=f"sync[{k}]")

        # ---- carrier segment: ||s_k - prev|| <= Vc * Tseg[k] ----
        car_x = m.addVar(lb=-GRB.INFINITY, name=f"car_x[{k}]")
        car_y = m.addVar(lb=-GRB.INFINITY, name=f"car_y[{k}]")

        if k == 0:
            m.addConstr(car_x == sx[k] - float(depot[0]), name=f"car_x0_def")
            m.addConstr(car_y == sy[k] - float(depot[1]), name=f"car_y0_def")
        else:
            m.addConstr(car_x == sx[k] - lx[k-1], name=f"car_x_def[{k}]")
            m.addConstr(car_y == sy[k] - ly[k-1], name=f"car_y_def[{k}]")

        m.addGenConstrNorm(d_car[k], [car_x, car_y], 2, name=f"norm_car[{k}]")
        m.addConstr(d_car[k] <= float(Vc) * Tseg[k], name=f"car_seg[{k}]")

    # ---- final return: ||depot - l_{n-1}|| <= Vc * Tseg[n] ----
    back_x = m.addVar(lb=-GRB.INFINITY, name="back_x")
    back_y = m.addVar(lb=-GRB.INFINITY, name="back_y")
    m.addConstr(back_x == float(depot[0]) - lx[n-1], name="back_x_def")
    m.addConstr(back_y == float(depot[1]) - ly[n-1], name="back_y_def")
    m.addGenConstrNorm(d_back, [back_x, back_y], 2, name="norm_back")
    m.addConstr(d_back <= float(Vc) * Tseg[n], name="back_to_depot")

    # solve
    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"CVP solve failed: status={m.Status}")

    if not test:
        return float(m.objVal)

    return {
        "obj": float(m.objVal),
        "status": int(m.Status),

        "sx": np.array([[sx[k].X, sy[k].X] for k in range(n)], dtype=float),
        "lx": np.array([[lx[k].X, ly[k].X] for k in range(n)], dtype=float),

        "t1": np.array([t1[k].X for k in range(n)], dtype=float),
        "t2": np.array([t2[k].X for k in range(n)], dtype=float),
        "tau": np.array([tau[k].X for k in range(n)], dtype=float),
        "Tseg": np.array([Tseg[k].X for k in range(n+1)], dtype=float),

        # optional debug distances
        "d_out": np.array([d_out[k].X for k in range(n)], dtype=float),
        "d_in": np.array([d_in[k].X for k in range(n)], dtype=float),
        "d_sync": np.array([d_sync[k].X for k in range(n)], dtype=float),
        "d_car": np.array([d_car[k].X for k in range(n)], dtype=float),
        "d_back": float(d_back.X),
    }


def check_cvp_solution(depot, targets, tour, Vc, Vv, a, sol, tol=1e-6):
    """
    Validate constraints numerically.

    Supports:
      targets: (N,2) full list
      tour: 0-based indices into targets, length n
      sol arrays are visit-order length n (sx/lx/t1/t2/tau) and Tseg length n+1.
    """
    depot = np.asarray(depot, dtype=float).reshape(2,)
    targets = np.asarray(targets, dtype=float)

    n = len(tour)
    N = targets.shape[0]

    assert all(0 <= int(j) < N for j in tour), f"tour must be 0-based indices within 0..{N-1}"

    sx = np.asarray(sol["sx"], dtype=float)
    lx = np.asarray(sol["lx"], dtype=float)
    t1 = np.asarray(sol["t1"], dtype=float)
    t2 = np.asarray(sol["t2"], dtype=float)
    tau = np.asarray(sol["tau"], dtype=float)
    Tseg = np.asarray(sol["Tseg"], dtype=float)

    assert sx.shape == (n, 2) and lx.shape == (n, 2), "sx/lx must be (n,2)"
    assert t1.shape == (n,) and t2.shape == (n,) and tau.shape == (n,), "t1/t2/tau must be (n,)"
    assert Tseg.shape == (n+1,), "Tseg must be (n+1,)"

    violations = []

    for k in range(n):
        j = int(tour[k])
        pj = targets[j]

        # outbound
        lhs = np.linalg.norm(sx[k] - pj)
        rhs = Vv * t1[k]
        if lhs > rhs + tol:
            violations.append(("out", k, lhs, rhs))

        # inbound
        lhs = np.linalg.norm(lx[k] - pj)
        rhs = Vv * t2[k]
        if lhs > rhs + tol:
            violations.append(("in", k, lhs, rhs))

        # sync
        lhs = np.linalg.norm(lx[k] - sx[k])
        rhs = Vc * tau[k]
        if lhs > rhs + tol:
            violations.append(("sync", k, lhs, rhs))

        # endurance
        lhs = tau[k]
        rhs = a
        if lhs > rhs + tol:
            violations.append(("endurance", k, lhs, rhs))

        # tau definition
        lhs = abs(tau[k] - (t1[k] + t2[k]))
        if lhs > 1e-5:
            violations.append(("tau_def", k, lhs, 0.0))

        # carrier segment
        prev = depot if k == 0 else lx[k-1]
        lhs = np.linalg.norm(sx[k] - prev)
        rhs = Vc * Tseg[k]
        if lhs > rhs + tol:
            violations.append(("car_seg", k, lhs, rhs))

    # back to depot
    lhs = np.linalg.norm(depot - lx[n-1])
    rhs = Vc * Tseg[n]
    if lhs > rhs + tol:
        violations.append(("back", n, lhs, rhs))

    ok = (len(violations) == 0)
    return ok, violations
