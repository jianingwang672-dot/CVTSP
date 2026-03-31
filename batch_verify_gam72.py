# import re
# from pathlib import Path
# import numpy as np
# import pandas as pd

# from CVTSP_SOCP import gurobi_cvp_socp, check_cvp_solution


# # ========= 配置区：改这里就够了 =========
# INST_DIR = Path(r"C:\Users\WJN\Documents\wjn的文件\CVTSP\instance\Gam72")
# REF_DIR  = Path(r"C:\Users\WJN\Documents\wjn的文件\CVTSP\refs\Gam72")

# # 容差
# TOL_FEAS = 1e-4
# ABS_TOL  = 1e-3
# REL_TOL  = 1e-6

# THREADS = None
# TIME_LIMIT = None
# # ======================================


# _NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


# def parse_instance_txt(text: str):
#     """
#     解析 instance Example_i.txt
#     Gam72 命名约定（你已验证）：
#       Vd = UAV speed
#       Vv = carrier speed
#       MT = endurance time a
#       p: position coordinates =  (共 J+1 行，首行为 depot)

#     返回：
#       J, depot(2,), targets(J,2), Vc(carrier), Vv(uav), a
#     """
#     J  = int(re.search(r"\bJ\s*=\s*(\d+)", text).group(1))
#     MT = float(re.search(r"\bMT\s*=\s*(" + _NUM_RE + r")", text).group(1))
#     Vd = float(re.search(r"\bVd\s*=\s*(" + _NUM_RE + r")", text).group(1))
#     Vc_dat = float(re.search(r"\bVv\s*=\s*(" + _NUM_RE + r")", text).group(1))

#     m = re.search(r"p:\s*position coordinates\s*=\s*([\s\S]+)", text, flags=re.IGNORECASE)
#     if not m:
#         raise ValueError("Instance missing 'p: position coordinates =' block")

#     coords_lines = [ln.strip() for ln in m.group(1).strip().splitlines() if ln.strip()]

#     coords = []
#     for ln in coords_lines:
#         # 遇到非数字开头行就停（通常是下一段标题）
#         if re.match(r"^[A-Za-z]", ln):
#             break
#         parts = ln.split()
#         if len(parts) >= 2 and re.match(r"^[\+\-]?\d", parts[0]):
#             coords.append([float(parts[0]), float(parts[1])])

#     coords = np.array(coords, dtype=float)
#     if coords.shape[0] != J + 1:
#         raise ValueError(f"Expected {J+1} points (including depot), got {coords.shape[0]}")

#     depot = coords[0]
#     targets = coords[1:]  # (J,2)

#     # 映射到 CVP 记号
#     Vc = Vc_dat   # carrier speed
#     Vv = Vd       # UAV speed
#     a  = MT       # endurance time
#     return J, depot, targets, Vc, Vv, a


# def parse_ref_example_txt(text: str):
#     """解析 ref 文件中的 objective + route"""
#     mobj = re.search(r"objective\s*=\s*(" + _NUM_RE + r")", text, flags=re.IGNORECASE)
#     if not mobj:
#         raise ValueError("Ref file missing 'objective='")
#     ref_obj = float(mobj.group(1))

#     mroute = re.search(r"=+Route=+\s*([\s\S]*?)=+", text, flags=re.IGNORECASE)
#     if not mroute:
#         raise ValueError("Ref file missing '========Route=========' block")

#     route_block = mroute.group(1)
#     nodes = [int(x) for x in re.findall(r"\d+", route_block)]
#     if len(nodes) < 2:
#         raise ValueError("Route parse failed: too few nodes")
#     if nodes[0] != 0 or nodes[-1] != 0:
#         raise ValueError("Route should start/end with depot 0.")
#     return ref_obj, nodes


# def parse_ref_takeoff_land(txt: str, J: int):
#     """
#     解析 ref 文件中的 Take-off/Land 位置（支持多行、逗号、科学计数法）
#     返回：
#       sx_ref_raw, lx_ref_raw shape (J,2)
#     重要约定（我们在主逻辑里会“定死”）：
#       这里返回的 raw 默认按 label-order（第 l 行对应 label=l）
#     """
#     def grab_block(headers):
#         for header in headers:
#             m = re.search(re.escape(header) + r"\s*\n([\s\S]*?)(?=\n=+|$)", txt)
#             if m:
#                 nums = re.findall(_NUM_RE, m.group(1))
#                 return np.array([float(x) for x in nums], dtype=float)
#         raise ValueError(f"Header not found (any of): {headers}")

#     sx_x = grab_block(["========Take-off loaction(x)=========", "========Take-off location(x)========="])
#     sx_y = grab_block(["========Take-off loaction(y)=========", "========Take-off location(y)========="])
#     lx_x = grab_block(["========Land loaction(x)=========",     "========Land location(x)========="])
#     lx_y = grab_block(["========Land loaction(y)=========",     "========Land location(y)========="])

#     if not (len(sx_x) == len(sx_y) == len(lx_x) == len(lx_y) == J):
#         raise ValueError(
#             f"take-off/land length mismatch: "
#             f"len(sx_x)={len(sx_x)}, len(sx_y)={len(sx_y)}, len(lx_x)={len(lx_x)}, len(lx_y)={len(lx_y)}, J={J}"
#         )

#     sx_ref_raw = np.column_stack([sx_x, sx_y])
#     lx_ref_raw = np.column_stack([lx_x, lx_y])
#     return sx_ref_raw, lx_ref_raw


# def parse_ref_times(txt: str, J: int):
#     """
#     解析 ref 里的 t[k], t1[k], t2[k]（如果存在）。
#     关键：遇到下一段（如 T[i][j]= 或下一个 ======标题======）就截断，避免把矩阵数字吞进去。
#     返回：t_ref, t1_ref, t2_ref（可能为 None）
#     """
#     def grab_by_header(header_regex: str):
#         m = re.search(header_regex, txt, flags=re.IGNORECASE)
#         if not m:
#             return None

#         start = m.end()

#         next_markers = [
#             r"\n\s*T\s*\[i\]\s*\[j\]\s*=",   # T[i][j]=
#             r"\n\s*=+\s*[A-Za-z].*?=+\s*\n", # 下一个标题块
#         ]

#         end = len(txt)
#         for pat in next_markers:
#             mm = re.search(pat, txt[start:], flags=re.IGNORECASE)
#             if mm:
#                 end = min(end, start + mm.start())

#         block = txt[start:end]
#         nums = re.findall(_NUM_RE, block)
#         if not nums:
#             return None
#         return np.array([float(x) for x in nums], dtype=float)

#     t_ref  = grab_by_header(r"=+\s*t\s*\[k\]\s*=+\s*\n")
#     t1_ref = grab_by_header(r"=+\s*t1\s*\[k\]\s*=+\s*\n")
#     t2_ref = grab_by_header(r"=+\s*t2\s*\[k\]\s*=+\s*\n")

#     for name, arr in [("t", t_ref), ("t1", t1_ref), ("t2", t2_ref)]:
#         if arr is not None and len(arr) != J:
#             raise ValueError(f"ref {name}[k] length mismatch: len={len(arr)} J={J}")

#     return t_ref, t1_ref, t2_ref


# def parse_ref_T_matrix(txt: str, J: int):
#     """
#     解析 ref 里的 T[i][j] 矩阵（如果存在）。
#     期望矩阵大小为 (J+1, J+1)，包含 depot=0。
#     返回：T_mat 或 None
#     """
#     m = re.search(r"T\s*\[i\]\s*\[j\]\s*=\s*\n([\s\S]+)", txt)
#     if not m:
#         return None

#     tail = m.group(1)
#     nums = re.findall(_NUM_RE, tail)
#     need = (J + 1) * (J + 1)
#     if len(nums) < need:
#         return None

#     vals = np.array([float(x) for x in nums[:need]], dtype=float)
#     return vals.reshape((J + 1, J + 1))


# def save_csv_safe(df, path_str: str):
#     """安全保存 CSV，避免被占用文件报错"""
#     path = Path(path_str)

#     if path.exists():
#         try:
#             path.unlink()
#         except Exception as e:
#             print(f"  [Warning] Could not delete old {path}: {e}")

#     try:
#         df.to_csv(path, index=False, encoding="utf-8-sig")
#         print(f"Saved: {path.resolve()}")
#         return True
#     except PermissionError as e:
#         print(f"ERROR: Cannot write {path}")
#         print(f"  Reason: {e}")
#         print(f"  -> Please close this file in Excel or any other editor, then run again")
#         return False


# def run_one(idx: int):
#     fname = f"Example_{idx}.txt"
#     inst_path = INST_DIR / fname
#     ref_path  = REF_DIR  / fname

#     if not inst_path.exists():
#         raise FileNotFoundError(f"Missing instance file: {inst_path}")
#     if not ref_path.exists():
#         raise FileNotFoundError(f"Missing ref file: {ref_path}")

#     inst_text = inst_path.read_text(encoding="utf-8", errors="ignore")
#     ref_text  = ref_path.read_text(encoding="utf-8", errors="ignore")

#     J, depot, targets, Vc, Vv, a = parse_instance_txt(inst_text)
#     ref_obj, route = parse_ref_example_txt(ref_text)

#     route_ref_str = "->".join(map(str, route))

#     # ======== Order 定死（核心修复）========
#     # route 中间是 label(1..J) 的访问顺序；我们统一转成 0-based 给 Gurobi/Check
#     # tour0: visit-order, 0-based 目标索引（0..J-1）
#     tour0 = [node - 1 for node in route[1:-1]]
#     if len(tour0) != J:
#         raise ValueError(f"tour length mismatch: len(tour0)={len(tour0)} J={J}")

#     if not (min(tour0) == 0 and max(tour0) == J - 1 and len(set(tour0)) == J):
#         raise ValueError(f"tour0 invalid (not a permutation 0..{J-1}): min={min(tour0)}, max={max(tour0)}, unique={len(set(tour0))}")

#     # labels_visit: visit-order 下对应的 label(1..J)
#     labels_visit = [j0 + 1 for j0 in tour0]
#     tour_used_str = ",".join(map(str, tour0))
#     # ======================================

#     sol = gurobi_cvp_socp(
#         depot, targets, tour0,
#         Vc=Vc, Vv=Vv, endurance_a=a,
#         threads=THREADS, time_limit=TIME_LIMIT,
#         test=True
#     )

#     ok, viol = check_cvp_solution(depot, targets, tour0, Vc, Vv, a, sol, tol=TOL_FEAS)

#     ours_obj = float(sol["obj"])
#     abs_diff = abs(ours_obj - ref_obj)
#     rel_diff = abs_diff / max(1.0, abs(ref_obj))
#     pass_obj = (abs_diff <= ABS_TOL) or (rel_diff <= REL_TOL)

#     status = int(sol.get("status", -1))

#     main_row = {
#         "idx": idx,
#         "file": fname,
#         "J": J,
#         "Vc(carrier)": Vc,
#         "Vv(uav)": Vv,
#         "a(MT)": a,
#         "ref_obj": ref_obj,
#         "ours_obj": ours_obj,
#         "abs_diff": abs_diff,
#         "rel_diff": rel_diff,
#         "feasible": ok,
#         "pass_obj": pass_obj,
#         "status": status,
#         "n_viol": 0 if ok else len(viol),
#         "first_viol": "" if ok else str(viol[0]),
#         "route_ref_str": route_ref_str,
#         "tour0_used_str": tour_used_str,
#         "tour0_len": len(tour0),
#         "tour0_min": min(tour0),
#         "tour0_max": max(tour0),
#     }

#     # ========= pointwise + edge(Tij) =========
#     point_rows = []
#     edge_rows = []

#     try:
#         # ref raw（我们定死它是 label-order）
#         sx_ref_raw, lx_ref_raw = parse_ref_takeoff_land(ref_text, J)
#         t_ref, t1_ref, t2_ref = parse_ref_times(ref_text, J)

#         sx_ours_visit = np.asarray(sol["sx"], dtype=float)  # (J,2) visit-order
#         lx_ours_visit = np.asarray(sol["lx"], dtype=float)
#         t1_ours = np.asarray(sol.get("t1", np.full(J, np.nan)), dtype=float)
#         t2_ours = np.asarray(sol.get("t2", np.full(J, np.nan)), dtype=float)
#         tau_ours = np.asarray(sol.get("tau", np.full(J, np.nan)), dtype=float)

#         # ======== Order 定死（核心修复）========
#         # ref 强制视为 label-order（第 l 行就是 label=l）
#         # 重排到 visit-order：按 labels_visit（1..J）取 l-1 行
#         sx_ref_visit = np.vstack([sx_ref_raw[l - 1] for l in labels_visit])
#         lx_ref_visit = np.vstack([lx_ref_raw[l - 1] for l in labels_visit])
#         ref_mode = "label-order -> visit-order (fixed)"

#         if t1_ref is not None and t2_ref is not None:
#             t1_ref_visit = np.array([t1_ref[l - 1] for l in labels_visit], dtype=float)
#             t2_ref_visit = np.array([t2_ref[l - 1] for l in labels_visit], dtype=float)
#             if t_ref is not None:
#                 tau_ref_visit = np.array([t_ref[l - 1] for l in labels_visit], dtype=float)
#             else:
#                 tau_ref_visit = t1_ref_visit + t2_ref_visit
#         else:
#             t1_ref_visit = t2_ref_visit = tau_ref_visit = None
#         # ======================================

#         # pointwise 行（visit-order）
#         for k in range(J):  # k = 0..J-1
#             j0 = tour0[k]           # 0..J-1
#             label = j0 + 1          # 1..J
#             p = targets[j0]

#             so = sx_ours_visit[k]; lo = lx_ours_visit[k]
#             sr = sx_ref_visit[k];  lr = lx_ref_visit[k]

#             row = {
#                 "case_idx": idx,
#                 "file": fname,
#                 "ref_mode": ref_mode,

#                 "k_visit_0based": k,
#                 "label_1based": label,
#                 "target_idx0": j0,
#                 "p_x": p[0], "p_y": p[1],

#                 "sx_ours_x": so[0], "sx_ours_y": so[1],
#                 "sx_ref_x":  sr[0], "sx_ref_y":  sr[1],
#                 "sx_err_norm": float(np.linalg.norm(so - sr)),

#                 "lx_ours_x": lo[0], "lx_ours_y": lo[1],
#                 "lx_ref_x":  lr[0], "lx_ref_y":  lr[1],
#                 "lx_err_norm": float(np.linalg.norm(lo - lr)),

#                 "t1_ours": float(t1_ours[k]) if k < len(t1_ours) else np.nan,
#                 "t2_ours": float(t2_ours[k]) if k < len(t2_ours) else np.nan,
#                 "tau_ours": float(tau_ours[k]) if k < len(tau_ours) else np.nan,
#             }

#             if t1_ref_visit is not None:
#                 row["t1_ref"] = float(t1_ref_visit[k])
#                 row["t2_ref"] = float(t2_ref_visit[k])
#                 row["tau_ref"] = float(tau_ref_visit[k])
#                 row["t1_abs_err"] = abs(row["t1_ours"] - row["t1_ref"])
#                 row["t2_abs_err"] = abs(row["t2_ours"] - row["t2_ref"])
#                 row["tau_abs_err"] = abs(row["tau_ours"] - row["tau_ref"])
#             else:
#                 row["t1_ref"] = row["t2_ref"] = row["tau_ref"] = np.nan
#                 row["t1_abs_err"] = row["t2_abs_err"] = row["tau_abs_err"] = np.nan

#             point_rows.append(row)

#         # edge Tij 表（如果 ref 有 T 矩阵）
#         T_mat = parse_ref_T_matrix(ref_text, J)
#         if T_mat is not None:
#             for e in range(len(route) - 1):
#                 i = route[e]
#                 jnode = route[e + 1]
#                 edge_rows.append({
#                     "case_idx": idx,
#                     "file": fname,
#                     "edge_idx_1based": e + 1,
#                     "i": i,
#                     "j": jnode,
#                     "T_ref": float(T_mat[i, jnode]),
#                 })

#     except Exception as e:
#         print(f"  [Warning] Case {idx}: Failed to parse takeoff/land/times/Tij: {e}")

#     return main_row, point_rows, edge_rows


# def main():
#     rows = []
#     fails = []
#     all_point_rows = []
#     all_edge_rows = []

#     for idx in range(1, 73):
#         try:
#             r, point_rows, edge_rows = run_one(idx)
#             rows.append(r)
#             all_point_rows.extend(point_rows)
#             all_edge_rows.extend(edge_rows)

#             flag = (r.get("feasible", False) is True) and (r.get("pass_obj", False) is True)
#             if not flag:
#                 fails.append(r)

#             print(f"[{idx:02d}] abs={r.get('abs_diff', float('nan')):.3e}, "
#                   f"rel={r.get('rel_diff', float('nan')):.3e}, "
#                   f"feas={r.get('feasible')}, pass_obj={r.get('pass_obj')}")
#         except Exception as e:
#             rr = {"idx": idx, "file": f"Example_{idx}.txt", "error": repr(e)}
#             rows.append(rr)
#             fails.append(rr)
#             print(f"[{idx:02d}] ERROR: {e}")

#     df = pd.DataFrame(rows)
#     if not save_csv_safe(df, "gam72_cvp_batch_report.csv"):
#         return

#     if fails:
#         df_fail = pd.DataFrame(fails)
#         save_csv_safe(df_fail, "gam72_failed_cases.csv")

#     if all_point_rows:
#         df_point = pd.DataFrame(all_point_rows)
#         save_csv_safe(df_point, "gam72_pointwise_compare.csv")

#     if all_edge_rows:
#         df_edge = pd.DataFrame(all_edge_rows)
#         save_csv_safe(df_edge, "gam72_edge_Tij.csv")

#     if "feasible" in df.columns and "pass_obj" in df.columns:
#         n_total = len(df)
#         n_pass = int(((df["feasible"] == True) & (df["pass_obj"] == True)).sum())
#         print("\n==== SUMMARY ====")
#         print(f"Total: {n_total}, Pass: {n_pass}, Fail: {n_total - n_pass}")


# if __name__ == "__main__":
#     main()
# batch_verify_gam72.py
# -*- coding: utf-8 -*-

# batch_verify_gam72.py
# -*- coding: utf-8 -*-

import re
from pathlib import Path
import numpy as np
import pandas as pd

# ---- matplotlib: 强制无GUI后端，避免 plt.show 卡死/阻塞 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from CVTSP_SOCP import gurobi_cvp_socp, check_cvp_solution


# ========= 配置区：改这里就够了 =========
INST_DIR = Path(r"C:\Users\WJN\Documents\wjn的文件\CVTSP\instance\Gam72")
REF_DIR  = Path(r"C:\Users\WJN\Documents\wjn的文件\CVTSP\refs\Gam72")

# 容差
TOL_FEAS = 1e-4
ABS_TOL  = 1e-3
REL_TOL  = 1e-6

THREADS = None
TIME_LIMIT = None

# 先只跑 Example_1，确认没问题再改成 73
RUN_FROM = 1
RUN_TO_EXCLUSIVE = 2   # 2 => only Example_1; 改成 73 => 跑 1..72
# ======================================


_NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def save_csv_safe(df: pd.DataFrame, path_str: str) -> bool:
    """安全保存 CSV，避免被占用文件报错"""
    path = Path(path_str)

    if path.exists():
        try:
            path.unlink()
        except Exception as e:
            print(f"  [Warning] Could not delete old {path}: {e}")

    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved: {path.resolve()}")
        return True
    except PermissionError as e:
        print(f"ERROR: Cannot write {path}")
        print(f"  Reason: {e}")
        print(f"  -> Please close this file in Excel / VSCode preview / any editor, then run again")
        return False


def parse_instance_txt(text: str):
    """
    解析 instance Example_i.txt
    Gam72 命名约定（你已验证）：
      Vd = UAV speed
      Vv = carrier speed
      MT = endurance time a
      p: position coordinates =  (共 J+1 行，首行为 depot)

    返回：
      J, depot(2,), targets(J,2), Vc(carrier), Vv(uav), a
    """
    J  = int(re.search(r"\bJ\s*=\s*(\d+)", text).group(1))
    MT = float(re.search(r"\bMT\s*=\s*(" + _NUM_RE + r")", text).group(1))
    Vd = float(re.search(r"\bVd\s*=\s*(" + _NUM_RE + r")", text).group(1))
    Vc_dat = float(re.search(r"\bVv\s*=\s*(" + _NUM_RE + r")", text).group(1))

    m = re.search(r"p:\s*position coordinates\s*=\s*([\s\S]+)", text, flags=re.IGNORECASE)
    if not m:
        raise ValueError("Instance missing 'p: position coordinates =' block")

    coords_lines = [ln.strip() for ln in m.group(1).strip().splitlines() if ln.strip()]

    coords = []
    for ln in coords_lines:
        # 遇到非数字开头行就停（通常是下一段标题）
        if re.match(r"^[A-Za-z]", ln):
            break
        parts = ln.split()
        if len(parts) >= 2 and re.match(r"^[\+\-]?\d", parts[0]):
            coords.append([float(parts[0]), float(parts[1])])

    coords = np.array(coords, dtype=float)
    if coords.shape[0] != J + 1:
        raise ValueError(f"Expected {J+1} points (including depot), got {coords.shape[0]}")

    depot = coords[0]
    targets = coords[1:]  # (J,2)

    # 映射到 CVP 记号
    Vc = Vc_dat   # carrier speed
    Vv = Vd       # UAV speed
    a  = MT       # endurance time
    return J, depot, targets, Vc, Vv, a


def parse_ref_example_txt(text: str):
    """解析 ref 文件中的 objective + route"""
    mobj = re.search(r"objective\s*=\s*(" + _NUM_RE + r")", text, flags=re.IGNORECASE)
    if not mobj:
        raise ValueError("Ref file missing 'objective='")
    ref_obj = float(mobj.group(1))

    mroute = re.search(r"=+Route=+\s*([\s\S]*?)=+", text, flags=re.IGNORECASE)
    if not mroute:
        raise ValueError("Ref file missing '========Route=========' block")

    route_block = mroute.group(1)
    nodes = [int(x) for x in re.findall(r"\d+", route_block)]
    if len(nodes) < 2:
        raise ValueError("Route parse failed: too few nodes")
    if nodes[0] != 0 or nodes[-1] != 0:
        raise ValueError("Route should start/end with depot 0.")
    return ref_obj, nodes


def plot_example_points_and_circles(
    depot, targets, tour0, Vv, sol,
    save_path="example1_plot.png",
    title="Example_1 - ours (label + circles)"
):
    """
    只画：depot、targets、我们的 takeoff/landing、两类圆：
      圆心 p_j，半径 Vv*t1、Vv*t2
    不弹窗，只保存 PNG，避免卡住。
    """
    depot = np.asarray(depot, float).reshape(2,)
    targets = np.asarray(targets, float)

    sx = np.asarray(sol["sx"], float)   # (J,2) visit-order
    lx = np.asarray(sol["lx"], float)
    t1 = np.asarray(sol["t1"], float)
    t2 = np.asarray(sol["t2"], float)

    J = len(tour0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)

    # depot
    ax.scatter([depot[0]], [depot[1]], marker="s", s=120)

    for k in range(J):
        j0 = int(tour0[k])
        label = j0 + 1
        p = targets[j0]
        s = sx[k]
        l = lx[k]

        r1 = float(Vv) * float(t1[k])
        r2 = float(Vv) * float(t2[k])

        # points
        ax.scatter([p[0]], [p[1]], marker="^", s=90)
        ax.scatter([s[0]], [s[1]], marker="o", s=45)
        ax.scatter([l[0]], [l[1]], marker="x", s=55)

        # segments
        ax.plot([s[0], p[0]], [s[1], p[1]], linewidth=1)
        ax.plot([p[0], l[0]], [p[1], l[1]], linewidth=1)

        # circles: two styles
        ax.add_patch(Circle((p[0], p[1]), r1, fill=False, edgecolor="C0", linestyle="--", linewidth=1))
        ax.add_patch(Circle((p[0], p[1]), r2, fill=False, edgecolor="C1", linestyle=":",  linewidth=1))

        # labels (label / s{label} / l{label})
        ax.text(p[0], p[1], f"{label}", fontsize=10, ha="left", va="bottom")
        ax.text(s[0], s[1], f"s{label}", fontsize=9, ha="right", va="top")
        ax.text(l[0], l[1], f"l{label}", fontsize=9, ha="left", va="bottom")

    legend_elements = [
        Line2D([0],[0], marker='s', color='w', markerfacecolor='C2', markersize=10, label='Depot (0)'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='C3', markersize=10, label='Target p(label)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='C4', markersize=8,  label='Take-off s(label)'),
        Line2D([0],[0], marker='x', color='C5', markersize=8, label='Landing l(label)'),
        Line2D([0],[0], color='C0', linestyle='--', label='Outbound circle: radius = Vv * t1'),
        Line2D([0],[0], color='C1', linestyle=':',  label='Inbound circle: radius = Vv * t2'),
    ]
    ax.legend(handles=legend_elements, loc="best")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def slack_table(depot, targets, tour0, Vc, Vv, a, sol) -> pd.DataFrame:
    """
    输出每个 k 的 slack：
      slack_out = Vv*t1 - ||s - p||
      slack_in  = Vv*t2 - ||l - p||
      slack_sync = Vc*tau - ||l - s||
      slack_car  = Vc*Tseg - ||s - prev||
      slack_endurance = a - tau
    同时写入距离与 rhs，便于你对比“贴边/不贴边”。
    """
    depot = np.asarray(depot, float).reshape(2,)
    targets = np.asarray(targets, float)

    sx = np.asarray(sol["sx"], float)
    lx = np.asarray(sol["lx"], float)
    t1 = np.asarray(sol["t1"], float)
    t2 = np.asarray(sol["t2"], float)
    tau = np.asarray(sol["tau"], float)
    Tseg = np.asarray(sol["Tseg"], float)

    rows = []
    n = len(tour0)

    for k in range(n):
        j0 = int(tour0[k])
        label = j0 + 1
        p = targets[j0]
        prev = depot if k == 0 else lx[k-1]

        d_out = float(np.linalg.norm(sx[k] - p))
        d_in  = float(np.linalg.norm(lx[k] - p))
        d_sync = float(np.linalg.norm(lx[k] - sx[k]))
        d_car = float(np.linalg.norm(sx[k] - prev))

        rhs_out = float(Vv * t1[k])
        rhs_in  = float(Vv * t2[k])
        rhs_sync = float(Vc * tau[k])
        rhs_car  = float(Vc * Tseg[k])

        rows.append({
            "k_visit_1based": k + 1,
            "label_1based": label,

            "d_out": d_out,
            "Vv_t1": rhs_out,
            "slack_out": rhs_out - d_out,

            "d_in": d_in,
            "Vv_t2": rhs_in,
            "slack_in": rhs_in - d_in,

            "d_sync": d_sync,
            "Vc_tau": rhs_sync,
            "slack_sync": rhs_sync - d_sync,

            "d_car": d_car,
            "Vc_Tseg": rhs_car,
            "slack_car": rhs_car - d_car,

            "tau": float(tau[k]),
            "a": float(a),
            "slack_endurance": float(a - tau[k]),

            "tau_minus_t1t2": float(tau[k] - (t1[k] + t2[k])),
        })

    # final return
    back_d = float(np.linalg.norm(depot - lx[n-1]))
    back_rhs = float(Vc * Tseg[n])
    rows.append({
        "k_visit_1based": "back",
        "label_1based": -1,
        "d_out": np.nan, "Vv_t1": np.nan, "slack_out": np.nan,
        "d_in": np.nan,  "Vv_t2": np.nan, "slack_in": np.nan,
        "d_sync": np.nan,"Vc_tau": np.nan, "slack_sync": np.nan,
        "d_car": back_d, "Vc_Tseg": back_rhs, "slack_car": back_rhs - back_d,
        "tau": np.nan, "a": np.nan, "slack_endurance": np.nan,
        "tau_minus_t1t2": np.nan,
    })

    return pd.DataFrame(rows)


def print_slack_summary(df_slack: pd.DataFrame):
    """终端打印：最不贴边的点（slack 最大） + 最小 slack（检查是否违反）"""
    dfk = df_slack[df_slack["k_visit_1based"] != "back"].copy()

    print("\nTop-5 slack_out (largest => outbound 最不贴边):")
    print(dfk.sort_values("slack_out", ascending=False)
          [["k_visit_1based", "label_1based", "slack_out", "d_out", "Vv_t1"]]
          .head(5)
          .to_string(index=False))

    print("\nTop-5 slack_in (largest => inbound 最不贴边):")
    print(dfk.sort_values("slack_in", ascending=False)
          [["k_visit_1based", "label_1based", "slack_in", "d_in", "Vv_t2"]]
          .head(5)
          .to_string(index=False))

    print("\nMin slacks (should be >= -tol if feasible):")
    cols = ["slack_out", "slack_in", "slack_sync", "slack_car", "slack_endurance"]
    print(dfk[cols].min().to_string())


def run_one(idx: int):
    fname = f"Example_{idx}.txt"
    inst_path = INST_DIR / fname
    ref_path  = REF_DIR  / fname

    if not inst_path.exists():
        raise FileNotFoundError(f"Missing instance file: {inst_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing ref file: {ref_path}")

    inst_text = inst_path.read_text(encoding="utf-8", errors="ignore")
    ref_text  = ref_path.read_text(encoding="utf-8", errors="ignore")

    J, depot, targets, Vc, Vv, a = parse_instance_txt(inst_text)
    ref_obj, route = parse_ref_example_txt(ref_text)

    route_ref_str = "->".join(map(str, route))

    # route 中间是 label(1..J) 的访问顺序；转成 0-based target index
    tour0 = [node - 1 for node in route[1:-1]]
    if len(tour0) != J:
        raise ValueError(f"tour length mismatch: len(tour0)={len(tour0)} J={J}")

    if not (min(tour0) == 0 and max(tour0) == J - 1 and len(set(tour0)) == J):
        raise ValueError(f"tour0 invalid permutation 0..{J-1}")

    # ---- solve ----
    sol = gurobi_cvp_socp(
        depot, targets, tour0,
        Vc=Vc, Vv=Vv, endurance_a=a,
        threads=THREADS, time_limit=TIME_LIMIT,
        test=True
    )

    # ---- feasibility check ----
    ok, viol = check_cvp_solution(depot, targets, tour0, Vc, Vv, a, sol, tol=TOL_FEAS)

    ours_obj = float(sol["obj"])
    abs_diff = abs(ours_obj - ref_obj)
    rel_diff = abs_diff / max(1.0, abs(ref_obj))
    pass_obj = (abs_diff <= ABS_TOL) or (rel_diff <= REL_TOL)

    status = int(sol.get("status", -1)) if isinstance(sol, dict) else -1

    main_row = {
        "idx": idx,
        "file": fname,
        "J": J,
        "Vc(carrier)": Vc,
        "Vv(uav)": Vv,
        "a(MT)": a,
        "ref_obj": float(ref_obj),
        "ours_obj": float(ours_obj),
        "abs_diff": float(abs_diff),
        "rel_diff": float(rel_diff),
        "feasible": bool(ok),
        "pass_obj": bool(pass_obj),
        "status": status,
        "n_viol": 0 if ok else len(viol),
        "first_viol": "" if ok else str(viol[0]),
        "route_ref_str": route_ref_str,
        "tour0_used_str": ",".join(map(str, tour0)),
    }

    # ---- Example_1 extra debug: slack + plot ----
    if idx == 1:
        # slack report
        df_slack = slack_table(depot, targets, tour0, Vc, Vv, a, sol)
        save_csv_safe(df_slack, "example1_slack_report.csv")
        print_slack_summary(df_slack)

        # plot
        plot_example_points_and_circles(
            depot, targets, tour0, Vv, sol,
            save_path="example1_plot.png",
            title="Example_1 - Gurobi (label + circles Vv*t1 & Vv*t2)"
        )

    # 这版先不做 pointwise/ref 坐标对比，避免你再被 ref 解析问题卡住
    return main_row, [], []


def main():
    rows = []
    fails = []
    all_point_rows = []
    all_edge_rows = []

    for idx in range(RUN_FROM, RUN_TO_EXCLUSIVE):
        try:
            print(f"\n--- Running {idx} ---")
            r, point_rows, edge_rows = run_one(idx)

            rows.append(r)
            all_point_rows.extend(point_rows)
            all_edge_rows.extend(edge_rows)

            flag = (r.get("feasible", False) is True) and (r.get("pass_obj", False) is True)
            if not flag:
                fails.append(r)

            print(f"[{idx:02d}] abs={r.get('abs_diff', float('nan')):.3e}, "
                  f"rel={r.get('rel_diff', float('nan')):.3e}, "
                  f"feas={r.get('feasible')}, pass_obj={r.get('pass_obj')}")
        except Exception as e:
            rr = {"idx": idx, "file": f"Example_{idx}.txt", "error": repr(e)}
            rows.append(rr)
            fails.append(rr)
            print(f"[{idx:02d}] ERROR: {e}")

    df = pd.DataFrame(rows)
    if not save_csv_safe(df, "gam72_cvp_batch_report.csv"):
        return

    if fails:
        df_fail = pd.DataFrame(fails)
        save_csv_safe(df_fail, "gam72_failed_cases.csv")

    if all_point_rows:
        df_point = pd.DataFrame(all_point_rows)
        save_csv_safe(df_point, "gam72_pointwise_compare.csv")

    if all_edge_rows:
        df_edge = pd.DataFrame(all_edge_rows)
        save_csv_safe(df_edge, "gam72_edge_Tij.csv")

    if "feasible" in df.columns and "pass_obj" in df.columns:
        n_total = len(df)
        n_pass = int(((df["feasible"] == True) & (df["pass_obj"] == True)).sum())
        print("\n==== SUMMARY ====")
        print(f"Total: {n_total}, Pass: {n_pass}, Fail: {n_total - n_pass}")

    print("\nDone.")
    print(" - Plot saved as: example1_plot.png")
    print(" - Slack report:  example1_slack_report.csv")
    print(" - Batch report:  gam72_cvp_batch_report.csv")


if __name__ == "__main__":
    main()