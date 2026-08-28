#!/usr/bin/env python
"""Why does the trained policy place its feet off the table, when the reference doesn't?

Two questions, kept separate:
  1. GEOMETRY - where do the policy's feet actually end up, relative to the
     table's real footprint and surface, at the frames where the eval fails?
  2. REWARD - does the tracking reward that trained this policy actually see a
     bad foot as bad, or does it get diluted away by the other four tracked
     points?

(2) matters because `tracking_vr_5point_local` (weight 2.0, the only per-body
position term) averages squared error across FIVE points - torso, both
wrists, both feet - into one scalar before exponentiating:

    error = mean_over_5_points(squared_error)
    reward = exp(-error / std^2)

A 20 cm miss on one foot, averaged against four points near 0, contributes
1/5 of its own squared error to the mean. This script reproduces that exact
formula (env pelvis = anchor, same offsets, same std) per point, so you can
see the diluted reward next to the per-foot error that produced it.

Usage:
    python data/stage4_foot_placement_diagnostic.py <ref_lib> <trajectory.pkl ...>
"""
import argparse, glob, importlib.util, os
import numpy as np, joblib, mujoco
from scipy.spatial.transform import Rotation as R

_spec = importlib.util.spec_from_file_location(
    "render", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "stage3_render_reference.py"))
_render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_render)
F = _render._feas

POINTS = ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
          "left_ankle_roll_link", "right_ankle_roll_link"]
OFFSETS = np.array([[0, 0, 0.5], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
STD = 0.25  # terrain_tracking_grail.yaml: tracking_vr_5point_local.params.std

TABLE_XY = (-1.00, 1.15, -0.65, 0.67)   # x0,x1,y0,y1 (measured on grail_table_v3)
TABLE_Z = 0.905


def fk_points(m, d, qpos):
    """World position of each of the 5 reward points, and the pelvis pose."""
    pts = np.zeros((len(qpos), len(POINTS), 3))
    pel_p = np.zeros((len(qpos), 3)); pel_q = np.zeros((len(qpos), 4))
    B = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in POINTS]
    Bp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    for f, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        pel_p[f] = d.xpos[Bp]
        pel_q[f] = d.xquat[Bp]
        for i, b in enumerate(B):
            pts[f, i] = d.xpos[b] + OFFSETS[i]
    return pts, pel_p, pel_q


def local_frame(pts, anchor_p, anchor_q):
    """pts expressed in the anchor's local frame: R(anchor_q)^-1 @ (pts - anchor_p)."""
    Rq = R.from_quat(anchor_q[:, [1, 2, 3, 0]]).inv()
    return np.einsum("fij,fpj->fpi",
                     np.stack([r.as_matrix() for r in Rq]),
                     pts - anchor_p[:, None, :])


def five_point_reward(ref_local, pol_local):
    diff = pol_local - ref_local
    err = np.square(diff).sum(-1)              # (T, 5) per-point squared error
    diluted = np.exp(-err.mean(-1) / STD**2)    # what the policy was actually trained on
    per_point = np.exp(-err / STD**2)           # hypothetical if each were its own term
    return err, diluted, per_point


def foot_geometry(m, d, qpos, side):
    B = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == B and m.geom_contype[g]]
    out = np.zeros((len(qpos), 3))
    for f, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        low = min(d.geom_xpos[g][2] - m.geom_rbound[g] for g in gs)
        xy = np.mean([d.geom_xpos[g][:2] for g in gs], axis=0)
        out[f] = [xy[0], xy[1], low]
    return out


def margin_and_gap(sole_xyz):
    x0, x1, y0, y1 = TABLE_XY
    x, y, z = sole_xyz[:, 0], sole_xyz[:, 1], sole_xyz[:, 2]
    inside = (x > x0) & (x < x1) & (y > y0) & (y < y1)
    margin = np.minimum.reduce([x - x0, x1 - x, y - y0, y1 - y])   # + inside, - outside
    gap = np.where(inside, TABLE_Z - z, -z)                        # vs table if over it, vs floor if not
    return margin, gap, inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_lib")
    ap.add_argument("trajectories", nargs="+")
    ap.add_argument("--start", type=int, default=55)
    ap.add_argument("--end", type=int, default=97)
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()

    M = list(joblib.load(glob.glob(f"{a.ref_lib}/robot/*.pkl")[0]).values())[0]
    ref_qpos = np.concatenate([np.array(M["root_trans_offset"]), np.array(M["root_rot"]),
                               np.array(M["dof"])], axis=1)

    m = mujoco.MjModel.from_xml_path(F.G1_MJCF)
    d = mujoco.MjData(m)
    ref_pts, ref_pp, ref_pq = fk_points(m, d, ref_qpos)
    ref_l = local_frame(ref_pts, ref_pp, ref_pq)
    ref_L = foot_geometry(m, d, ref_qpos, "left")
    ref_R = foot_geometry(m, d, ref_qpos, "right")

    # average the policy trajectories (near-identical across eval envs; averaging
    # cancels float/stochastic-action noise without hiding a real per-env split)
    pol_qpos_list = []
    for tp in a.trajectories:
        import pickle
        T = pickle.load(open(tp, "rb"))
        dof = np.asarray(T["dof_pos"])[:, _render.isaac_to_mjcf(m)]
        q = np.concatenate([np.asarray(T["root_pos_w"]), np.asarray(T["root_quat_w"]), dof], 1)
        pol_qpos_list.append(q)
    n = min(len(q) for q in pol_qpos_list)
    pol_qpos = np.mean([q[:n] for q in pol_qpos_list], axis=0)
    ref_qpos_c = ref_qpos[:n]; ref_l_c = ref_l[:n]; ref_L_c = ref_L[:n]; ref_R_c = ref_R[:n]

    pol_pts, pol_pp, pol_pq = fk_points(m, d, pol_qpos)
    pol_l = local_frame(pol_pts, pol_pp, pol_pq)
    pol_L = foot_geometry(m, d, pol_qpos, "left")
    pol_R = foot_geometry(m, d, pol_qpos, "right")

    err, diluted, per_point = five_point_reward(ref_l_c, pol_l)
    mL, gL, inL = margin_and_gap(pol_L)
    mR, gR, inR = margin_and_gap(pol_R)
    mL_ref, gL_ref, _ = margin_and_gap(ref_L_c)
    mR_ref, gR_ref, _ = margin_and_gap(ref_R_c)

    win = slice(a.start, min(a.end, n))
    li, ri = POINTS.index("left_ankle_roll_link"), POINTS.index("right_ankle_roll_link")
    ti = POINTS.index("torso_link")
    lwi, rwi = POINTS.index("left_wrist_yaw_link"), POINTS.index("right_wrist_yaw_link")
    print(f"\nALL 5 tracked points, squared error (m^2) per frame - which point is actually worst?")
    print(f"{'f':>4} {'torso':>8} {'Lwrist':>8} {'Rwrist':>8} {'Lfoot':>8} {'Rfoot':>8}   worst")
    for f in range(a.start, min(a.end, n), a.stride):
        row = err[f]
        worst = POINTS[int(np.argmax(row))]
        print(f"{f:4d} {row[ti]:8.4f} {row[lwi]:8.4f} {row[rwi]:8.4f} {row[li]:8.4f} {row[ri]:8.4f}   {worst}")
    print(f"\nmean squared error per point over the window:")
    for i,nm in enumerate(POINTS):
        print(f"  {nm:<22} {err[win,i].mean():.4f}")
    print(f"\n=== foot placement diagnostic: {os.path.basename(a.ref_lib)}, "
          f"{len(a.trajectories)} eval traj(s) averaged ===")
    print(f"reward as trained: mean over 5 points, std={STD}   (dilution factor exactly 5x)")
    print(f"\n{'f':>4} | {'L margin':>9} {'L gap':>7} {'L err':>7} {'L rew_alone':>11} | "
          f"{'R margin':>9} {'R gap':>7} {'R err':>7} {'R rew_alone':>11} | {'5pt reward':>10}")
    for f in range(a.start, min(a.end, n), a.stride):
        print(f"{f:4d} | {mL[f]:9.3f} {gL[f]:7.3f} {err[f,li]:7.3f} {per_point[f,li]:11.3f} | "
              f"{mR[f]:9.3f} {gR[f]:7.3f} {err[f,ri]:7.3f} {per_point[f,ri]:11.3f} | {diluted[f]:10.3f}")

    print(f"\nREFERENCE for comparison (same frames, same table):")
    print(f"{'f':>4} | {'L margin':>9} {'L gap':>7} | {'R margin':>9} {'R gap':>7}")
    for f in range(a.start, min(a.end, n), a.stride):
        print(f"{f:4d} | {mL_ref[f]:9.3f} {gL_ref[f]:7.3f} | {mR_ref[f]:9.3f} {gR_ref[f]:7.3f}")

    print(f"\n--- window f{a.start}-{a.end} summary ---")
    print(f"  policy L: margin min {mL[win].min():+.3f} (outside table on "
          f"{int((mL[win] < 0).sum())}/{win.stop-win.start} frames), gap max {gL[win].max():+.3f}")
    print(f"  policy R: margin min {mR[win].min():+.3f} (outside table on "
          f"{int((mR[win] < 0).sum())}/{win.stop-win.start} frames), gap max {gR[win].max():+.3f}")
    print(f"  reference L: margin min {mL_ref[win].min():+.3f}, gap max {gL_ref[win].max():+.3f}")
    print(f"  reference R: margin min {mR_ref[win].min():+.3f}, gap max {gR_ref[win].max():+.3f}")
    print(f"  mean per-foot squared error in window: L {err[win,li].mean():.4f}  R {err[win,ri].mean():.4f}")
    print(f"  mean 5-point reward actually paid in window: {diluted[win].mean():.3f}")
    print(f"  what a foot-ONLY reward would have paid (mean): "
          f"L {per_point[win,li].mean():.3f}  R {per_point[win,ri].mean():.3f}")


if __name__ == "__main__":
    main()
