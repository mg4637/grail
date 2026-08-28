#!/usr/bin/env python
"""Fix a leg that never plants during a walk, by extending it at its own footfalls.

THE DEFECT
----------
data/stage3_mujoco_feasibility.py-style FK on grail_chair_v2's walk phase (source
frames 0-96, the chair carry) shows the retargeted right foot never within 2 cm
of the floor for the ~4 s walk - its minimum sole height is +0.023 m, while the
left foot repeatedly reaches -0.007..-0.02 m (genuine floor contact).

Cross-checked against the pre-retargeting SMPL-X reconstruction (independent of
GMR): the same left-lower/right-higher bias is already present upstream, at a
few cm, with real alternation (range 0.11-0.41 m both feet). So the DIRECTION of
the asymmetry is not introduced by retargeting - but its severity is: a human
walk with a slightly short right step is still a walk; a robot whose right foot
never touches the ground for 4 s is not. Something amplifies a small upstream
bias into an outright non-contact, and that amplification is what this fixes,
independent of settling exactly which stage is responsible for the original cm
of bias.

THE FIX
-------
The right leg's OWN height trajectory already has local minima - the moments
where its (currently insufficient) swing brings it closest to the floor. These
are the natural footfall instants; a real gait would have the foot fully
grounded there. At each one, extend the leg (hip pitch/roll/yaw, knee, ankle
pitch/roll only - never touching the arms, the grasp, or the other leg) via a
damped Jacobian correction until the sole reaches the floor exactly, then blend
the correction back to zero toward the neighbouring swing frames. This is the
same equality-not-inequality mechanism used successfully for
stage3_reground_chair.py, generalised from one contact per motion to one per
stance window.

Deliberately NOT touching:
  - the root path (preserves the walk's forward progress and the arms' grasp
    on the chair, which the root position feeds into)
  - the left leg (already plants correctly)
  - anything outside the identified stance windows (preserves the swing)

Usage:
    python data/stage3_fix_foot_planting.py <motion_lib_dir> --side right
                                             [--start 0] [--end 97] [--write]
"""
import argparse, glob, importlib.util, os
import numpy as np, joblib, mujoco
from scipy.ndimage import minimum_filter1d
from scipy.spatial.transform import Rotation as R

_spec = importlib.util.spec_from_file_location(
    "feas", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "stage3_mujoco_feasibility.py"))
_feas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feas)

LEG_JOINTS = {
    "left": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
             "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
    "right": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
              "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
}


def sole_z(m, d, side):
    gs = [g for g in range(m.ngeom) if m.geom_contype[g]
          and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
          == f"{side}_ankle_roll_link"]
    return min(d.geom_xpos[g][2] - m.geom_rbound[g] for g in gs)


def find_footfalls(heights, win=9, min_sep=6, max_height=0.15):
    """Local minima of the (still-elevated) foot trajectory: the natural
    footfall instants, wherever the current swing happens to bottom out.

    `max_height` is load-bearing, not cosmetic. A relative local minimum can
    occur anywhere - including in the middle of a mostly-elevated region where
    the foot is nowhere close to landing (measured: 0.297-0.378 m through
    f188-200 of the basket step-down, a broad, barely-dipping plateau, not a
    step). Driving THAT frame's sole to exactly z=0 asked the IK to travel too
    far and it found a degenerate solution - knee 1.506 -> 0.071 rad (fully
    unbent) and hip yaw swung 1.3 rad to compensate, an anatomically impossible
    correction for one frame. Only a candidate already within `max_height` of
    the floor is treated as a real, physically small footfall correction.
    """
    env = minimum_filter1d(heights, win, mode="nearest")
    is_min = (heights <= env + 1e-6) & (heights <= max_height)
    idx = np.where(is_min)[0]
    if len(idx) == 0:
        return []
    # collapse runs to their centre, and merge instants closer than min_sep
    groups, cur = [], [idx[0]] if len(idx) else []
    for i in idx[1:]:
        if i - cur[-1] <= min_sep:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    if cur:
        groups.append(cur)
    return [int(np.mean(g)) for g in groups]


def plant_frame(m, d, qpos, joint_ids, tol=0.001, iters=60, lam=1e-2, max_step=0.05):
    """Drive this frame's sole to z=0 exactly, moving only the given joints."""
    q = qpos.copy().astype(np.float64)   # mj_integratePos requires float64
    jacp = np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        gs = [g for g in range(m.ngeom) if m.geom_contype[g] and g in _sole_geoms]
        low_g = min(gs, key=lambda g: d.geom_xpos[g][2] - m.geom_rbound[g])
        z = d.geom_xpos[low_g][2] - m.geom_rbound[low_g]
        if abs(z) <= tol:
            break
        mujoco.mj_jac(m, d, jacp, None, d.geom_xpos[low_g], m.geom_bodyid[low_g])
        col = jacp[2, joint_ids]                      # d(sole_z)/d(these joints only)
        dq_sub = col / (col @ col + lam) * (-z)
        dq = np.zeros(m.nv); dq[joint_ids] = np.clip(dq_sub, -max_step, max_step)
        mujoco.mj_integratePos(m, q, dq, 1.0)
        lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
        q[7:] = np.clip(q[7:], lo, hi)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion_dir")
    ap.add_argument("--side", choices=["left", "right"], default="right")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=97)
    ap.add_argument("--blend", type=int, default=5, help="frames to ease in/out")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rp = glob.glob(f"{a.motion_dir}/robot/*.pkl")[0]
    data = joblib.load(rp); key = list(data)[0]; M = data[key]
    rt, rr, dof = (np.array(M["root_trans_offset"]), np.array(M["root_rot"]),
                   np.array(M["dof"]))
    qpos = np.concatenate([rt, rr, dof], axis=1).astype(np.float64)

    m = mujoco.MjModel.from_xml_path(_feas.G1_MJCF)
    d = mujoco.MjData(m)
    global _sole_geoms
    _sole_geoms = {g for g in range(m.ngeom) if m.geom_contype[g]
                   and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
                   == f"{a.side}_ankle_roll_link"}
    jnames = LEG_JOINTS[a.side]
    jids = [m.joint(n).dofadr[0] for n in jnames]

    before = np.array([sole_z(m, d, a.side) if _set_q(m, d, qpos[f]) or True else 0
                       for f in range(len(qpos))]) if False else None
    heights = np.zeros(len(qpos))
    for f in range(len(qpos)):
        d.qpos[:] = qpos[f]; mujoco.mj_forward(m, d)
        heights[f] = sole_z(m, d, a.side)

    win = slice(a.start, a.end)
    falls = [a.start + i for i in find_footfalls(heights[win])]
    print(f"\n=== foot planting fix: {a.motion_dir}  side={a.side}  window f{a.start}-{a.end} ===")
    print(f"before: min {heights[win].min():.4f}  max {heights[win].max():.4f}"
          f"  (never within 2 cm on {int((heights[win] < 0.02).sum())}/{a.end-a.start} frames)")
    print(f"detected footfall instants: {falls}")

    out = qpos.copy()
    corrections = {}
    for f in falls:
        planted = plant_frame(m, d, qpos[f], jids)
        corrections[f] = planted[7:] - qpos[f][7:]   # dof-space delta only

    # blend each correction into its neighbourhood, tapering to 0 at --blend frames out
    dq_total = np.zeros((len(qpos), m.nv - 6))
    for f, delta in corrections.items():
        for df in range(-a.blend, a.blend + 1):
            g = f + df
            if a.start <= g < a.end:
                w = 1.0 - abs(df) / (a.blend + 1)
                dq_total[g] = np.maximum(dq_total[g], w * np.abs(delta)) * np.sign(delta + 1e-12)
    out[:, 7:] += dq_total
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    out[:, 7:] = np.clip(out[:, 7:], lo, hi)

    after = np.zeros(len(qpos))
    for f in range(len(qpos)):
        d.qpos[:] = out[f]; mujoco.mj_forward(m, d)
        after[f] = sole_z(m, d, a.side)
    print(f"after:  min {after[win].min():.4f}  max {after[win].max():.4f}"
          f"  (within 2 cm on {int((after[win] < 0.02).sum())}/{a.end-a.start} frames)")
    print(f"max joint change (this leg only): {np.abs(dq_total[win]).max():.3f} rad")
    print(f"swing shape preserved outside stance (max height unchanged: "
          f"{heights[win].max():.3f} -> {after[win].max():.3f})")

    if a.write:
        M["root_trans_offset"] = out[:, :3].astype(np.float32)
        M["dof"] = out[:, 7:].astype(np.float32)
        pose_aa = np.array(M["pose_aa"]).copy()
        axes = np.array([m.jnt_axis[j] for j in range(1, m.njnt)])
        pose_aa[:, 1:] = out[:, 7:, None] * axes[None]
        M["pose_aa"] = pose_aa.astype(np.float32)
        joblib.dump({key: M}, rp)
        print(f"written {rp}")


def _set_q(m, d, q):
    d.qpos[:] = q; mujoco.mj_forward(m, d); return True


if __name__ == "__main__":
    main()
