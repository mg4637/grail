#!/usr/bin/env python
"""Push a retargeted motion out of the scene it was retargeted through.

WHY
---
GMR retargets against the robot MJCF alone - the table is not in the model, and
`mink` is given only a ConfigurationLimit. Nothing in Stage 3 knows the scene
exists, so the IK happily places the hands inside the tabletop and the feet
below the floor. Measured on the reference, by pose alone with no dynamics:

    grail_table      116/121 frames penetrate >5 mm, worst 73 mm (right knee)
    grail_table_v2   121/121 frames penetrate >5 mm, worst 72 mm (left ankle)

That is not cosmetic. SONIC's terminations measure exactly these bodies -
`ee_body_pos` at 0.15 m and `foot_pos_xyz` at 0.20 m - against targets the robot
physically cannot reach, because they are inside a rigid body. A 70 mm offset
spends 35-47% of the termination budget before the policy has made a single
mistake, and on the tabletop it asks the foot to stand *below* the surface,
which the contact solver answers by ejecting the foot. That is the slip.

WHAT THIS DOES
--------------
Per frame, a damped-least-squares projection along the contact normals:

    dq = J^T (J J^T + lambda I)^-1 d       J[i] = d(separation_i)/dq

iterated until the deepest penetration is under `--tol`. The root is free to
move, so the usual answer to "both feet are 5 cm under the floor" is to lift the
whole robot 5 cm rather than to bend it. Joint angles are clamped to the MJCF
ranges afterwards. The correction is the minimum that reaches touching contact,
so surfaces the motion is *meant* to rest on stay rested on - this removes
interpenetration, it does not lift the robot off the table.

Two things keep the result temporally coherent, because SONIC tracks root
velocity as well as root position and per-frame independent solving wrecks both:

  * each frame is warm-started from the previous frame's correction, accepted
    only if it is already an improvement. Near a contact the push-out has many
    near-equivalent solutions and independent solves flip between them; seeding
    with the neighbour's answer picks the continuous one.
  * between passes the *correction* is low-pass filtered in time and the
    residual re-solved. Filtering the correction rather than the motion leaves
    whatever dynamics the reference already had untouched; only the push-out
    is smoothed.

Measured on grail_table_v2, max |3rd difference| of the root: 0.0097 m in the
reference, 0.4178 m solving frames independently, 0.015 m with both guards - and
the largest per-frame root step stays at the reference's own 47-49 mm, so no
velocity spikes are introduced.

`pose_aa` is rewritten from the corrected `dof` using each joint's own axis, so
the two stay consistent; SONIC's fk_batch reads `pose_aa`.

Usage:
    python data/stage3_resolve_penetration.py <src_motion_lib> <dst_motion_lib>
    python data/stage3_resolve_penetration.py <src> --dry-run
"""
import argparse, glob, os, shutil, sys
import numpy as np, joblib, mujoco
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation as R
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "feas", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "stage3_mujoco_feasibility.py"))
_feas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feas)


def penetrations(m, d):
    """(depth, normal_jacobian_row) for every robot-vs-scene contact that overlaps."""
    body_of = m.geom_bodyid
    scene = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("world", "table")}
    scene.discard(-1)
    rows, depths, jac1, jac2 = [], [], np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist >= 0:
            continue
        b1, b2 = body_of[c.geom1], body_of[c.geom2]
        if (b1 in scene) == (b2 in scene):      # robot-robot or scene-scene
            continue
        n = c.frame[:3].copy()                  # points geom1 -> geom2
        mujoco.mj_jac(m, d, jac1, None, c.pos, b1)
        mujoco.mj_jac(m, d, jac2, None, c.pos, b2)
        rows.append(n @ (jac2 - jac1))          # d(separation)/dq
        depths.append(-c.dist)
    return np.array(depths), (np.array(rows) if rows else np.zeros((0, m.nv)))


def _worst(m, d, q):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    depth, J = penetrations(m, d)
    return (depth.max() if len(depth) else 0.0), depth, J


def carry_correction(q_ref, q_src, q_src0):
    """Re-apply the correction (q_src - q_src0) on top of a different pose."""
    o = q_ref.copy()
    o[:3] += q_src[:3] - q_src0[:3]
    o[7:] += q_src[7:] - q_src0[7:]
    R0 = R.from_quat(q_src0[[4, 5, 6, 3]])
    Rc = R.from_quat(q_src[[4, 5, 6, 3]])
    Rr = R.from_quat(q_ref[[4, 5, 6, 3]])
    o[3:7] = (Rr * (R0.inv() * Rc)).as_quat()[[3, 0, 1, 2]]
    return o


def retract(m, d, q, q0, tol, steps=14, window=5):
    """Shrink a correction to the smallest one that still clears the scene.

    Three passes of push-out, each warm-started and re-solved after smoothing,
    accumulate overshoot: measured on grail_table_v3 the root ends up 94 mm
    higher on average to clear ~25 mm of penetration, which lifts the robot off
    the floor it is supposed to be standing on. Bisecting the scale of the whole
    correction back toward the reference recovers the minimal fix - the motion
    keeps only the displacement it actually needed.

    The per-frame scale is then run through a MAXIMUM filter, not a Gaussian.
    Bisection is noisy frame to frame (root jerk 0.046 -> 0.222, per-frame root
    step 55 -> 127 mm), but a low-pass would pull some frames back below the
    scale they need and undo the fix. A running max only ever raises a scale, so
    every frame still clears, and the down-spikes that caused the jitter go away.
    """
    scale = np.ones(len(q))
    for f in range(len(q)):
        if _worst(m, d, q0[f])[0] <= tol:
            scale[f] = 0.0
            continue
        lo, hi = 0.0, 1.0                       # hi is known to clear
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            if _worst(m, d, _blend(q0[f], q[f], mid))[0] <= tol:
                hi = mid
            else:
                lo = mid
        scale[f] = hi
    scale = maximum_filter1d(scale, size=window, mode="nearest")
    return np.array([_blend(q0[f], q[f], scale[f]) for f in range(len(q))])


def _blend(qa, qb, t):
    """Interpolate a pose t of the way from qa to qb (slerp on the root quat)."""
    o = qa.copy()
    o[:3] = qa[:3] + t * (qb[:3] - qa[:3])
    o[7:] = qa[7:] + t * (qb[7:] - qa[7:])
    Ra, Rb = R.from_quat(qa[[4, 5, 6, 3]]), R.from_quat(qb[[4, 5, 6, 3]])
    o[3:7] = (Ra * R.from_rotvec(t * (Ra.inv() * Rb).as_rotvec())).as_quat()[[3, 0, 1, 2]]
    return o


def smooth_correction(q, q0, sigma):
    """Low-pass the per-frame correction (q - q0) in time, orientation included."""
    if sigma <= 0:
        return q
    out = q.copy()
    out[:, :3] = q0[:, :3] + gaussian_filter1d(q[:, :3] - q0[:, :3], sigma, axis=0, mode="nearest")
    out[:, 7:] = q0[:, 7:] + gaussian_filter1d(q[:, 7:] - q0[:, 7:], sigma, axis=0, mode="nearest")
    # rotation correction as a rotvec in the original frame, so it filters linearly
    R0 = R.from_quat(q0[:, [4, 5, 6, 3]])
    Rc = R.from_quat(q[:, [4, 5, 6, 3]])
    rv = gaussian_filter1d((R0.inv() * Rc).as_rotvec(), sigma, axis=0, mode="nearest")
    out[:, 3:7] = (R0 * R.from_rotvec(rv)).as_quat()[:, [3, 0, 1, 2]]
    return out


def resolve(m, d, qpos, tol, iters=60, lam=5e-2, max_step=0.02):
    """Least-norm push-out, with a step cap and a backtracking line search.

    Both guards matter. Mesh contacts arrive in clusters whose normals are nearly
    parallel, so J J^T is close to singular and an under-damped solve answers a
    22 mm overlap with a 380 mm lateral slide of the root - measured, on frame
    102 - which un-penetrates while destroying the motion. lam damps that;
    max_step bounds how far one iteration may travel; and the line search
    rejects any step that does not actually reduce the deepest penetration,
    since new contacts appear as the robot moves.
    """
    q = np.array(qpos, dtype=float)
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]   # skip the free joint
    worst0, depth, J = _worst(m, d, q)
    worst = worst0
    for _ in range(iters):
        if not len(depth) or worst <= tol:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(len(depth)), depth)
        n = np.linalg.norm(dq)
        if n > max_step:
            dq *= max_step / n
        for scale in (1.0, 0.5, 0.25, 0.1):
            trial = q.copy()
            mujoco.mj_integratePos(m, trial, dq * scale, 1.0)
            trial[7:] = np.clip(trial[7:], lo, hi)
            w, dep, Jn = _worst(m, d, trial)
            if w < worst - 1e-6:
                q, worst, depth, J = trial, w, dep, Jn
                break
        else:
            break                                  # no accepted step; give up
    return q, worst0, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst", nargs="?")
    ap.add_argument("--tol", type=float, default=0.002, help="allowed penetration (m)")
    ap.add_argument("--max_step", type=float, default=0.02,
                    help="cap on |dq| per iteration (m / rad)")
    ap.add_argument("--lam", type=float, default=1e-2, help="DLS damping")
    ap.add_argument("--iters", type=int, default=200, help="solver iterations per frame")
    ap.add_argument("--sigma", type=float, default=1.5,
                    help="Gaussian sigma, in frames, for smoothing the correction")
    ap.add_argument("--passes", type=int, default=3,
                    help="resolve/smooth rounds")
    ap.add_argument("--window", type=int, default=5,
                    help="max-filter width, in frames, for the retraction scale")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run and not a.dst:
        sys.exit("need a destination motion_lib dir (or --dry-run)")

    robot_pkl = glob.glob(f"{a.src}/robot/*.pkl")[0]
    data = joblib.load(robot_pkl)
    key = list(data)[0]
    M = data[key]
    dof, rt, rr = (np.array(M["dof"]), np.array(M["root_trans_offset"]),
                   np.array(M["root_rot"]))

    m = mujoco.MjModel.from_xml_path(_feas.build_scene(glob.glob(f"{a.src}/objects/*.pkl")[0]))
    d = mujoco.MjData(m)

    # joint axes in the parent frame, for rebuilding pose_aa from dof
    axes = np.array([m.jnt_axis[j] for j in range(1, m.njnt)])

    q0 = np.concatenate([rt, rr, dof], axis=1)
    out = q0.copy()
    # measured on the raw reference: pass 1 warm-starts from a neighbour, so its
    # own "before" is already partly corrected and understates the defect
    before = np.array([_worst(m, d, q0[f])[0] for f in range(len(dof))])
    for p_ in range(a.passes):
        res = np.zeros_like(out); worst = []
        for f in range(len(dof)):
            start = out[f]
            if f > 0:
                cand = carry_correction(out[f], res[f - 1], out[f - 1])
                if _worst(m, d, cand)[0] < _worst(m, d, start)[0]:
                    start = cand
            res[f], b, aft = resolve(m, d, start, a.tol, iters=a.iters,
                                     lam=a.lam, max_step=a.max_step)
            worst.append((b, aft))
        after = np.array([w[1] for w in worst])
        # last pass keeps the raw solve, so the reported numbers are the ones on disk
        # Smooth between passes, never after the last one: filtering the final
        # correction pushes bodies back into the table (measured: 0/121 frames
        # over 5 mm before a trailing smooth, 36/121 after). By the last pass the
        # remaining correction is small, so leaving it unfiltered costs almost
        # nothing in smoothness.
        out = res if p_ == a.passes - 1 else smooth_correction(res, q0, a.sigma)
        print(f"  pass {p_+1}/{a.passes}: >5 mm {int((after > 0.005).sum()):3d}/{len(dof)}"
              f"  worst {1000*after.max():5.1f} mm  mean {1000*after.mean():4.1f} mm")
    out = retract(m, d, out, q0, a.tol, window=a.window)
    d.qpos[:] = out[0]
    fin = []
    for f in range(len(dof)):
        w, _, _ = _worst(m, d, out[f]); fin.append(w)
    after = np.array(fin)

    jerk = lambda x: np.abs(np.diff(x, 3, axis=0)).max()

    moved_root = np.linalg.norm(out[:, :3] - rt, axis=1)
    moved_dof = np.abs(out[:, 7:] - dof).max(1)
    print(f"\n=== penetration resolution: {a.src} ===")
    print(f"  frames penetrating >5 mm   {int((before > 0.005).sum())}/{len(dof)}"
          f"  ->  {int((after > 0.005).sum())}/{len(dof)}")
    print(f"  worst penetration          {1000*before.max():6.1f} mm"
          f"  ->  {1000*after.max():6.1f} mm")
    print(f"  mean penetration           {1000*before.mean():6.1f} mm"
          f"  ->  {1000*after.mean():6.1f} mm")
    print(f"  cost: root moved  max {1000*moved_root.max():5.1f} mm"
          f"  mean {1000*moved_root.mean():5.1f} mm")
    print(f"        joints moved max {moved_dof.max():5.3f} rad"
          f"  mean {moved_dof.mean():5.3f} rad")
    step = lambda x: np.linalg.norm(np.diff(x, axis=0), axis=1).max()
    print(f"  smoothness (max |3rd difference|): root {jerk(rt):.4f} -> {jerk(out[:, :3]):.4f} m"
          f" | dof {jerk(dof):.4f} -> {jerk(out[:, 7:]):.4f} rad")
    print(f"  max per-frame root step    {1000*step(rt):5.1f} mm"
          f"  ->  {1000*step(out[:, :3]):5.1f} mm")

    # The opposite failure to penetration: having pushed the robot out, is it now
    # hovering? d.ncon cannot answer this - MuJoCo only creates a contact once
    # geoms overlap, so a foot resting exactly on the floor reports nothing at
    # all. mj_geomDistance does report a gap for separated geoms. Only its
    # POSITIVE side is trustworthy: in deep mesh-mesh overlap it returns
    # nonsense (-386.8 mm for a knee inside a 175 mm plank, where the contact
    # solver says 18.5 mm), so penetration is measured from contacts above and
    # this block only asks how far clear the robot is.
    scene_b = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("world", "table")}
    scene_b.discard(-1)
    sg = [g for g in range(m.ngeom) if m.geom_bodyid[g] in scene_b and m.geom_contype[g]]
    rg = [g for g in range(m.ngeom) if m.geom_bodyid[g] not in scene_b and m.geom_contype[g]]
    def gaps(q):
        g = np.empty(len(q))
        for f in range(len(q)):
            d.qpos[:] = q[f]
            mujoco.mj_forward(m, d)
            g[f] = min(mujoco.mj_geomDistance(m, d, i, j, 0.5, None)
                       for i in rg for j in sg)
        return g
    g0, g1 = gaps(q0), gaps(out)
    print(f"  max clearance from scene   {1000*max(g0.max(), 0):5.1f} mm"
          f"  ->  {1000*max(g1.max(), 0):5.1f} mm")
    print(f"  frames hovering >10 mm clear of everything: "
          f"{int((g0 > 0.010).sum())} -> {int((g1 > 0.010).sum())}")

    if a.dry_run:
        return

    M["root_trans_offset"] = out[:, :3].astype(np.float32)
    M["root_rot"] = out[:, 3:7].astype(np.float32)
    M["dof"] = out[:, 7:].astype(np.float32)
    # pose_aa[:, 0] is the root rotation as a rotvec, and every hinge after it is
    # angle * its own axis - verified against the untouched pkl at 0.0 and 8e-8.
    # Both halves have to be rewritten: correcting root_rot alone leaves pose_aa,
    # which is what SONIC's fk_batch actually reads, 0.04 rad out of step.
    pose_aa = np.array(M["pose_aa"]).copy()
    pose_aa[:, 0] = R.from_quat(out[:, [4, 5, 6, 3]]).as_rotvec()
    pose_aa[:, 1:] = out[:, 7:, None] * axes[None]
    M["pose_aa"] = pose_aa.astype(np.float32)

    os.makedirs(a.dst, exist_ok=True)
    for sub in ("objects", "object_usd", "meta", "common"):
        s = os.path.join(a.src, sub)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(a.dst, sub), dirs_exist_ok=True)
    os.makedirs(os.path.join(a.dst, "robot"), exist_ok=True)
    joblib.dump({key: M}, os.path.join(a.dst, "robot", os.path.basename(robot_pkl)))
    print(f"  written: {a.dst}")


if __name__ == "__main__":
    main()
