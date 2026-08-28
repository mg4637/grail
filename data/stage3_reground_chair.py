#!/usr/bin/env python
"""Sit the placed chair on the floor, and drop the robot with it.

THE DEFECT
----------
FoundationPose's chair drifts upward through the clip. Measured as the lowest
point of the posed mesh:

    at rest, start   (f0-12)     +0.030 m
    carried          (f20-60)     0.19 -> 0.37 m   (held, so unconstrained)
    at rest, placed  (f104-239)  +0.086 .. +0.112 m

so after placement the chair hovers ~9.5 cm. The robot does NOT share this
error - its soles sit at -0.007..+0.003 on the floor at the start and
+0.021..+0.029 at the end - so the two are inconsistent with each other.

It matters in `__carry`, where the chair is a DYNAMIC body: physics drops it
9.5 cm the moment it is released, which is 95% of the 0.1 m object_pos_deviation
termination threshold on its own, and leaves the step-up reaching for a seat
that is no longer there. (In `__basket` the chair is baked into the static USD,
so a hovering chair is only cosmetic - kinematic bodies do not fall.)

THE FIX, AND WHY IT MOVES THE ROBOT TOO
---------------------------------------
Lowering the chair alone would leave the robot's feet 9.5 cm above the seat it
is standing on. The robot is standing on the chair in reality, so if the seat is
really 9.5 cm lower, so is the robot. Both are corrected together.

The correction starts at f96, AFTER both hands have released (the grasp signal
has the right hand closed to f93, the left to f91), so it cannot disturb the
carry. The robot's share is scaled by how high its soles are:

    w(f) = clamp((min_sole_z - 0.05) / 0.15, 0, 1)

which is 0 with the feet on the floor and 1 once they are up on the chair, so
the walk-up and the final step-down are untouched and only the on-chair phase
moves. During the step-up and step-down one foot is on the chair and one is not,
and no rigid shift can satisfy both - that residual is reported below.

Usage:
    python data/stage3_reground_chair.py <motion_lib_dir> [--write]
"""
import argparse, glob, importlib.util, os
import numpy as np, joblib, mujoco, trimesh
from scipy.spatial.transform import Rotation as R

_spec = importlib.util.spec_from_file_location(
    "feas", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "stage3_mujoco_feasibility.py"))
_feas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion_dir")
    ap.add_argument("--from_frame", type=int, default=96, help="first corrected frame")
    ap.add_argument("--ramp", type=int, default=6, help="frames to ease the shift in")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rp = glob.glob(f"{a.motion_dir}/robot/*.pkl")[0]
    rdata = joblib.load(rp); rkey = list(rdata)[0]; M = rdata[rkey]
    op_ = glob.glob(f"{a.motion_dir}/objects/*.pkl")[0]
    odata = joblib.load(op_); okey = list(odata)[0]; O = odata[okey]

    rt = np.array(M["root_trans_offset"], dtype=float)
    rr, dof = np.array(M["root_rot"]), np.array(M["dof"])
    op = np.array(O["root_pos"], dtype=float)
    oq = np.array(O["root_quat"])
    T = len(dof)

    cm = trimesh.load("results/generation/mesh/ComAsset/chair/model.obj")
    if isinstance(cm, trimesh.Scene):
        cm = cm.dump(concatenate=True)
    cv = np.asarray(cm.vertices) * np.asarray(O["scale"]).reshape(-1)[:3]

    m = mujoco.MjModel.from_xml_path(_feas.G1_MJCF)
    d = mujoco.MjData(m)
    foot = [g for g in range(m.ngeom) if m.geom_contype[g]
            and "ankle_roll" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                                   m.geom_bodyid[g]) or "")]

    def chair_low(f, dz=0.0):
        w = cv @ R.from_quat(oq[f, 0][[1, 2, 3, 0]]).as_matrix().T + op[f, 0]
        return w[:, 2].min() + dz

    def sole_low(f, dz=0.0):
        d.qpos[0:3], d.qpos[3:7], d.qpos[7:36] = rt[f] + [0, 0, dz], rr[f], dof[f]
        mujoco.mj_forward(m, d)
        return min(d.geom_xpos[g][2] - m.geom_rbound[g] for g in foot)

    rest = np.array([chair_low(f) for f in range(max(a.from_frame + a.ramp, T - 60), T)])
    drop = float(np.median(rest))
    soles = np.array([sole_low(f) for f in range(T)])
    w_up = np.clip((soles - 0.05) / 0.15, 0.0, 1.0)

    ramp = np.zeros(T)
    for f in range(T):
        if f >= a.from_frame + a.ramp:
            ramp[f] = 1.0
        elif f > a.from_frame:
            ramp[f] = (f - a.from_frame) / float(a.ramp)

    obj_dz = -drop * ramp
    rob_dz = -drop * ramp * w_up

    print(f"\n=== reground: {a.motion_dir} ===")
    print(f"  placed chair hovers {100*drop:.1f} cm (median of the resting frames)")
    print(f"  correction ramps in over f{a.from_frame}-{a.from_frame + a.ramp}, "
          f"after both hands release")
    print(f"  robot shifted by up to {100*abs(rob_dz).max():.1f} cm while its feet are up")

    op2 = op.copy(); op2[:, 0, 2] += obj_dz
    rt2 = rt.copy(); rt2[:, 2] += rob_dz

    def report(tag, opx, rtx):
        lo = [chair_low(f, opx[f, 0, 2] - op[f, 0, 2]) for f in range(T)]
        rest_lo = [lo[f] for f in range(a.from_frame + a.ramp, T)]
        s0 = [sole_low(f, rtx[f, 2] - rt[f, 2]) for f in range(0, 10)]
        s1 = [sole_low(f, rtx[f, 2] - rt[f, 2]) for f in range(T - 15, T)]
        # foot-to-seat gap while standing on the chair
        gaps = []
        for f in range(a.from_frame + a.ramp + 20, T - 50):
            gaps.append(sole_low(f, rtx[f, 2] - rt[f, 2]) - (lo[f] + 0.395))
        print(f"  {tag:<7} placed chair z {np.median(rest_lo):+.3f} | "
              f"soles floor start {np.min(s0):+.3f} end {np.min(s1):+.3f} | "
              f"foot-vs-seat drift {np.std(gaps):.3f}")

    report("before", op, rt)
    report("after", op2, rt2)

    if a.write:
        O["root_pos"] = op2.astype(np.float32)
        M["root_trans_offset"] = rt2.astype(np.float32)
        joblib.dump({okey: O}, op_)
        joblib.dump({rkey: M}, rp)
        print(f"  written {rp} and {op_}")


if __name__ == "__main__":
    main()
