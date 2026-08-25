#!/usr/bin/env python
"""Is the reference motion executable by a G1 at all, given unilateral contact?

This is the question every other check dodged. `--pin_base` feasibility says the
LIMBS can hit the poses, but an external constraint holds the pelvis on the
reference path, so the pin silently does the work of lifting the robot. The real
test is whether the contacts the motion actually has can produce the wrench the
motion actually needs.

For a floating-base robot the first 6 rows of the equations of motion are
unactuated - no motor appears in them:

    [M(q) qddot + h(q, qdot)]_{0:6}  =  sum_i (J_i^T f_i)_{0:6}

The left side is fixed by the reference kinematics. The right side is whatever
the contacts can supply, subject to physics: a contact can PUSH but never PULL
(f_n >= 0), and tangential force is bounded by friction (|f_t| <= mu f_n,
linearised here as a pyramid). If no admissible f solves those 6 equations, the
motion is impossible - not hard, impossible - and no reward tuning, curriculum
or extra training will ever produce it.

Reported per frame:
  residual  - norm of the base wrench that no admissible contact force can
              supply. 0 means feasible. Nonzero means the reference asks for
              force out of thin air.
  mu_req    - smallest friction coefficient that makes the frame feasible
              (inf if it is infeasible at any friction).

Usage: python data/stage3_contact_feasibility.py <motion_lib_dir> [--mu 1.0]
"""
import argparse, glob, os, importlib.util
import numpy as np, joblib, mujoco
from scipy.optimize import linprog

_spec = importlib.util.spec_from_file_location(
    "feas", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "stage3_mujoco_feasibility.py"))
_feas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feas)

CONTACT_TOL = 0.010          # overridden by --tol


def contact_set(m, d, scene_bodies, tol=CONTACT_TOL):
    """(point, normal, body) for every robot geom at/near the scene."""
    out = []
    for i in range(m.ngeom):
        if not m.geom_contype[i] or m.geom_bodyid[i] in scene_bodies:
            continue
        for j in range(m.ngeom):
            if not m.geom_contype[j] or m.geom_bodyid[j] not in scene_bodies:
                continue
            ft = np.zeros(6)
            dist = mujoco.mj_geomDistance(m, d, i, j, 0.05, ft)
            if dist < tol:
                n = ft[3:] - ft[:3]
                nn = np.linalg.norm(n)
                if nn < 1e-9:
                    n = np.array([0.0, 0.0, 1.0])
                else:
                    n = n / nn
                if n[2] < 0:
                    n = -n                      # point away from the surface
                out.append((0.5 * (ft[:3] + ft[3:]), n, m.geom_bodyid[i]))
    return out


def base_wrench_residual(m, d, contacts, rhs, mu):
    """Min ||rhs - sum (J_i^T f_i)_{0:6}|| over admissible contact forces."""
    if not contacts:
        return np.linalg.norm(rhs)
    cols = []
    jacp = np.zeros((3, m.nv))
    for p, n, bid in contacts:
        # friction pyramid basis: normal plus 4 tangential edges
        t1 = np.cross(n, [0, 0, 1.0])
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.cross(n, [1.0, 0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        mujoco.mj_jac(m, d, jacp, None, p, bid)
        for dirv in (n + mu * t1, n - mu * t1, n + mu * t2, n - mu * t2):
            cols.append((jacp.T @ dirv)[:6])
    A = np.array(cols).T                        # 6 x (4*nc), all coeffs >= 0
    n_var = A.shape[1]
    # minimise L1 slack: A x + s+ - s- = rhs, x >= 0
    c = np.concatenate([np.zeros(n_var), np.ones(12)])
    Aeq = np.hstack([A, np.eye(6), -np.eye(6)])
    res = linprog(c, A_eq=Aeq, b_eq=rhs, bounds=[(0, None)] * (n_var + 12),
                  method="highs")
    return float(res.fun) if res.success else np.linalg.norm(rhs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion_dir")
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.010,
                    help="a robot geom within this distance of the scene counts "
                         "as load-bearing. Too tight and a flat foot contributes "
                         "only 2 of its 8 contact spheres, which cannot generate "
                         "the moment a real foot does - that shows up as phantom "
                         "infeasibility on frames that are obviously fine.")
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="Gaussian sigma (frames) applied to qvel before "
                         "differentiating; 0 disables")
    ap.add_argument("--severe", type=float, default=100.0,
                    help="residual (N) above which a frame is counted severe")
    ap.add_argument("--static", action="store_true",
                    help="zero the accelerations and velocities: tests whether an "
                         "infinitely slow version of the motion would be feasible. "
                         "If a frame fails here, retiming cannot rescue it - the "
                         "geometry of the contacts simply cannot hold the robot up.")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    a = ap.parse_args()

    M = list(joblib.load(glob.glob(f"{a.motion_dir}/robot/*.pkl")[0]).values())[0]
    rt, rr, dof = (np.array(M["root_trans_offset"]), np.array(M["root_rot"]),
                   np.array(M["dof"]))
    fps = float(M["fps"])
    m = mujoco.MjModel.from_xml_path(
        _feas.build_scene(glob.glob(f"{a.motion_dir}/objects/*.pkl")[0]))
    d = mujoco.MjData(m)
    scene = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("world", "table")}
    scene.discard(-1)
    # the table is a static prop in this scene; exclude its mass from the robot
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
    robot_mass = m.body_mass[[b for b in range(1, m.nbody) if b != tb]].sum()

    smooth_sigma = a.smooth
    T = len(dof)
    qpos = np.zeros((T, m.nq))
    for f in range(T):
        qpos[f] = np.concatenate([rt[f], rr[f], dof[f]])

    # velocities/accelerations in MuJoCo's tangent space
    qvel = np.zeros((T, m.nv))
    for f in range(T - 1):
        mujoco.mj_differentiatePos(m, qvel[f], 1.0 / fps, qpos[f], qpos[f + 1])
    qvel[-1] = qvel[-2]
    # Differentiating 24 fps kinematics twice amplifies noise into phantom
    # accelerations, which show up as small phantom infeasibility. Smooth first
    # so the severity numbers mean something.
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        qvel = gaussian_filter1d(qvel, smooth_sigma, axis=0, mode="nearest")
    qacc = np.gradient(qvel, 1.0 / fps, axis=0)

    lo, hi = max(a.start, 1), min(a.end, T - 1)
    print(f"\n=== {a.motion_dir}  contact-force feasibility (mu={a.mu}) ===")
    print(f"robot mass {robot_mass:.1f} kg,  weight {robot_mass*9.81:.0f} N\n")
    print(f"{'f':>4} {'nc':>3} {'contact bodies':<34} {'|rhs| N':>8} {'residual N':>11} {'':>4}")
    bad = []
    for f in range(lo, hi):
        d.qpos[:] = qpos[f]
        d.qvel[:] = 0.0 if a.static else qvel[f]
        mujoco.mj_forward(m, d)
        cs = contact_set(m, d, scene, a.tol)
        # qacc must be written AFTER mj_forward - mj_forward runs forward
        # dynamics and overwrites it, so setting it first silently substitutes
        # MuJoCo's own acceleration for the reference's.
        d.qacc[:] = 0.0 if a.static else qacc[f]
        # M(q) qddot + h(q, qdot), WITHOUT contact forces - what the contacts
        # must supply. Two traps here, both of which silently produce a
        # meaningless answer:
        #   * mj_inverse subtracts the constraint forces MuJoCo already resolved,
        #     so qfrc_inverse[:6] comes back ~0 and every frame looks feasible.
        #   * mj_rne does NOT include gravity - for a robot standing still it
        #     returns exactly zero, so the test would never ask the contacts to
        #     carry the 327 N the robot actually weighs.
        # qfrc_bias is the term that carries both Coriolis AND gravity
        # (verified: [0,0,327.1,...] for a static G1), so M*qacc + qfrc_bias.
        # mj_inverse with the constraint solver DISABLED: qfrc_inverse is then
        # exactly M(q) qacc + bias, with no contact force subtracted.
        saved = m.opt.disableflags
        m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
                               | mujoco.mjtDisableBit.mjDSBL_CONTACT)
        mujoco.mj_inverse(m, d)
        m.opt.disableflags = saved
        rhs = d.qfrc_inverse[:6].copy()
        r = base_wrench_residual(m, d, cs, rhs, a.mu)
        labs = sorted({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                       .replace("_link", "").replace("left_", "L.").replace("right_", "R.")
                       for _, _, b in cs})
        flag = "SEVERE" if r > a.severe else ("marginal" if r > 1.0 else "")
        if r > 1.0:
            bad.append((f, r))
        print(f"{f:4d} {len(cs):3d} {','.join(labs)[:33]:<34} "
              f"{np.linalg.norm(rhs):8.1f} {r:11.1f} {flag:>4}")
    sev = [f for f, r in bad if r > a.severe]
    print(f"\ninfeasible {len(bad)}/{hi-lo}   SEVERE (>{a.severe:.0f} N) {len(sev)}"
          + (f"  frames {sev}" if sev else ""))


if __name__ == "__main__":
    main()
