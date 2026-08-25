#!/usr/bin/env python
"""Re-plant a retargeted climb onto contacts a G1 can actually push on.

WHY THIS EXISTS
---------------
data/stage3_contact_feasibility.py shows the retargeted mount is impossible: for
reference frames 60-73 the only contacts are the right hip links, and no
push-only friction-bounded force can supply the required base wrench (residual
up to 546 N even quasi-statically, so retiming cannot rescue it).

Looking at where the bodies actually are explains why. The source is a KID, and
a kid climbs by:
  * hooking FINGERS over the table edge - the left wrist sits 1-11 cm BELOW the
    top surface, out at y=0.60 against the 0.67 edge, i.e. gripping;
  * bracing a hip against the table's vertical side rail, 32 cm below the top;
  * leaving the other hand hovering ~1 cm clear of the surface.

The G1 has no fingers (and Stage 3 runs --zero_out_wrist, so the wrists are
frozen), so it cannot grip an edge. A vertical rail has a horizontal contact
normal and cannot carry 327 N of weight. A hovering hand carries nothing. The
retargeter produced this because GMR matches joint POSITIONS with no scene, no
contact and no dynamics - it has never heard of the table.

WHAT THIS DOES
--------------
Replaces the kid's strategy with the one the robot's geometry actually affords.
The tabletop is 0.905 m, which is 0.18 m BELOW the G1's 1.085 m shoulders, so
the robot can plant both hands flat on top with its shoulders above them - the
arms then act as compression struts instead of levers, which is why the 25 N.m
shoulder limit is not what binds.

Per frame, a damped Gauss-Newton step drives:
  * PLANT   - each hand's closest point to the tabletop, to exactly 0 (an
              EQUALITY, not the >= 0 that stage3_resolve_penetration.py used;
              that inequality is satisfied by a robot floating in mid-air, which
              is what it produced - 21 of 50 mount frames touching nothing);
  * CLEAR   - hips/pelvis/torso pushed off the table to a margin, so the robot
              stops resting on the side rail;
  * KEEP    - everything else regularised toward the retargeted pose, so the
              motion stays the motion.
Warm-started from the previous frame and re-smoothed between passes, as in
stage3_resolve_penetration.py, so the root path stays differentiable.

The gate is stage3_contact_feasibility.py, not this script's own output.
"""
import argparse, glob, os, shutil, importlib.util
import numpy as np, joblib, mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

_spec = importlib.util.spec_from_file_location(
    "feas", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "stage3_mujoco_feasibility.py"))
_feas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feas)

PLANT_BODIES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
CLEAR_BODIES = ["pelvis", "left_hip_yaw_link", "right_hip_yaw_link",
                "left_hip_roll_link", "right_hip_roll_link",
                "left_hip_pitch_link", "right_hip_pitch_link", "torso_link"]


def geom_extent_z(m, d, g):
    """(zmin, zmax) of a geom in world coords.

    geom_rbound is the BOUNDING SPHERE radius, so for a 2 m tabletop plank it is
    ~1 m and `xpos.z + rbound` reports the tabletop at 1.895 m instead of 0.905.
    Mesh geoms have to be measured from their vertices; primitives are fine with
    rbound because they are small spheres here.
    """
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        return float(w[:, 2].min()), float(w[:, 2].max()), w
    r = m.geom_rbound[g]
    return float(d.geom_xpos[g][2] - r), float(d.geom_xpos[g][2] + r), None


def top_surface(m):
    """z of the tabletop and the 2D convex hull of its footprint."""
    from scipy.spatial import ConvexHull
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == tb and m.geom_contype[g]]
    ext = {g: geom_extent_z(m, d, g) for g in gs}
    z = max(e[1] for e in ext.values())
    pts = []
    for g in gs:
        lo_, hi_, w = ext[g]
        if hi_ < z - 0.03 or w is None:
            continue
        pts.append(w[w[:, 2] > z - 0.03][:, :2])
    P = np.vstack(pts)
    return z, P[ConvexHull(P).vertices]


def inward_push(hull, xy, inset):
    """(violation, unit direction) if xy is outside the hull shrunk by `inset`."""
    c = hull.mean(0)
    worst, dirv = 0.0, np.zeros(2)
    for k in range(len(hull)):
        a, b = hull[k], hull[(k + 1) % len(hull)]
        e = b - a
        n = np.array([e[1], -e[0]]); n /= max(np.linalg.norm(n), 1e-9)
        if n @ (c - a) < 0:
            n = -n                                   # inward normal
        sd = n @ (xy - a) - inset                    # >0 means safely inside
        if sd < worst:
            worst, dirv = sd, n
    return -worst, dirv


def scene_geoms(m):
    """Split the table into its TOP surface and everything else."""
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
    world = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "world")
    table = [g for g in range(m.ngeom) if m.geom_bodyid[g] == tb and m.geom_contype[g]]
    floor = [g for g in range(m.ngeom) if m.geom_bodyid[g] == world and m.geom_contype[g]]
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    tops = [g for g in table if d.geom_xpos[g][2] + m.geom_rbound[g] > 0.88]
    return tops, table, floor


def body_geoms(m, name):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    return b, [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]]


def closest(m, d, gs, targets, distmax=0.6):
    """(distance, point-on-robot, unit vector robot->target, body) of the nearest pair."""
    best = (1e9, None, None, None)
    ft = np.zeros(6)
    for i in gs:
        for j in targets:
            v = mujoco.mj_geomDistance(m, d, i, j, distmax, ft)
            if v < best[0]:
                n = ft[3:] - ft[:3]
                nn = np.linalg.norm(n)
                n = n / nn if nn > 1e-9 else np.array([0.0, 0.0, -1.0])
                best = (v, ft[:3].copy(), n, m.geom_bodyid[i])
    return best


def solve_frame(m, d, q, q0, tops, table, floor, plant, clear,
                margin=0.02, iters=80, lam=3e-2, max_step=0.03, w_reg=0.06,
                w_root=(0.6, 0.6, 0.12), w_rot=0.4):
    q = np.array(q, dtype=float)
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    jacp = np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        rows, res = [], []

        if plant:
            top_z, hull = tops
            for bid, gs in plant:
                # lowest point of the hand, and the geom that owns it
                lows = {g: geom_extent_z(m, d, g)[0] for g in gs}
                g = min(lows, key=lows.get)
                p = d.geom_xpos[g].copy()
                drop = p[2] - lows[g]                # centre-to-sole offset
                mujoco.mj_jac(m, d, jacp, None, p, bid)
                # (a) put the sole of the hand ON the surface, not merely near a
                #     plank: distance-to-geom cannot tell "resting on top" from
                #     "touching the side", and pulled hands into the edge.
                rows.append(jacp[2][None])
                res.append(np.array([top_z + drop - p[2]]))
                # (b) keep it over the tabletop
                viol, dirv = inward_push(hull, p[:2], 0.06)
                if viol > 0:
                    rows.append((dirv @ jacp[:2])[None])
                    res.append(np.array([viol]))

        for _, gs in clear:
            dist, p, n, bid = closest(m, d, gs, table + floor)
            if p is None or dist >= margin:
                continue
            mujoco.mj_jac(m, d, jacp, None, p, bid)
            rows.append(-(n @ jacp))
            res.append(margin - dist)             # push out to the margin

        # Stay near the retargeted pose. The ROOT needs its own regulariser: with
        # only the joints held, the cheapest way to put a hand on the table is to
        # translate the whole robot, and the solver duly moved the root 616 mm.
        # z is weighted much softer than x/y because the pelvis genuinely has to
        # rise off the side rail - that is the point of the edit.
        Jr = np.zeros((m.nv - 6, m.nv)); Jr[:, 6:] = np.eye(m.nv - 6)
        rows.append(w_reg * Jr)
        res.append(w_reg * (q0[7:] - q[7:]))

        W = np.diag(w_root)
        Jt = np.zeros((3, m.nv)); Jt[:, :3] = np.eye(3)
        rows.append(W @ Jt)
        res.append(W @ (q0[:3] - q[:3]))

        Jw = np.zeros((3, m.nv)); Jw[:, 3:6] = np.eye(3)
        rows.append(w_rot * Jw)
        dR = (R.from_quat(q[[4, 5, 6, 3]]).inv() * R.from_quat(q0[[4, 5, 6, 3]])).as_rotvec()
        res.append(w_rot * dR)

        A = np.vstack([r if r.ndim == 2 else r[None] for r in rows])
        b = np.concatenate([np.atleast_1d(x) for x in res])
        if np.linalg.norm(b) < 1e-4:
            break
        dq = A.T @ np.linalg.solve(A @ A.T + lam * np.eye(len(b)), b)
        nrm = np.linalg.norm(dq)
        if nrm > max_step:
            dq *= max_step / nrm
        mujoco.mj_integratePos(m, q, dq, 1.0)
        q[7:] = np.clip(q[7:], lo, hi)
    return q


def smooth(q, q0, sigma):
    out = q.copy()
    out[:, :3] = q0[:, :3] + gaussian_filter1d(q[:, :3] - q0[:, :3], sigma, axis=0, mode="nearest")
    out[:, 7:] = q0[:, 7:] + gaussian_filter1d(q[:, 7:] - q0[:, 7:], sigma, axis=0, mode="nearest")
    R0 = R.from_quat(q0[:, [4, 5, 6, 3]]); Rc = R.from_quat(q[:, [4, 5, 6, 3]])
    rv = gaussian_filter1d((R0.inv() * Rc).as_rotvec(), sigma, axis=0, mode="nearest")
    out[:, 3:7] = (R0 * R.from_rotvec(rv)).as_quat()[:, [3, 0, 1, 2]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst", nargs="?")
    ap.add_argument("--plant_from", type=int, default=48)
    ap.add_argument("--plant_to", type=int, default=86, help="exclusive")
    ap.add_argument("--ramp", type=int, default=5, help="frames to fade the plant in/out")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    pkl = glob.glob(f"{a.src}/robot/*.pkl")[0]
    data = joblib.load(pkl); key = list(data)[0]; M = data[key]
    rt, rr, dof = (np.array(M["root_trans_offset"]), np.array(M["root_rot"]),
                   np.array(M["dof"]))
    m = mujoco.MjModel.from_xml_path(
        _feas.build_scene(glob.glob(f"{a.src}/objects/*.pkl")[0]))
    d = mujoco.MjData(m)
    _, table, floor = scene_geoms(m)
    tops = top_surface(m)
    print(f"tabletop z={tops[0]:.3f}, footprint hull {len(tops[1])} verts")
    plant = [body_geoms(m, n) for n in PLANT_BODIES]
    clear = [body_geoms(m, n) for n in CLEAR_BODIES]
    axes = np.array([m.jnt_axis[j] for j in range(1, m.njnt)])

    q0 = np.concatenate([rt, rr, dof], axis=1)
    out = q0.copy()
    for p in range(a.passes):
        res = np.zeros_like(out)
        for f in range(len(dof)):
            on = a.plant_from <= f < a.plant_to
            start = out[f] if f == 0 else res[f - 1].copy()
            if f > 0:                                  # carry the neighbour's fix
                start = out[f] + (res[f - 1] - out[f - 1])
                start[3:7] /= np.linalg.norm(start[3:7])
            res[f] = solve_frame(m, d, start, q0[f], tops, table, floor,
                                 plant if on else [], clear if on else [],
                                 margin=a.margin)
        out = res if p == a.passes - 1 else smooth(res, q0, a.sigma)

    print(f"\n=== contact re-plant: {a.src}  frames {a.plant_from}-{a.plant_to-1} ===")
    print(f"  root moved   max {1000*np.linalg.norm(out[:,:3]-rt,axis=1).max():6.1f} mm"
          f"  mean {1000*np.linalg.norm(out[:,:3]-rt,axis=1).mean():6.1f} mm")
    print(f"  joints moved max {np.abs(out[:,7:]-dof).max():6.3f} rad"
          f"  mean {np.abs(out[:,7:]-dof).mean():6.3f} rad")
    jerk = lambda x: np.abs(np.diff(x, 3, axis=0)).max()
    print(f"  root jerk    {jerk(rt):.4f} -> {jerk(out[:,:3]):.4f} m")
    if a.dry_run:
        return

    M["root_trans_offset"] = out[:, :3].astype(np.float32)
    M["root_rot"] = out[:, 3:7].astype(np.float32)
    M["dof"] = out[:, 7:].astype(np.float32)
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
    joblib.dump({key: M}, os.path.join(a.dst, "robot", os.path.basename(pkl)))
    print(f"  written: {a.dst}")


if __name__ == "__main__":
    main()
