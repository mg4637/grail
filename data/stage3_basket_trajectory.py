#!/usr/bin/env python
"""Give the basket a trajectory by rigidly attaching it to the hand that takes it.

Stage 2 reconstructed exactly one object - the chair - so the basket has a known
START pose (from the Blender scene's static_objects, mapped into the motion-lib
frame by data/stage3_static_props.py) but no motion. It needs one: the clip's
last phase is the robot taking the basket off the shelf and carrying it down.

Rather than guess, the grasp frame is measured. Palm-surface distance to the
basket over the reach phase:

    left  hand   0.294 m (f120) -> 0.050 m (f150-156) -> 1.07 m (f228)
    right hand   never closer than 0.37 m

so the take is left-handed, closest approach ~5 cm around f150-156, after which
the hand departs carrying it. From the attach frame on, the basket is welded to
the hand frame:

    T_basket(f) = T_hand(f) @ T_hand(g)^-1 @ T_basket(g)

which is exactly the constraint a real grasp imposes, so the basket cannot drift
relative to the hand or pass through it. Before the attach frame it stays
exactly where the shelf put it.

ORIENTATION: TURNS WITH THE ROBOT
---------------------------------
A carried basket does not turn in the hands, but it does turn with the body -
and the robot yaws 80 deg during this phase (root yaw -52 deg by the end, via a
-76 deg excursion), so the two are very different things.

    quat(f) = Rz( yaw_root(f) - yaw_root(attach) ) * quat_on_shelf

i.e. the basket holds a constant orientation RELATIVE TO THE ROBOT, and follows
it in the world. In the source video this reads as "no rotation" because the
basket always looks the same relative to the kid - which is exactly what this
reproduces.

Two earlier attempts were wrong. Deriving yaw from the line between the hands
span the basket through two 90 deg turns (the hand line is noisy, and the sign
was wrong). Holding the orientation world-fixed removed that, but then the
basket stays pointing the same way while the robot turns 80 deg away from it.
Root yaw is the stable frame: it carries the body's turn without the wrist's
wobble.

TWO-HANDED HOLD
---------------
A pure one-hand weld is wrong once the basket is off the shelf. The two wrists
stay 0.38-0.57 m apart through the descent and the basket's long axis is 0.48 m,
so it is carried BETWEEN the hands, one at each end - which is what the video
shows. But at the moment of the take only the left hand reaches it (0.038 m,
against 0.37 m for the right), so the grasp itself really is one-handed.

So the weld blends: left-hand frame at the take, easing over `--blend` frames
into a two-handed frame whose origin is the wrist midpoint and whose long axis
(the basket's local +Y, its 0.482 m dimension) lies along the line between the
hands. That reproduces both facts instead of picking one.

This is a DERIVED trajectory, not an observed one. It is physically consistent
with the hand motion, but it is not evidence of what the basket did in the video.

Usage:
    python data/stage3_basket_trajectory.py <motion_lib_dir> [--attach auto] [--write]
"""
import argparse, glob, json, os, importlib.util
import numpy as np, joblib, mujoco, trimesh
from scipy.spatial.transform import Rotation as R

_spec = importlib.util.spec_from_file_location(
    "feas", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "stage3_mujoco_feasibility.py"))
_feas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feas)

HANDS = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}


def body_local_surface(m, d, bid, stride=4):
    """Collision-surface points of a body, expressed in ITS OWN frame.

    Computed once and then transformed per frame. Rebuilding these from the mesh
    assets every frame, against thousands of object vertices, is what made the
    first version of this script time out.
    """
    mujoco.mj_forward(m, d)
    Rb, tb = d.xmat[bid].reshape(3, 3), d.xpos[bid]
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or not m.geom_contype[g]:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]][::stride]
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        else:
            w = d.geom_xpos[g][None]
        pts.append((w - tb) @ Rb)                     # world -> body frame
    return np.vstack(pts)


def _slerp(q0, q1, w):
    """Shortest-path interpolation between two rotations."""
    rv = (q0.inv() * q1).as_rotvec()
    return q0 * R.from_rotvec(rv * w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion_dir")
    ap.add_argument("--attach", default="auto",
                    help="frame to weld the basket to the hand, or 'auto'")
    ap.add_argument("--orient", choices=["body", "fixed", "hands"], default="body",
                    help="body turns the object with the robot's root yaw "
                         "(correct for a carried object); fixed keeps it "
                         "world-locked; hands aligns it with the hand line")
    ap.add_argument("--blend", type=int, default=30,
                    help="frames to ease from the one-hand weld into the "
                         "two-handed hold (0 disables)")
    ap.add_argument("--release", type=int, default=None,
                    help="frame to let go again (default: never)")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    props = json.load(open(f"{a.motion_dir}/common/static_props.json"))
    bpos = np.asarray(props["basket"]["pos"], dtype=float)
    bm = trimesh.load(props["basket"]["obj"])
    if isinstance(bm, trimesh.Scene):
        bm = bm.dump(concatenate=True)
    BV = np.asarray(bm.vertices) + bpos

    rp = glob.glob(f"{a.motion_dir}/robot/*.pkl")[0]
    data = joblib.load(rp); key = list(data)[0]; M = data[key]
    rt, rr, dof = (np.array(M["root_trans_offset"]), np.array(M["root_rot"]),
                   np.array(M["dof"]))
    T = len(dof)

    m = mujoco.MjModel.from_xml_path(_feas.G1_MJCF)
    d = mujoco.MjData(m)
    B = {h: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for h, n in HANDS.items()}

    # subsample the basket too - 10729 vertices is far more resolution than a
    # 5 cm proximity test needs
    BVs = BV[::20]
    d.qpos[0:3], d.qpos[3:7], d.qpos[7:36] = rt[0], rr[0], dof[0]
    local = {h: body_local_surface(m, d, bid) for h, bid in B.items()}

    dist = {h: np.full(T, np.inf) for h in HANDS}
    xpos = {h: np.zeros((T, 3)) for h in HANDS}
    xmat = {h: np.zeros((T, 3, 3)) for h in HANDS}
    from scipy.spatial import cKDTree
    tree = cKDTree(BVs)
    for f in range(T):
        d.qpos[0:3], d.qpos[3:7], d.qpos[7:36] = rt[f], rr[f], dof[f]
        mujoco.mj_forward(m, d)
        for h, bid in B.items():
            Rb, tb = d.xmat[bid].reshape(3, 3), d.xpos[bid]
            dist[h][f] = tree.query(local[h] @ Rb.T + tb)[0].min()
            xpos[h][f] = tb
            xmat[h][f] = Rb

    root_yaw = np.unwrap([R.from_quat(q[[1, 2, 3, 0]]).as_euler("zyx")[0]
                          for q in np.array(M["root_rot"])])
    hand = min(HANDS, key=lambda h: dist[h].min())
    g = int(np.argmin(dist[hand])) if a.attach == "auto" else int(a.attach)
    rel = a.release if a.release is not None else T

    print(f"\n=== basket trajectory: {a.motion_dir} ===")
    for h in HANDS:
        print(f"  {h:<5} closest {dist[h].min():.3f} m at frame {int(np.argmin(dist[h]))}")
    print(f"  attaching to the {hand} hand at frame {g} (gap {dist[hand][g]:.3f} m)")

    # basket pose before attach: exactly where the shelf put it
    pos = np.repeat(bpos[None], T, 0)
    quat = np.repeat(np.array([1.0, 0, 0, 0])[None], T, 0)   # baked mesh -> identity
    Rg, tg = xmat[hand][g], xpos[hand][g]
    off_R = Rg.T                                   # basket rot is identity at g
    off_t = Rg.T @ (bpos - tg)
    def carry_yaw(f):
        """Heading of the line between the hands, projected to horizontal."""
        v = xpos["right"][f] - xpos["left"][f]
        if np.hypot(v[0], v[1]) < 1e-6:
            return None
        # the basket's long axis is its local +Y (0.482 m of 0.301/0.482/0.254),
        # so local Y must lie along the hand line: yaw = heading - 90 deg
        return np.arctan2(v[1], v[0]) - 0.5 * np.pi

    # ORIENTATION: upright, yaw only.
    #
    # Welding the basket to the wrist's full rotation makes it tumble - measured
    # 21 deg of tilt through the blend - and a carried basket stays level. So the
    # basket keeps world up and only turns about z, following the line between
    # the hands. The yaw is unwrapped before smoothing so it cannot wind the
    # short way round and spin the basket through half a turn.
    yaws = []
    for f in range(g, min(rel, T)):
        y = carry_yaw(f)
        yaws.append(yaws[-1] if y is None and yaws else (y if y is not None else 0.0))
    yaws = np.unwrap(np.array(yaws))
    # The basket sits on the shelf at yaw 0 and is carried at the hand heading,
    # so those two disagree at the instant of the take - blending FROM the carry
    # heading left that disagreement as a 120 deg jump on one frame. Ease from
    # the shelf orientation instead, and keep the unwrapped target so the basket
    # turns the short way rather than winding through half a turn.
    shelf_yaw = 0.0
    if len(yaws):
        yaws = yaws - 2 * np.pi * np.round((yaws[0] - shelf_yaw) / (2 * np.pi))

    for i, f in enumerate(range(g, min(rel, T))):
        Rf, tf = xmat[hand][f], xpos[hand][f]
        p_hand = Rf @ off_t + tf
        p_mid = 0.5 * (xpos["left"][f] + xpos["right"][f])
        w = min(1.0, i / float(a.blend)) if a.blend > 0 else 0.0
        pos[f] = (1 - w) * p_hand + w * p_mid
        if a.orient == "body":
            q0 = R.from_quat((quat[g - 1] if g > 0 else np.array([1.0, 0, 0, 0]))[[1, 2, 3, 0]])
            quat[f] = (R.from_euler("z", root_yaw[f] - root_yaw[g]) * q0).as_quat()[[3, 0, 1, 2]]
        elif a.orient == "fixed":
            quat[f] = quat[g - 1] if g > 0 else np.array([1.0, 0, 0, 0])
        else:
            yaw = (1 - w) * shelf_yaw + w * yaws[i]
            quat[f] = R.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]

    for f in range(min(rel, T), T):                # frozen after release
        pos[f], quat[f] = pos[min(rel, T) - 1], quat[min(rel, T) - 1]

    trav = np.linalg.norm(pos[-1] - pos[0])
    print(f"  basket travels {trav:.2f} m; z {pos[:,2].min():.3f}..{pos[:,2].max():.3f}"
          f"  (start {bpos[2]:.3f}, end {pos[-1,2]:.3f})")

    if a.write:
        out = {key: {"root_pos": pos[:, None, :].astype(np.float32),
                     "root_quat": quat[:, None, :].astype(np.float32),
                     "object_name": "basket", "fps": float(M["fps"]),
                     "scale": np.ones((3, 1), dtype=np.float32),
                     "attach_frame": g, "attach_hand": hand,
                     "mesh": props["basket"]["obj"]}}
        od = os.path.join(a.motion_dir, "objects_basket")
        os.makedirs(od, exist_ok=True)
        p = os.path.join(od, os.path.basename(rp))
        joblib.dump(out, p)
        print(f"  written {p}")


if __name__ == "__main__":
    main()
