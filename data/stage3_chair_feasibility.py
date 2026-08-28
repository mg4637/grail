#!/usr/bin/env python
"""Contact-force feasibility for the chair task, with carried objects as payload.

WHY THIS IS NOT stage3_contact_feasibility.py
---------------------------------------------
That script was written for the table climb, where the object is a static
SUPPORT the robot pushes against. Here the objects are CARRIED, which is the
opposite: the chair is not something the robot stands on during the carry, it is
a 3 kg load hanging off the hands, and the only contacts with the world are the
feet on the floor. Treating a carried object as scenery would let the analysis
"support" the robot on the very thing it is holding up.

So per phase:

    __carry    f<place: support = floor,               payload = chair (3 kg)
               f>=place: support = floor + placed chair, payload = none
    __basket   support = floor + placed chair,           payload = basket (0.5 kg)

The chair changes ROLE inside the carry motion: it is a load while carried and a
support once set down and stood on. Modelling it as a payload throughout reports
the whole step-up as unsupported (nc=0, residual 300-500 N), which is an artefact
of the model, not of the motion. `--payload_until` is the switch.

The payload enters through the unactuated base rows. For a floating base an
external force f applied anywhere on the robot contributes [f; (p - p_base) x f]
regardless of which link carries it, so no decomposition of the payload mesh is
needed - only its mass and the trajectory it is already following:

    f_payload = -m (a_obj - g)          (static: -m*9.81 z, i.e. it pulls down)
    rhs = [M qddot + h]_{0:6} - [f; (p_obj - p_base) x f]

COLLISION GEOMETRY
------------------
The chair the robot stands on in __basket does need real geometry, and neither
splitting nor hulling works: `split` breaks these meshes into coplanar polygons
that MuJoCo refuses to hull, and keeping them whole makes the chair a solid
block whose top face is the top of its BACK, not the seat. With no convex
decomposition backend installed (coacd/vhacd absent both on the host and in the
container), the support is voxelised instead - 426 boxes at a 4 cm pitch for the
chair, which reproduces the seat, the gap under it and the legs.

Usage:
    python data/stage3_chair_feasibility.py <split_lib> --key <motion_key> [--mu 1.0]
"""
import argparse, glob, os, importlib.util
import numpy as np, joblib, mujoco, trimesh
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation as R

_spec = importlib.util.spec_from_file_location(
    "cf", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "stage3_contact_feasibility.py"))
_cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cf)
_feas = _cf._feas


def voxel_scene(mjcf, meshes, pitch=0.04):
    """G1 MJCF + a floor + support meshes as voxel boxes (world coords baked in)."""
    import xml.etree.ElementTree as ET, tempfile
    tree = ET.parse(mjcf); root = tree.getroot()
    world = root.findall("worldbody")[-1]
    tmp = tempfile.mkdtemp(prefix="chair_feas_")
    n = 0
    for path in meshes:
        m = trimesh.load(path)
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        pts = np.asarray(m.voxelized(pitch=pitch).points)
        for p in pts:
            ET.SubElement(world, "geom", {
                "name": f"sup_{n}", "type": "box",
                "size": f"{pitch/2:.4f} {pitch/2:.4f} {pitch/2:.4f}",
                "pos": " ".join(f"{v:.5f}" for v in p),
                "rgba": "0.55 0.38 0.22 1", "friction": "1.0 0.005 0.0001"})
            n += 1
    root.find("compiler").set("meshdir", os.path.abspath(
        os.path.join(os.path.dirname(mjcf), "../meshes/g1/")))
    # `x or y` is wrong for ElementTree: an Element with no children is falsy,
    # so an existing <visual> gets shadowed and a duplicate <global> is appended.
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    g = vis.find("global")
    if g is None:
        g = ET.SubElement(vis, "global")
    g.set("offwidth", "1920"); g.set("offheight", "1080")
    out = os.path.join(tmp, "scene.xml"); tree.write(out)
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lib"); ap.add_argument("--key", required=True)
    ap.add_argument("--payload_mass", type=float, default=3.0)
    ap.add_argument("--payload_until", type=int, default=None,
                    help="frame at which the payload is set down and stops being a load")
    ap.add_argument("--payload_from", type=int, default=None,
                    help="frame at which the payload is picked up")
    ap.add_argument("--support_object_at", type=int, default=None,
                    help="voxelise the motion's own object at this frame as support")
    ap.add_argument("--supports", nargs="*", default=None,
                    help="meshes to voxelise as support (floor is always there)")
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--smooth", type=float, default=1.0)
    ap.add_argument("--severe", type=float, default=100.0)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()

    rp = [p for p in glob.glob(f"{a.lib}/robot/*.pkl") if a.key in p][0]
    M = joblib.load(rp)[a.key]
    op_ = [p for p in glob.glob(f"{a.lib}/objects/*.pkl") if a.key in p][0]
    O = joblib.load(op_)[a.key]
    rt, rr, dof = (np.array(M["root_trans_offset"]), np.array(M["root_rot"]),
                   np.array(M["dof"]))
    obj_p = np.array(O["root_pos"])[:, 0]
    fps = float(M["fps"]); T = len(dof)

    supports = list(a.supports or [])
    if a.support_object_at is not None:
        import tempfile
        mesh_path = O.get("mesh") or "results/generation/mesh/ComAsset/chair/model.obj"
        om = trimesh.load(mesh_path)
        if isinstance(om, trimesh.Scene):
            om = om.dump(concatenate=True)
        sc = np.asarray(O.get("scale", 1.0)).reshape(-1)
        v = np.asarray(om.vertices) * (sc[:3] if sc.size >= 3 else float(sc[0]))
        k = a.support_object_at
        q = np.array(O["root_quat"])[k, 0]
        om.vertices = v @ R.from_quat(q[[1, 2, 3, 0]]).as_matrix().T + np.array(O["root_pos"])[k, 0]
        f_ = os.path.join(tempfile.mkdtemp(), "support_object.obj")
        om.export(f_)
        supports.append(f_)
    scene, nvox = voxel_scene(_feas.G1_MJCF, supports)
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    robot_mass = m.body_mass[1:].sum()

    qpos = np.concatenate([rt, rr, dof], axis=1)
    qvel = np.zeros((T, m.nv))
    for f in range(T - 1):
        mujoco.mj_differentiatePos(m, qvel[f], 1.0 / fps, qpos[f], qpos[f + 1])
    qvel[-1] = qvel[-2]
    if a.smooth > 0:
        from scipy.ndimage import gaussian_filter1d
        qvel = gaussian_filter1d(qvel, a.smooth, axis=0, mode="nearest")
        obj_p = gaussian_filter1d(obj_p, a.smooth, axis=0, mode="nearest")
    qacc = np.gradient(qvel, 1.0 / fps, axis=0)
    obj_a = np.gradient(np.gradient(obj_p, 1.0 / fps, axis=0), 1.0 / fps, axis=0)

    world = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "world")
    scene_b = {world}
    print(f"\n=== {a.key.split('__')[-1]} ===")
    print(f"robot {robot_mass:.1f} kg, payload {a.payload_mass:.1f} kg, "
          f"{nvox} support voxels + floor, mu={a.mu}")
    print(f"{'f':>4} {'nc':>3} {'|rhs| N':>9} {'residual N':>11}  contacts")
    bad = []
    for f in range(1, T - 1, a.stride):
        d.qpos[:] = qpos[f]; d.qvel[:] = qvel[f]
        mujoco.mj_forward(m, d)
        cs = _cf.contact_set(m, d, scene_b, a.tol)
        d.qacc[:] = qacc[f]
        saved = m.opt.disableflags
        m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
                               | mujoco.mjtDisableBit.mjDSBL_CONTACT)
        mujoco.mj_inverse(m, d)
        m.opt.disableflags = saved
        rhs = d.qfrc_inverse[:6].copy()
        # payload: force it applies to the robot, at its own COM
        held = True
        if a.payload_until is not None and f >= a.payload_until:
            held = False
        if a.payload_from is not None and f < a.payload_from:
            held = False
        pm = a.payload_mass if held else 0.0
        fpay = -pm * (obj_a[f] - np.array([0.0, 0.0, -9.81]))
        r = obj_p[f] - d.qpos[:3]
        rhs[:3] -= fpay
        rhs[3:6] -= np.cross(r, fpay)
        res = _cf.base_wrench_residual(m, d, cs, rhs, a.mu)
        labs = sorted({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                       .replace("_link", "").replace("left_", "L.").replace("right_", "R.")
                       for _, _, b in cs})
        if res > 1.0:
            bad.append((f, res))
        if f % (a.stride * 6) == 1 or res > a.severe:
            flag = "SEVERE" if res > a.severe else ""
            print(f"{f:4d} {len(cs):3d} {np.linalg.norm(rhs):9.1f} {res:11.1f} {flag:>7}  "
                  f"{','.join(labs)[:44]}")
    sev = [f for f, r in bad if r > a.severe]
    n = len(range(1, T - 1, a.stride))
    print(f"\ninfeasible {len(bad)}/{n}   SEVERE (>{a.severe:.0f} N) {len(sev)}"
          + (f"  frames {sev[:14]}" if sev else ""))


if __name__ == "__main__":
    main()
