#!/usr/bin/env python
"""Open-loop physics replay of a retargeted G1 motion, in MuJoCo.

Answers a question no amount of RL tuning can: is the reference something the
G1 can execute at all, given its real actuator limits? The reference joint
angles are fed straight to a PD controller using SONIC's own gains and effort
limits (gear_sonic/envs/manager_env/robots/g1.py:589-707); there is no policy.

USE --pin_base. THE FREE-BASE MODE CANNOT ANSWER ANYTHING.
------------------------------------------------------------------------------
Joint-space PD contains no balance strategy, so a free-floating humanoid topples
whatever you feed it. Measured control: holding reference frame 0 - a static,
two-foot stance - drops the pelvis 0.755 -> 0.165 m in one second, with the leg
joints tracking to within 0.02 rad the whole way down. The robot did not fail to
hold the pose; it held it perfectly and fell over, exactly as an uncontrolled
inverted pendulum must. Any "DID NOT CLIMB" from the free-base mode is therefore
uninformative, and its saturation table is contaminated by the flailing.

`--pin_base` overwrites the floating base with the reference root pose every sim
step, making it kinematic. That removes the balance problem - which is precisely
the problem the RL policy exists to solve - and isolates the remaining one: given
the root path, can the limbs hit the reference angles against gravity and the
table within the real effort limits? Saturation there is a property of the
motion, and no amount of Stage 4 tuning will fix it.

Usage: python data/stage3_mujoco_feasibility.py <motion_lib_dir> [--start FRAME]
                                                [--pin_base] [--hold FRAME]
"""
import argparse, glob, os, tempfile
import xml.etree.ElementTree as ET
import numpy as np, joblib, trimesh, mujoco
from scipy.spatial.transform import Rotation as R

G1_MJCF = "imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
MESH_OBJ = ("results/generation/4dhoi_recon_smplx_valid/ComAsset/table/"
            "kid_indoor2-climb-table_rand00001/mesh_data/model.obj")
# SONIC actuator groups: (regex fragment, kp, kd, effort)
GAINS = [("hip_yaw", 40.18, 2.558, 88.0), ("waist_yaw", 40.18, 2.558, 88.0),
         ("hip_roll", 99.10, 6.309, 139.0), ("hip_pitch", 99.10, 6.309, 139.0),
         ("knee", 99.10, 6.309, 139.0),
         ("ankle_pitch", 28.50, 1.814, 50.0), ("ankle_roll", 28.50, 1.814, 50.0),
         ("waist_roll", 28.50, 1.814, 50.0), ("waist_pitch", 28.50, 1.814, 50.0),
         ("wrist_pitch", 16.78, 1.068, 5.0), ("wrist_yaw", 16.78, 1.068, 5.0),
         ("wrist_roll", 14.25, 0.907, 25.0), ("shoulder", 14.25, 0.907, 25.0),
         ("elbow", 14.25, 0.907, 25.0)]


def gains_for(name):
    for frag, kp, kd, eff in GAINS:          # longest fragment wins
        pass
    best = max((g for g in GAINS if g[0] in name), key=lambda g: len(g[0]), default=None)
    if best is None:
        raise KeyError(name)
    return best[1], best[2], best[3]


def build_scene(obj_pkl):
    """G1 MJCF + the table, as 15 convex mesh geoms at the reference object pose."""
    O = list(joblib.load(obj_pkl).values())[0]
    pos = np.array(O["root_pos"])[0, 0]
    quat = np.array(O["root_quat"])[0, 0]              # wxyz
    mesh = trimesh.load(MESH_OBJ)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    Rm = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    parts = mesh.split(only_watertight=False)

    tmpdir = tempfile.mkdtemp(prefix="g1_feas_")
    tree = ET.parse(G1_MJCF)
    root = tree.getroot()
    asset = root.findall("asset")[0]
    world = root.findall("worldbody")[-1]
    table = ET.SubElement(world, "body", {"name": "table", "pos": "0 0 0"})
    for i, p in enumerate(parts):
        p = p.copy()
        p.vertices = p.vertices @ Rm.T + pos           # bake the pose into the mesh
        f = os.path.join(tmpdir, f"table_{i}.obj")
        p.export(f)
        ET.SubElement(asset, "mesh", {"name": f"table_{i}", "file": f})
        ET.SubElement(table, "geom", {"type": "mesh", "mesh": f"table_{i}",
                                      "rgba": "0.6 0.4 0.2 1", "friction": "1.0 0.005 0.0001"})
    # The G1 MJCF has no <visual> block, so the offscreen framebuffer defaults to
    # 640x480 and any larger Renderer fails. Needed by stage3_render_reference.py.
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    g = vis.find("global")
    if g is None:
        g = ET.SubElement(vis, "global")
    g.set("offwidth", "1920")
    g.set("offheight", "1080")

    out = os.path.join(tmpdir, "scene.xml")
    # keep meshdir working: compiler path is relative to the ORIGINAL xml
    root.find("compiler").set("meshdir", os.path.abspath(
        os.path.join(os.path.dirname(G1_MJCF), "../meshes/g1/")))
    tree.write(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion_dir")
    ap.add_argument("--start", type=int, default=0, help="reference frame to start from")
    ap.add_argument("--ctrl_hz", type=float, default=50.0)
    ap.add_argument("--pin_base", action="store_true",
                    help="drive the floating base along the reference root path "
                         "(kinematic), isolating limb feasibility from balance")
    ap.add_argument("--hold", type=int, default=None,
                    help="instead of replaying, hold this reference frame for 2 s")
    a = ap.parse_args()

    M = list(joblib.load(glob.glob(f"{a.motion_dir}/robot/*.pkl")[0]).values())[0]
    dof, rt, rr = np.array(M["dof"]), np.array(M["root_trans_offset"]), np.array(M["root_rot"])
    fps = float(M["fps"])
    scene = build_scene(glob.glob(f"{a.motion_dir}/objects/*.pkl")[0])
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    m.opt.timestep = 0.002

    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    kp, kd, eff = (np.array(x) for x in zip(*[gains_for(n) for n in names]))

    # SONIC drives the G1 with IsaacLab ImplicitActuators: the PD damping term is
    # solved inside the integrator, not applied as an external torque. Reproducing
    # it as an explicit torque at MuJoCo's 2 ms step is unstable on the low-inertia
    # wrists (kd*dt/I > 1) and produces pure numerical garbage - measured: the
    # wrist rolls "saturate" 94% of the time and drift 2 rad off a zero target,
    # which vanishes entirely at dt=0.5 ms. So put kd in dof_damping and let
    # IMPLICITFAST integrate it, matching the real actuator model.
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.dof_damping[6:6 + m.nu] = kd

    # upsample the reference to the control rate
    T = len(dof)
    t_src = np.arange(T) / fps
    t_ctl = np.arange(a.start / fps, t_src[-1], 1.0 / a.ctrl_hz)
    q_ref = np.stack([np.interp(t_ctl, t_src, dof[:, j]) for j in range(29)], 1)
    z_ref = np.interp(t_ctl, t_src, rt[:, 2])

    # root path at the control rate, for --pin_base
    p_ref = np.stack([np.interp(t_ctl, t_src, rt[:, j]) for j in range(3)], 1)
    q_slerp = R.from_quat(rr[:, [1, 2, 3, 0]])
    from scipy.spatial.transform import Slerp
    r_ref = Slerp(t_src, q_slerp)(np.clip(t_ctl, t_src[0], t_src[-1])).as_quat()[:, [3, 0, 1, 2]]

    # base twist: linear in world, angular in body frame (MuJoCo free-joint order)
    v_ref = np.gradient(p_ref, 1.0 / a.ctrl_hz, axis=0)
    Rc = R.from_quat(r_ref[:, [1, 2, 3, 0]])
    dR = Rc[:-1].inv() * Rc[1:]
    w_ref = np.zeros_like(p_ref)
    w_ref[:-1] = dR.as_rotvec() * a.ctrl_hz
    w_ref[-1] = w_ref[-2]

    if a.hold is not None:
        q_ref = np.repeat(dof[a.hold][None], int(2.0 * a.ctrl_hz), 0)
        p_ref = np.repeat(rt[a.hold][None], len(q_ref), 0)
        r_ref = np.repeat(rr[a.hold][None], len(q_ref), 0)
        z_ref = p_ref[:, 2]
        v_ref = np.zeros_like(p_ref); w_ref = np.zeros_like(p_ref)
        t_ctl = np.arange(len(q_ref)) / a.ctrl_hz
        a.start = a.hold

    d.qpos[0:3], d.qpos[3:7], d.qpos[7:36] = rt[a.start], rr[a.start], dof[a.start]
    mujoco.mj_forward(m, d)

    steps_per_ctl = int(round(1.0 / a.ctrl_hz / m.opt.timestep))
    sat = np.zeros(m.nu); z_act = []; jerr = np.zeros(m.nu)
    for k in range(len(t_ctl)):
        for _ in range(steps_per_ctl):
            if a.pin_base:
                # Kinematic base: the root follows the reference exactly, so the
                # only thing under test is whether the limbs can keep up. The
                # base VELOCITY must be set too - zeroing it while teleporting
                # the position each step shakes the whole kinematic chain and
                # shows up as metres of phantom tracking error in the arms.
                d.qpos[0:3], d.qpos[3:7] = p_ref[k], r_ref[k]
                d.qvel[0:3] = v_ref[k]
                d.qvel[3:6] = w_ref[k]
            # tau is what the actuator would command; the -kd*qdot half is already
            # applied implicitly via dof_damping, so only the remainder is fed to
            # ctrl. Saturation is still judged on the full commanded torque.
            damp = kd * d.qvel[6:35]
            tau = kp * (q_ref[k] - d.qpos[7:36]) - damp
            sat += np.abs(tau) >= eff * 0.999
            d.ctrl[:] = np.clip(tau, -eff, eff) + damp
            mujoco.mj_step(m, d)
        jerr = np.maximum(jerr, np.abs(q_ref[k] - d.qpos[7:36]))
        z_act.append(d.qpos[2])
    z_act = np.array(z_act)
    n = len(t_ctl) * steps_per_ctl

    mode = "base PINNED to reference" if a.pin_base else "FREE base (uninformative)"
    what = f"hold frame {a.hold} for 2 s" if a.hold is not None else f"replay from frame {a.start}"
    print(f"\n=== {a.motion_dir}  ({what}, {mode}) ===")
    print(f"pelvis z   reference {z_ref[0]:.3f} -> {z_ref[-1]:.3f}")
    print(f"           replay    {z_act[0]:.3f} -> {z_act[-1]:.3f}   "
          f"(final gap {abs(z_act[-1]-z_ref[-1]):.3f} m)")
    if not a.pin_base:
        climbed = z_act[-1] > 1.35
        print(f"           => {'CLIMBED' if climbed else 'DID NOT CLIMB'}  "
              f"(free base: says nothing about the reference, see docstring)")
    B = lambda s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s)
    print(f"final feet z: L {d.xpos[B('left_ankle_roll_link')][2]:.3f} "
          f"R {d.xpos[B('right_ankle_roll_link')][2]:.3f}  (tabletop 0.905)")
    print("\nactuator saturation (% of sim steps at the effort limit), worst 12:")
    print(f"   {'joint':<28} {'sat%':>6} {'limit':>7}  {'max track err (rad)':>20}")
    for i in np.argsort(-sat)[:12]:
        if sat[i] == 0:
            break
        print(f"   {names[i]:<28} {100*sat[i]/n:5.1f}% {eff[i]:6.0f}  {jerr[i]:20.3f}")
    print(f"\n   worst joint tracking error overall: "
          f"{names[int(np.argmax(jerr))]} {jerr.max():.3f} rad")


if __name__ == "__main__":
    main()
