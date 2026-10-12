"""Lesson 3: replay the retargeted Unitree G1 motion on the stairs in MuJoCo.

Two modes:
  window (interactive):   docs_learning/.venv/bin/mjpython docs_learning/lessons/03_view_retargeted_motion.py
  video  (no window):     docs_learning/.venv/bin/python   docs_learning/lessons/03_view_retargeted_motion.py --video out.mp4

Key ideas this teaches
  * KINEMATIC playback: we never call mj_step (no physics). For every frame we write the pose into
    data.qpos and call mj_forward, which only computes positions. It shows the REFERENCE motion,
    exactly what the RL policy will later be rewarded for imitating.
  * qpos layout of the G1: [x y z | qw qx qy qz | 29 hinge angles] = 3 + 4 + 29 = 36 numbers.
  * MuJoCo quaternions are scalar-first (w, x, y, z); GRAIL saves root_rot in that same order.
  * On macOS the interactive viewer needs `mjpython` (it runs the GUI on the main thread).
"""
import argparse
import ast
import pickle
import re
import sys
import time
from pathlib import Path

import joblib
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KEY = "stairs_wide3_kid_indoor1-syn-stairs_rand00001"
MOTION_DIR = ROOT / "data/motion_lib/stairs_wide3"   # overridden by --motion_dir
G1_XML = ROOT / "imports/GMR/assets/unitree_g1/g1_mocap_29dof.xml"

# stairs_wide3 geometry in its own (mesh) frame, from docs_learning/lessons/02_wide_stairs.py
N_STEPS, RISE, TREAD, WIDTH = 3, 0.12, 0.35, 1.5


def build_model():
    """G1 model + the stairs as 3 boxes + a floor, placed with the pose stored in the motion data."""
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    obj = joblib.load(MOTION_DIR / f"objects/{KEY}.pkl")[KEY]
    pos = obj["root_pos"][0, 0]          # (x, y, z)
    quat = obj["root_quat"][0, 0]        # (w, x, y, z)
    total = 2 * N_STEPS * TREAD
    body = spec.worldbody.add_body(name="stairs", pos=[float(p) for p in pos], quat=[float(q) for q in quat])
    for i in range(N_STEPS):
        x0, x1 = TREAD * i, total - TREAD * i       # same box rule as gen_terrain.generate_stair_pieces
        h = RISE * (i + 1)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, name=f"step{i}",
                      pos=[(x0 + x1) / 2, 0, h / 2], size=[(x1 - x0) / 2, WIDTH / 2, h / 2],
                      rgba=[0.3 + 0.2 * i, 0.5, 0.85 - 0.2 * i, 1], contype=1, conaffinity=1)
    # the G1 XML already defines a ground plane called 'floor' at z=0 (world origin), so we add none
    return spec.compile()


def load_policy_trajectory(path, env_idx):
    """Load a SONIC `*.trajectory.pkl` (policy rollout recorded by TrajectoryRecorderTerm) as a motion dict.

    The recorder stores joint angles in ISAAC LAB joint order, root position relative to the env origin and the
    root quaternion as wxyz. The MuJoCo model (and our retargeted motion) uses GMR/MuJoCo joint order, so the
    angles are re-ordered with SONIC's own list G1_ISAACLAB_TO_MUJOCO_DOF (parsed from its source file, so we never
    retype 29 numbers by hand).
    """
    src = (ROOT / "imports/SONIC/gear_sonic/envs/manager_env/robots/g1.py").read_text()
    m = re.search(r"G1_ISAACLAB_TO_MUJOCO_DOF\s*=\s*(\[.*?\])", src, re.S)
    i2m = ast.literal_eval(m.group(1))
    m2 = re.search(r"G1_MUJOCO_TO_ISAACLAB_DOF\s*=\s*(\[.*?\])", src, re.S)
    m2i = ast.literal_eval(m2.group(1))
    assert sorted(i2m) == list(range(29)) and [i2m[j] for j in m2i] == list(range(29)), "permutation lists inconsistent"
    p = Path(path)
    if p.is_dir():
        p = sorted(p.glob("*.trajectory.pkl"))[env_idx]
    with open(p, "rb") as f:
        traj = pickle.load(f)
    assert traj.get("quat_format") == "wxyz", traj.get("quat_format")
    dof = np.asarray(traj["dof_pos"])[:, i2m]
    print(f"loaded {p.name}: {len(dof)} frames @ {traj['fps']:.1f} fps")
    return {"root_trans_offset": np.asarray(traj["root_pos_w"]), "root_rot": np.asarray(traj["root_quat_w"]),
            "dof": dof, "fps": traj["fps"]}


def set_frame(m, d, motion, t):
    d.qpos[0:3] = motion["root_trans_offset"][t]
    d.qpos[3:7] = motion["root_rot"][t]          # w, x, y, z
    d.qpos[7:] = motion["dof"][t]                # 29 joints, same order as the XML hinges
    mujoco.mj_forward(m, d)


def draw_force_arrows(v, d, forces, foot_bodies, t, scale, min_newton=5.0):
    """Draw one arrow per foot (green = left, orange = right) from the ankle body, length proportional to the force.
    The force is the REAL contact force on that foot from the Isaac Sim/PhysX run (not computed by MuJoCo).
    Arrows start at the ankle link, not at the true contact point (the contact position was not recorded)."""
    scn = v.user_scn
    scn.ngeom = 0
    colors = [np.array([0.1, 0.8, 0.2, 1], dtype=np.float32), np.array([1.0, 0.5, 0.0, 1], dtype=np.float32)]
    for k, body in enumerate(foot_bodies):
        f = forces[t, k]
        if np.linalg.norm(f) < min_newton:
            continue
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), colors[k])
        p0 = np.array(d.xpos[body], dtype=np.float64)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, p0, p0 + np.asarray(f, dtype=np.float64) * scale)
        scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="write an mp4 instead of opening a window")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--motion_dir", default=None, help="another motion_lib folder, relative to the GRAIL root")
    ap.add_argument("--trajectory", default=None, help="a policy rollout (*.trajectory.pkl or a folder of them) instead of the reference motion")
    ap.add_argument("--env", type=int, default=0, help="which environment of the rollout to show")
    ap.add_argument("--forces", action="store_true", help="draw the REAL PhysX foot contact forces recorded in <rollout>/forces.npz as arrows")
    ap.add_argument("--force_scale", type=float, default=0.0015, help="arrow length in metres per newton (robot weight ~337 N -> ~0.5 m)")
    args = ap.parse_args()
    global MOTION_DIR
    if args.motion_dir:
        MOTION_DIR = ROOT / args.motion_dir

    motion = load_policy_trajectory(args.trajectory, args.env) if args.trajectory else joblib.load(MOTION_DIR / f"robot/{KEY}.pkl")[KEY]
    fps, T = motion["fps"], len(motion["dof"])
    m = build_model(); d = mujoco.MjData(m)
    print(f"{T} frames @ {fps} fps = {T / fps:.1f} s; qpos size {m.nq}")
    forces, foot_bodies = None, []
    if args.forces:
        assert args.trajectory, "--forces needs --trajectory <rollout folder>"
        fz = np.load(Path(args.trajectory) / "forces.npz", allow_pickle=True)
        forces = fz["foot_net_force_w"][:, args.env]            # (frames, 2 feet, 3): force ON the foot, world frame, newtons
        foot_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        mass = float(fz["robot_mass_kg"]) if "robot_mass_kg" in fz.files else float("nan")
        print(f"loaded forces.npz: {len(forces)} frames; robot mass {mass:.1f} kg (weight {mass * 9.81:.0f} N); arrow scale {args.force_scale} m/N")

    if args.video:
        import cv2
        r = mujoco.Renderer(m, height=720, width=1280)
        cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = 150, -15, 4.2
        w = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
        for t in range(T):
            set_frame(m, d, motion, t)
            cam.lookat[:] = [-0.3, 0.0, 0.5]
            r.update_scene(d, camera=cam)
            w.write(r.render()[:, :, ::-1])
        w.release(); print("wrote", args.video)
        return

    from mujoco import viewer  # NOT `import mujoco.viewer` here: that would make `mujoco` a local name in main()
    with viewer.launch_passive(m, d) as v:
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 150, -15, 4.2
        v.cam.lookat[:] = [-0.3, 0.0, 0.5]
        t = 0
        while v.is_running():
            tic = time.time()
            set_frame(m, d, motion, t % T)
            if forces is not None:
                with v.lock():
                    draw_force_arrows(v, d, forces, foot_bodies, (t % T) % len(forces), args.force_scale)
            v.sync()
            t += 1
            time.sleep(max(0.0, 1.0 / (fps * args.speed) - (time.time() - tic)))


if __name__ == "__main__":
    sys.exit(main())
