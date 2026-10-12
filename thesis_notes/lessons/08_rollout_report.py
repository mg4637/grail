"""Lesson 8: numerical report on a policy rollout (SONIC TrajectoryRecorder files) — no window needed.

  docs_learning/.venv/bin/python docs_learning/lessons/08_rollout_report.py <rollout_folder> [reference_motion_dir]

For every recorded environment it reports
  1. resets: jumps of the root position between consecutive frames (the simulator teleports a robot back to the
     start when its episode ends),
  2. tracking: distance of the root from the REFERENCE motion at the same elapsed time, pelvis height,
  3. contact under physics: gap between each foot and the stairs surface beneath it (same method as lesson 04;
     gap > 0 hovering, ~0 touching, < 0 inside). A simulated robot cannot hover, so this also checks that our
     analytic stairs model and the simulator's stairs agree.
The rollout joints are re-ordered from Isaac Lab to MuJoCo order with SONIC's own permutation list (lesson 03 loader).
"""
import glob
import importlib.util
import sys
from pathlib import Path

import joblib
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("v03", ROOT / "docs_learning/lessons/03_view_retargeted_motion.py")
v03 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v03)

rollout = Path(sys.argv[1])
ref_dir = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "data/motion_lib/stairs_wide3_terrain")
K = "stairs_wide3_kid_indoor1-syn-stairs_rand00001"
ref = joblib.load(ref_dir / f"robot/{K}.pkl")[K]
obj = joblib.load(ref_dir / f"objects/{K}.pkl")[K]
N, RISE, TREAD, WIDTH = 3, 0.12, 0.35, 1.5
pos0, quat0 = obj["root_pos"][0, 0].astype(float), obj["root_quat"][0, 0].astype(float)
Rw = np.zeros(9); mujoco.mju_quat2Mat(Rw, quat0); Rw = Rw.reshape(3, 3)


def surface_height(pw):
    pm = (pw - pos0) @ Rw
    x, y = pm[:, 0], pm[:, 1]; total = 2 * N * TREAD
    h = np.full(len(pw), pos0[2]); inside = np.abs(y) <= WIDTH / 2
    for i in range(N):
        cover = inside & (x >= TREAD * i) & (x <= total - TREAD * i)
        h = np.where(cover, np.maximum(h, pos0[2] + RISE * (i + 1)), h)
    return h


m = mujoco.MjModel.from_xml_path(str(ROOT / "imports/GMR/assets/unitree_g1/g1_mocap_29dof.xml")); d = mujoco.MjData(m)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
foot = {s: [i for i, n in enumerate(names) if n and n.startswith(s) and ("ankle_roll" in n or "toe" in n)] for s in ("left", "right")}
geoms = {s: [g for g in range(m.ngeom) if m.geom_bodyid[g] in b] for s, b in foot.items()}


def foot_points(s):
    pts = []
    for g in geoms[s]:
        p, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            a = m.geom_dataid[g]; vs = m.mesh_vert[m.mesh_vertadr[a]: m.mesh_vertadr[a] + m.mesh_vertnum[a]]; pts.append(p + vs @ R.T)
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            pts.append(p + np.array([[0, 0, -m.geom_size[g][0]]]))
    return np.vstack(pts)


files = sorted(glob.glob(str(rollout / "*.trajectory.pkl")))
print(f"rollout folder {rollout}: {len(files)} environments; reference = {ref_dir.name} ({len(ref['dof'])} frames @ {ref['fps']} fps)")
rp = ref["root_trans_offset"]
for env, _ in enumerate(files):
    mo = v03.load_policy_trajectory(rollout, env)
    p, q, dof, fps = mo["root_trans_offset"], mo["root_rot"], mo["dof"], mo["fps"]
    T = len(p)
    jumps = [int(i) + 1 for i in np.where(np.linalg.norm(np.diff(p, axis=0), axis=1) > 0.5)[0]]
    idx = np.clip(np.round(np.arange(T) * ref["fps"] / fps).astype(int), 0, len(rp) - 1)
    dist = np.linalg.norm(p - rp[idx], axis=1)
    gaps = np.zeros((T, 2))
    for t in range(T):
        d.qpos[0:3], d.qpos[3:7], d.qpos[7:] = p[t], q[t], dof[t]; mujoco.mj_forward(m, d)
        for k, s in enumerate(("left", "right")):
            P = foot_points(s); gaps[t, k] = (P[:, 2] - surface_height(P)).min()
    low = gaps.min(axis=1)
    print(f"\n=== env {env}: {T} frames ({T / fps:.1f} s at {fps:.0f} fps); resets at frames {jumps or 'none'}")
    print(f"root distance to reference: mean {dist.mean():.3f} m, max {dist.max():.3f} m (at frame {int(dist.argmax())}); "
          f"pelvis z {p[:, 2].min():.2f}..{p[:, 2].max():.2f} (reference {rp[:, 2].min():.2f}..{rp[:, 2].max():.2f}); final root y {p[-1, 1]:+.2f} (reference {rp[-1, 1]:+.2f})")
    print(f"lower-foot gap to stairs surface: median {np.median(low) * 100:+.1f} cm, min {low.min() * 100:+.1f}, max {low.max() * 100:+.1f}; "
          f"frames with a foot >2 cm inside a surface: {int((low < -0.02).sum())}")
    if env == 0:
        print("frame | t(s) | root y,z       | dist to ref | L gap, R gap (cm)")
        for t in list(range(0, T, 25)) + [T - 1]:
            print(f"{t:5d} | {t / fps:4.1f} | {p[t,1]:+.2f}, {p[t,2]:.2f} | {dist[t]:.3f} m    | {gaps[t,0]*100:+6.1f}, {gaps[t,1]*100:+6.1f}")
