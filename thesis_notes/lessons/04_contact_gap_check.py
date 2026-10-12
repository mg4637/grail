"""Lesson 4: measure the gap between each foot and the surface beneath it, per frame (kinematic, no physics).

gap = (lowest point of the foot geometry) - (height of floor/step directly under that point)
  gap > 0 : foot hovers      gap ~ 0 : foot touches      gap < 0 : foot is inside the stairs/floor
Run:  docs_learning/.venv/bin/python docs_learning/lessons/04_contact_gap_check.py
"""
import sys
from pathlib import Path
import joblib, mujoco, numpy as np

ROOT = Path(__file__).resolve().parents[2]
KEY = "stairs_wide3_kid_indoor1-syn-stairs_rand00001"
MD = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/motion_lib/stairs_wide3")   # optional: another motion_lib folder
print("motion folder:", MD)
N, RISE, TREAD, WIDTH = 3, 0.12, 0.35, 1.5          # stairs_wide3 (lessons/02_wide_stairs.py)

motion = joblib.load(MD / f"robot/{KEY}.pkl")[KEY]
obj = joblib.load(MD / f"objects/{KEY}.pkl")[KEY]
pos0, q = obj["root_pos"][0, 0].astype(float), obj["root_quat"][0, 0].astype(float)   # q = w,x,y,z
Rw = np.zeros(9); mujoco.mju_quat2Mat(Rw, q); Rw = Rw.reshape(3, 3)                  # mesh->world rotation

def surface_height(pw):
    """height of floor/stairs under world points pw (n,3); floor is z=0, stairs sit at pos0 with rotation Rw."""
    pm = (pw - pos0) @ Rw                              # world -> stairs frame (inverse rotation)
    x, y = pm[:, 0], pm[:, 1]; total = 2 * N * TREAD
    h = np.zeros(len(pw)); inside_y = np.abs(y) <= WIDTH / 2
    for i in range(N):
        cover = inside_y & (x >= TREAD * i) & (x <= total - TREAD * i)
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
            pts.append((p + np.array([[0, 0, -m.geom_size[g][0]]]))); 
    return np.vstack(pts)

T = len(motion["dof"]); gap = {s: np.zeros(T) for s in foot}; low = {s: np.zeros(T) for s in foot}; under = {s: np.zeros(T) for s in foot}
for t in range(T):
    d.qpos[0:3] = motion["root_trans_offset"][t]; d.qpos[3:7] = motion["root_rot"][t]; d.qpos[7:] = motion["dof"][t]; mujoco.mj_forward(m, d)
    for s in foot:
        P = foot_points(s); g = P[:, 2] - surface_height(P); k = int(np.argmin(g))
        gap[s][t], low[s][t], under[s][t] = g[k], P[k, 2], surface_height(P[k:k+1])[0]
pelvis = motion["root_trans_offset"]
print("frame | pelvis y,z      | left: low  under  gap | right: low  under  gap | min gap")
for t in list(range(0, T, 20)) + [T - 1]:
    print(f"{t:5d} | {pelvis[t,1]:+.2f}, {pelvis[t,2]:.2f} | {low['left'][t]:+.3f} {under['left'][t]:.3f} {gap['left'][t]*100:+6.1f}cm | {low['right'][t]:+.3f} {under['right'][t]:.3f} {gap['right'][t]*100:+6.1f}cm | {min(gap['left'][t], gap['right'][t])*100:+6.1f}cm")
mg = np.minimum(gap["left"], gap["right"])
print(f"\nsupporting-foot gap (min of both feet) over all frames: median {np.median(mg)*100:+.1f} cm, mean {mg.mean()*100:+.1f}, min {mg.min()*100:+.1f}, max {mg.max()*100:+.1f}")
for thr in (0.02, 0.05, 0.10):
    print(f"frames where NEITHER foot is within {thr*100:.0f} cm of a surface (airborne): {int((mg > thr).sum())}/{T}")
print("frames with some foot > 2cm INSIDE a surface:", int((np.minimum(gap['left'], gap['right']) < -0.02).sum()))
np.save(ROOT / f"docs_learning/frames/gap_min_{MD.name}.npy", mg)
