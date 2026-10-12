"""Lesson 1: MuJoCo basics. A ball is dropped next to a 3-step staircase.

Run:  docs_learning/.venv/bin/python docs_learning/lessons/01_mujoco_stairs_ball.py
"""
import mujoco
import numpy as np

# Step heights (cumulative tops) taken from GRAIL's stairs_007: 0.127, 0.266, 0.335 m.
tops = [0.127, 0.266, 0.335]
tread = 0.20  # step depth in x (metres), chosen for the lesson

# --- 1. MODEL: a static description, written in MJCF (MuJoCo's XML) -------------
# Every step is a <geom type="box">. Box "size" is the HALF-extent along x, y, z.
steps_xml = ""
for i, top in enumerate(tops):
    steps_xml += (
        f'<geom name="step{i}" type="box" '
        f'pos="{tread * (i + 0.5):.3f} 0 {top / 2:.4f}" '
        f'size="{tread / 2} 0.8 {top / 2:.4f}" rgba="0.7 0.5 0.3 1"/>\n'
    )

xml = f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    {steps_xml}
    <body name="ball" pos="0.5 0 1.0">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.05" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)

# --- 2. DATA: the dynamic state that changes every step -------------------------
data = mujoco.MjData(model)

print("nbody =", model.nbody, "| ngeom =", model.ngeom, "| nq (positions) =", model.nq, "| nv (velocities) =", model.nv)
print("A free body has 7 position numbers (xyz + quaternion) but 6 velocity numbers (3 lin + 3 ang).")

# --- 3. SIMULATE: mj_step advances physics by model.opt.timestep ----------------
for t in range(int(2.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
    if t % 250 == 0:
        print(f"t={data.time:5.2f}s  ball xyz = {np.round(data.qpos[:3], 3)}")

print("final ball z =", round(float(data.qpos[2]), 3), "(rests on a step top if ~top+0.05)")
print("contacts at end:", data.ncon)
for c in data.contact[: data.ncon]:
    print("  contact between", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
          "and", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))
