"""Lesson 9: what do the REAL contact forces of a policy rollout say?  (reads <rollout>/forces.npz, no window)

  docs_learning/.venv/bin/python docs_learning/lessons/09_forces_report.py <rollout_folder>

forces.npz is written by docs_learning/cluster/term_logger.py during the Isaac Sim evaluation:
  foot_net_force_w   (T, envs, 2, 3)  net contact force on the left/right ankle link (world frame, N)
  foot_on_terrain_w  (T, envs, 2, 3)  force exerted ON THE STAIRS by each foot (N); reaction on the robot = -that
  robot_mass_kg      total mass seen by PhysX
Physics check: when the robot stands still on the floor or on the top step, the summed VERTICAL force on the feet
must be close to its weight (mass * 9.81). During the climb it fluctuates (accelerations, one foot swinging).
"""
import sys
from pathlib import Path

import numpy as np

rollout = Path(sys.argv[1])
z = np.load(rollout / "forces.npz", allow_pickle=True)
net = z["foot_net_force_w"]                      # (T, E, 2, 3)
T, E = net.shape[:2]
fps = float(z["fps"]); mass = float(z["robot_mass_kg"]) if "robot_mass_kg" in z.files else float("nan")
W = mass * 9.81
print(f"{rollout}: {T} frames @ {fps:.0f} fps, {E} envs; robot mass {mass:.1f} kg -> weight {W:.0f} N; sensors: net_force on feet" +
      ("; feet->terrain force matrix" if "foot_on_terrain_w" in z.files else "; (no feet->terrain sensor in file)"))
names = [str(n) for n in z["foot_body_names"]]; print("foot bodies:", names)
for e in range(E):
    vz = net[:, e, :, 2].sum(axis=1)                         # summed vertical force on both feet, N
    print(f"\n=== env {e}: summed vertical foot force  (weight = {W:.0f} N)")
    for lab, a, b in (("standing on floor, 0.2-0.8 s", 5, 20), ("approach 2-3 s", 50, 75), ("climbing 5-8 s", 125, 200), ("standing on top, 8.6-9.6 s", 215, 240)):
        seg = vz[a:b]; print(f"  {lab:30s} mean {seg.mean():7.1f} N  (= {seg.mean() / W * 100:5.1f}% of weight), min {seg.min():7.1f}, max {seg.max():7.1f}")
    L = np.linalg.norm(net[:, e, 0], axis=1); R = np.linalg.norm(net[:, e, 1], axis=1)
    print(f"  frames with |force| > 10 N: left foot {int((L > 10).sum())}/{T}, right foot {int((R > 10).sum())}/{T}; frames with NO foot force > 10 N (airborne): {int(((L <= 10) & (R <= 10)).sum())}")
    if "foot_on_terrain_w" in z.files:
        ft = z["foot_on_terrain_w"][:, e]                    # force on the stairs from each foot
        tz = -ft[:, :, 2].sum(axis=1)                        # reaction on the robot = -force on the stairs
        n_on = int(((np.linalg.norm(ft, axis=2) > 10).any(axis=1)).sum())
        print(f"  feet->stairs sensor: frames with a foot pressing on the stairs (>10 N): {n_on}/{T}; "
              f"max vertical reaction from the stairs {tz.max():.0f} N at frame {int(tz.argmax())}")
