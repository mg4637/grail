#!/usr/bin/env python
"""Feasibility checks on a retargeted G1 motion (Stage 3 output).

Checks, against the failure signature of the original grail_table retarget:
  1. pinned joints        - flat runs at a joint limit (the -1.5700 hip clamp)
  2. soft-limit violations - what SONIC's `joint_limit` reward (weight -10) costs
                             if the policy tracks the reference perfectly
  3. sole heights          - do the feet reach the floor, and the tabletop
  4. shoulder torque       - static demand vs the G1's 25 N.m shoulder limit

Usage: python data/stage3_verify_reference.py <motion_lib_dir> [<motion_lib_dir> ...]
"""
import sys, numpy as np, joblib, mujoco
import xml.etree.ElementTree as ET

ORDER = ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint",
 "left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint",
 "right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
 "waist_yaw_joint","waist_roll_joint","waist_pitch_joint","left_shoulder_pitch_joint",
 "left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint",
 "left_wrist_pitch_joint","left_wrist_yaw_joint","right_shoulder_pitch_joint",
 "right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint",
 "right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
URDF = "imports/SONIC/gear_sonic/data/assets/robot_description/urdf/g1/main_nodex.urdf"
MJCF = "imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
WEIGHT_N = 33.34 * 9.81
SHOULDER_LIMIT = 25.0

def limits():
    r = ET.parse(URDF).getroot()
    J = {j.get("name"): j for j in r.findall("joint")}
    lo = np.array([float(J[n].find("limit").get("lower")) for n in ORDER])
    hi = np.array([float(J[n].find("limit").get("upper")) for n in ORDER])
    mid, rng = (lo + hi) / 2, hi - lo
    return lo, hi, mid - 0.45 * rng, mid + 0.45 * rng   # IsaacLab soft band, factor 0.9

def report(d):
    pkl = next(iter(__import__("glob").glob(f"{d}/robot/*.pkl")))
    M = list(joblib.load(pkl).values())[0]
    dof, rt, rr = np.array(M["dof"]), np.array(M["root_trans_offset"]), np.array(M["root_rot"])
    lo, hi, slo, shi = limits()
    T = len(dof)
    print(f"\n=== {d}   ({T} frames @ {int(M['fps'])} fps) ===")

    # 1. pinned joints
    pins = []
    for j, name in enumerate(ORDER):
        at = (np.abs(dof[:, j] - lo[j]) < 1e-4) | (np.abs(dof[:, j] - hi[j]) < 1e-4)
        run = best = 0
        for v in at:
            run = run + 1 if v else 0
            best = max(best, run)
        if best >= 5:
            pins.append((name, best, dof[at, j][0] if at.any() else 0))
    print("1. pinned joints (>=5 consecutive frames exactly at a hard limit):")
    print("     none" if not pins else "")
    for n, b, v in pins:
        print(f"     {n:<26} {b:3d} frames at {v:+.4f} rad")

    # 2. soft-limit cost
    over = np.clip(dof - shi, 0, None) + np.clip(slo - dof, 0, None)
    pen = over.sum(1)
    print(f"2. soft-limit violations: {(pen>1e-6).sum()}/{T} frames | "
          f"joint_limit reward if tracked perfectly: mean {-10*pen.mean():.2f}, worst {-10*pen.max():.2f}")

    # 3 & 4. FK
    m = mujoco.MjModel.from_xml_path(MJCF); dd = mujoco.MjData(m)
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    # The SOLE, not the ankle_roll body origin. The origin sits ~30 mm above the
    # contact spheres, so reading it makes a foot 53 mm through the floor look
    # like -0.004 m and a foot resting on the floor look like it hovers at
    # +0.039 m. Each foot carries 8 collision spheres of r=5 mm.
    sole_g = {side: [g for g in range(m.ngeom) if m.geom_contype[g]
                     and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
                     == f"{side}_ankle_roll_link"] for side in ("left", "right")}
    feet, taus = [], []
    for f in range(T):
        dd.qpos[0:3], dd.qpos[3:7], dd.qpos[7:36] = rt[f], rr[f], dof[f]
        mujoco.mj_forward(m, dd)
        lf, rf = (min(dd.geom_xpos[g][2] - m.geom_rbound[g] for g in sole_g[s])
                  for s in ("left", "right"))
        feet.append((lf, rf))
        com = np.array(dd.subtree_com[0])
        hands = (dd.xpos[B("left_wrist_yaw_link")] + dd.xpos[B("right_wrist_yaw_link")]) / 2
        foot = dd.xpos[B("left_ankle_roll_link")] if lf < rf else dd.xpos[B("right_ankle_roll_link")]
        dh, df = np.linalg.norm(com[:2]-hands[:2]), np.linalg.norm(com[:2]-foot[:2])
        share = df/(dh+df) if dh+df > 1e-6 else 0.0
        per_hand = WEIGHT_N * share / 2
        t = max(per_hand * np.linalg.norm(dd.xpos[B(f"{s}_shoulder_pitch_link")][:2]
                                          - dd.xpos[B(f"{s}_wrist_yaw_link")][:2]) for s in ("left","right"))
        taus.append(t)
    feet, taus = np.array(feet), np.array(taus)
    lowest = feet.min(1)
    print(f"3. soles: lowest over clip {lowest.min():+.3f} m | on-floor phase (f0-40) "
          f"min {lowest[:40].min():+.3f} mean {lowest[:40].mean():+.3f} | "
          f"final L {feet[-1,0]:.3f} R {feet[-1,1]:.3f} (tabletop ~0.905)")
    bad = (taus > SHOULDER_LIMIT).sum()
    print(f"4. shoulder torque: peak {taus.max():.1f} N.m ({taus.argmax()=}), "
          f"{bad}/{T} frames over the {SHOULDER_LIMIT:.0f} N.m limit")

for d in sys.argv[1:]:
    report(d.rstrip("/"))
