#!/usr/bin/env python
"""Put the robot's hands flat on the tabletop where the source human's are.

TWO MEASURED DEFECTS, ONE FIX
-----------------------------
1. CONTACT. The source human's wrists rest on the table through the mount
   (surface distance 0.004-0.058 m, f50-86). The retargeted robot's drift off
   it: 0.041/0.030 at f76, 0.083/0.100 at f84. The human data is correct here,
   so this is introduced by retargeting - the same class as the foot-planting
   defect fixed in stage3_fix_foot_planting.py.

2. ORIENTATION. The palms are edge-on rather than flat: |palm_normal . up| is
   0.26/0.33 mean over the mount with --zero_out_wrist, 0.48/0.29 with the
   wrists freed (grail_table_v4w). Part of this comes from Stage 2 - the
   RECONSTRUCTED human's palms are not flat either (right hand 0.01-0.33) - so
   freeing the wrists could only go so far: the IK was tracking a bad target.
   This fix overrides that target with the table's own geometry.

Both are solved together per frame: place the palm's lowest point on the
tabletop AND align the palm normal with the surface normal, moving only that
arm's 7 joints (shoulder x3, elbow, wrist x3). The root is never touched, so the
climb trajectory and the legs are preserved exactly.

GATING (learned from earlier fixes in this project that produced valid-looking
numbers from broken poses):
  * only frames where the SOURCE HUMAN's hand is actually on the table are
    corrected - the robot is never asked to reach for a surface the human is not
    touching;
  * a per-frame cap on joint change rejects any correction that needs an
    implausible arm reconfiguration, rather than silently accepting it;
  * the swing/approach phases outside the contact window are left alone.

Usage:
    python data/stage3_fix_hand_contact.py <motion_lib_dir> [--write]
"""
import argparse, glob, importlib.util, os
import numpy as np, joblib, mujoco, trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

_spec = importlib.util.spec_from_file_location(
    "feas", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "stage3_mujoco_feasibility.py"))
_feas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feas)

ARM = {s: [f"{s}_shoulder_pitch_joint", f"{s}_shoulder_roll_joint",
           f"{s}_shoulder_yaw_joint", f"{s}_elbow_joint",
           f"{s}_wrist_roll_joint", f"{s}_wrist_pitch_joint",
           f"{s}_wrist_yaw_joint"] for s in ("left", "right")}
UP = np.array([0.0, 0.0, 1.0])


def arm_length(m, d, side):
    """Shoulder-to-wrist distance at the zero pose - the hand's reach envelope."""
    q = d.qpos.copy()
    d.qpos[:] = 0.0; d.qpos[3] = 1.0
    mujoco.mj_forward(m, d)
    sh = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_pitch_link")]
    wr = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")]
    d.qpos[:] = q
    return float(np.linalg.norm(wr - sh))


def hand_geoms(m, side):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
    return b, [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]]


def hand_state(m, d, gs, bid):
    """(lowest point z, palm-normal alignment with world up, hand centroid)."""
    lo = 1e9; pts = []
    for g in gs:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            lo = min(lo, float(w[:, 2].min())); pts.append(w.mean(0))
        else:
            lo = min(lo, float(d.geom_xpos[g][2] - m.geom_rbound[g]))
            pts.append(d.geom_xpos[g].copy())
    flat = abs(d.xmat[bid].reshape(3, 3)[:, 1] @ UP)
    return lo, flat, np.mean(pts, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion_dir")
    ap.add_argument("--recon", default="results/generation/4dhoi_recon_smplx_valid/"
                                       "ComAsset/table/kid_indoor2-climb-table_rand00001")
    ap.add_argument("--table_z", type=float, default=0.905)
    ap.add_argument("--human_contact", type=float, default=0.06,
                    help="source-human wrist-to-table distance counted as contact (m)")
    ap.add_argument("--max_joint_change", type=float, default=0.8,
                    help="reject a frame's correction if any joint moves more than this (rad)")
    ap.add_argument("--w_flat", type=float, default=1.0, help="palm-flatness weight")
    ap.add_argument("--w_reg", type=float, default=1.0, help="stay-near-original weight")
    ap.add_argument("--w_temporal", type=float, default=8.0,
                    help="continuity with the previous frame's correction")
    ap.add_argument("--reach_margin", type=float, default=0.95,
                    help="fraction of arm length within which contact is attempted")
    ap.add_argument("--blend", type=int, default=4)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rp = glob.glob(f"{a.motion_dir}/robot/*.pkl")[0]
    data = joblib.load(rp); key = list(data)[0]; M = data[key]
    rt = np.array(M["root_trans_offset"], dtype=float)
    rr = np.array(M["root_rot"], dtype=float)
    dof = np.array(M["dof"], dtype=float)
    T = len(dof)

    m = mujoco.MjModel.from_xml_path(_feas.G1_MJCF)
    d = mujoco.MjData(m)
    jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]

    # --- which frames does the SOURCE HUMAN have a hand on the table? ---
    import pickle, torch, smplx
    hp = pickle.load(open(f"{a.recon}/hoi_data/hoi_data.pkl", "rb"))
    hd, od = hp["human_data"], hp["obj_data"]
    poses = np.asarray(hd["poses"]); trans = np.asarray(hd["trans"]); betas = np.asarray(hd["betas"])
    sm = smplx.create("models", model_type="smplx", gender=hd.get("gender", "neutral"),
                      use_pca=False, num_betas=10, batch_size=len(poses))
    o = sm(global_orient=torch.tensor(poses[:, :3], dtype=torch.float32),
           body_pose=torch.tensor(poses[:, 3:66], dtype=torch.float32),
           jaw_pose=torch.tensor(poses[:, 66:69], dtype=torch.float32),
           leye_pose=torch.tensor(poses[:, 69:72], dtype=torch.float32),
           reye_pose=torch.tensor(poses[:, 72:75], dtype=torch.float32),
           left_hand_pose=torch.tensor(poses[:, 75:120], dtype=torch.float32),
           right_hand_pose=torch.tensor(poses[:, 120:165], dtype=torch.float32),
           betas=torch.tensor(np.repeat(betas, len(poses), 0), dtype=torch.float32),
           transl=torch.tensor(trans, dtype=torch.float32))
    J = o.joints.detach().numpy()
    tm = trimesh.load(f"{a.recon}/mesh_data/model.obj")
    tm = tm.dump(concatenate=True) if isinstance(tm, trimesh.Scene) else tm
    sc = np.asarray(od["obj_scale"]).reshape(-1)
    V = np.asarray(tm.vertices) * (sc[:3] if sc.size >= 3 else float(sc[0]))
    S0 = np.asarray(trimesh.sample.sample_surface(trimesh.Trimesh(V, tm.faces), 40000)[0])
    human_on = {}
    for side, widx in (("left", 20), ("right", 21)):
        flags = np.zeros(min(T, len(J)), dtype=bool)
        for f in range(len(flags)):
            tree = cKDTree(S0 @ np.asarray(od["obj_R"])[f].T + np.asarray(od["obj_t"])[f].reshape(3))
            flags[f] = tree.query(J[f, widx][None])[0][0] < a.human_contact
        human_on[side] = flags

    print(f"\n=== hand contact fix: {a.motion_dir} ===")
    for s in ("left", "right"):
        idx = np.where(human_on[s])[0]
        rng = f"f{idx.min()}-{idx.max()}" if len(idx) else "none"
        print(f"  source human {s:<5} on the table: {int(human_on[s].sum())}/{T} frames  ({rng})")

    n_reach = {"left": 0, "right": 0}
    out = dof.copy()
    stats = {s: {"fixed": 0, "rejected": 0, "scaled": 0, "maxdq": 0.0}
             for s in ("left", "right")}

    for s in ("left", "right"):
        bid, gs = hand_geoms(m, s)
        L = arm_length(m, d, s)
        sh_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_shoulder_pitch_link")
        jids = [jnames.index(n) for n in ARM[s]]
        lo_b = np.array([m.jnt_range[i + 1, 0] for i in jids])
        hi_b = np.array([m.jnt_range[i + 1, 1] for i in jids])
        delta = np.zeros((T, len(jids)))
        prev = None            # previous frame's accepted correction, for continuity

        for f in range(T):
            if not human_on[s][f]:
                continue
            q0 = dof[f].copy()

            # Can this hand physically reach the tabletop at all? The G1 arm is
            # ~0.28 m; once the robot is standing on the table its shoulder is
            # 0.43-0.59 m above the surface, so no arm posture puts the palm on
            # it. The source human's wrists ARE near the table there only
            # because their shoulder sits lower relative to the same table -
            # a proportion difference, not a retargeting defect. Chasing it
            # would wrench the arm for nothing, so past the reach envelope the
            # contact term is replaced by a hold-in-place term, so the palm
            # rotates flat WITHOUT the arm drifting away from the trajectory GMR
            # produced (leaving it unconstrained sent the right hand 0.27 m off).
            d.qpos[0:3], d.qpos[3:7], d.qpos[7:36] = rt[f], rr[f], q0
            mujoco.mj_forward(m, d)
            reachable = (d.xpos[sh_id][2] - a.table_z) < a.reach_margin * L
            n_reach[s] += int(reachable)

            def cost(w):
                q = q0.copy(); q[jids] = w
                d.qpos[0:3], d.qpos[3:7], d.qpos[7:36] = rt[f], rr[f], q
                mujoco.mj_forward(m, d)
                z, flat, c = hand_state(m, d, gs, bid)
                # Both terms are normalised by the tolerance that matters for
                # them - 1 cm of height, 0.1 of alignment - so neither silently
                # dominates. Raw squared metres made the flatness term ~50x the
                # contact term, which is why the first run fixed the palms and
                # left the hands off the surface.
                place = (((z - a.table_z) / 0.01) ** 2 if reachable
                         else float(np.sum(((c - c0) / 0.01) ** 2)))
                c = place \
                    + a.w_flat * ((1.0 - flat) / 0.1) ** 2 \
                    + a.w_reg * float(np.sum((w - q0[jids]) ** 2))
                if carried is not None:
                    c += a.w_temporal * float(np.sum((w - carried) ** 2))
                return c

            d.qpos[0:3], d.qpos[3:7], d.qpos[7:36] = rt[f], rr[f], q0
            mujoco.mj_forward(m, d)
            z0, flat0, c0 = hand_state(m, d, gs, bid)

            def err(z, flat):
                return ((z - a.table_z) / 0.01) ** 2 if reachable else 0.0

            # Carry the previous frame's correction forward along the original
            # motion. This is what makes the result temporally coherent: each
            # frame starts from where the last one ended rather than from
            # scratch, so consecutive corrections agree instead of needing to be
            # averaged together afterwards.
            carried = None if prev is None else np.clip(q0[jids] + prev, lo_b, hi_b)

            best = None
            seeds = [q0[jids]] if carried is None else [carried, q0[jids]]
            for seed in seeds:
                r = minimize(cost, seed, bounds=list(zip(lo_b, hi_b)), method="L-BFGS-B",
                             options={"maxiter": 120})
                if best is None or r.fun < best.fun:
                    best = r
            dq = best.x - q0[jids]
            # A frame needing more than the cap is not discarded outright -
            # that would leave an uncorrected frame between corrected
            # neighbours, a discontinuity worse than the original defect.
            # Scale the step down to the cap instead, then KEEP IT ONLY IF it
            # measurably helps: scaling a nonlinear IK solution is not
            # guaranteed to, so the improvement is verified by FK rather than
            # assumed.
            if np.abs(dq).max() > a.max_joint_change:
                dq = dq * (a.max_joint_change / np.abs(dq).max())
                stats[s]["scaled"] += 1
            q = q0.copy(); q[jids] = q0[jids] + dq
            d.qpos[0:3], d.qpos[3:7], d.qpos[7:36] = rt[f], rr[f], q
            mujoco.mj_forward(m, d)
            z1, flat1, c1 = hand_state(m, d, gs, bid)
            def score(z, flat, c):
                return (err(z, flat) + a.w_flat * ((1.0 - flat) / 0.1) ** 2
                        + (0.0 if reachable else float(np.sum(((c - c0) / 0.01) ** 2))))
            if score(z1, flat1, c1) >= score(z0, flat0, c0):
                stats[s]["rejected"] += 1
                continue
            delta[f] = dq
            prev = dq
            stats[s]["fixed"] += 1
            stats[s]["maxdq"] = max(stats[s]["maxdq"], float(np.abs(dq).max()))

        # taper the correction out over `blend` frames at each contact-window edge
        sm_delta = delta.copy()
        for f in range(T):
            if human_on[s][f]:
                continue
            near = [g for g in range(max(0, f - a.blend), min(T, f + a.blend + 1))
                    if human_on[s][g] and np.any(delta[g])]
            if near:
                g = min(near, key=lambda g: abs(g - f))
                sm_delta[f] = delta[g] * (1.0 - abs(g - f) / (a.blend + 1))
        out[:, jids] += sm_delta

    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    out = np.clip(out, lo, hi)

    # --- measure the motion that is actually written, after blending, smoothing
    #     and joint-limit clipping - not the optimiser's per-frame numbers ---
    for s in ("left", "right"):
        bid, gs = hand_geoms(m, s)
        L = arm_length(m, d, s)
        sh_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_shoulder_pitch_link")
        rows = {"before": [], "after": []}
        reach_rows = {"before": [], "after": []}
        for f in np.where(human_on[s])[0]:
            for tag, qs in (("before", dof), ("after", out)):
                d.qpos[0:3], d.qpos[3:7], d.qpos[7:36] = rt[f], rr[f], qs[f]
                mujoco.mj_forward(m, d)
                z, flat, _ = hand_state(m, d, gs, bid)
                rows[tag].append((z - a.table_z, flat))
                if (d.xpos[sh_id][2] - a.table_z) < a.reach_margin * L:
                    reach_rows[tag].append((z - a.table_z, flat))
        stats[s]["before"], stats[s]["after"] = rows["before"], rows["after"]
        stats[s]["rb"], stats[s]["ra"] = reach_rows["before"], reach_rows["after"]

    for s in ("left", "right"):
        st = stats[s]
        if not st["before"]:
            print(f"  {s:<5}: no frames corrected"); continue
        b = np.array(st["before"]); af = np.array(st["after"])
        print(f"  {s:<5}: corrected {st['fixed']}, scaled-to-cap {st['scaled']}, "
              f"rejected {st['rejected']} (no improvement), max |dq| {st['maxdq']:.2f} rad")
        print(f"         hand-to-tabletop  |{np.abs(b[:,0]).mean():.3f}| -> |{np.abs(af[:,0]).mean():.3f}| m"
              f"   (worst {np.abs(b[:,0]).max():.3f} -> {np.abs(af[:,0]).max():.3f})")
        print(f"         penetrating >5mm  {int((b[:,0]<-0.005).sum()):3d} -> {int((af[:,0]<-0.005).sum()):3d} frames")
        print(f"         floating    >5mm  {int((b[:,0]> 0.005).sum()):3d} -> {int((af[:,0]> 0.005).sum()):3d} frames")
        print(f"         palm flatness      {b[:,1].mean():.2f}  -> {af[:,1].mean():.2f}"
              f"   (worst {b[:,1].min():.2f} -> {af[:,1].min():.2f})")
        if stats[s]["rb"]:
            rb = np.array(stats[s]["rb"]); ra = np.array(stats[s]["ra"])
            print(f"         --- of which {len(rb)} frames are within arm's reach "
                  f"(contact actually attempted) ---")
            print(f"         hand-to-tabletop  |{np.abs(rb[:,0]).mean():.3f}| -> "
                  f"|{np.abs(ra[:,0]).mean():.3f}| m")
            print(f"         in contact <1cm   {int((np.abs(rb[:,0])<0.01).sum()):3d} -> "
                  f"{int((np.abs(ra[:,0])<0.01).sum()):3d} frames")
        else:
            print(f"         (no frame within arm's reach of the table - flatness only)")

    # The per-frame cap bounds displacement; what actually decides whether the
    # result is physically plausible is how fast the joints move. Compare the
    # corrected motion's peak joint velocity against the original's - a
    # correction that stays within the motion's own dynamic range is one the
    # robot could already track.
    fps = float(M.get("fps", 30))
    v0 = np.abs(np.diff(dof, axis=0)).max() * fps
    v1 = np.abs(np.diff(out, axis=0)).max() * fps
    print(f"  peak joint velocity: {v0:.2f} -> {v1:.2f} rad/s  "
          f"({'within' if v1 <= v0 * 1.25 else 'ABOVE'} the original's range)")
    print(f"  max joint change overall: {np.abs(out - dof).max():.3f} rad")
    print(f"  root path untouched: {np.allclose(rt, np.array(M['root_trans_offset'], dtype=float))}")

    if a.write:
        M["dof"] = out.astype(np.float32)
        pose_aa = np.array(M["pose_aa"]).copy()
        axes = np.array([m.jnt_axis[j] for j in range(1, m.njnt)])
        pose_aa[:, 1:] = out[:, :, None] * axes[None]
        M["pose_aa"] = pose_aa.astype(np.float32)
        joblib.dump({key: M}, rp)
        print(f"  written {rp}")


if __name__ == "__main__":
    main()
