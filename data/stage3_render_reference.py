#!/usr/bin/env python
"""Render a retargeted motion_lib pkl to mp4, in MuJoCo, on the CPU-side GL.

The repo's own renderer (grail/visualization/batch_render_replay.py) needs an
IsaacSim session and therefore the GPU, which is not available while a Stage 4
run is training. This uses the same MuJoCo scene as the Stage 3 checks - G1 MJCF
plus the table baked to its reference pose - so what you see is exactly what the
verification numbers were measured on.

This is KINEMATIC playback: qpos is written straight from the pkl each frame and
mj_forward runs FK only. No physics, no controller. It shows what the reference
asks for, which is the point - including a foot 53 mm below the floor.

Needs a reachable X display; the EGL path is broken on this host (PyOpenGL is
pinned at 3.1.0, which has no EGLDeviceEXT) and osmesa is not installed:
    MUJOCO_GL=glfw DISPLAY=:3 python data/stage3_render_reference.py <dir> ...

Usage:
    python data/stage3_render_reference.py data/motion_lib/grail_table_v3f -o out.mp4
    python data/stage3_render_reference.py A B --labels "before" "after" -o ab.mp4
"""
import argparse, glob, os
import numpy as np, joblib, mujoco, imageio.v2 as imageio
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "feas", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "stage3_mujoco_feasibility.py"))
_feas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feas)


def load(spec, scene_dir=None):
    """`spec` is a motion_lib dir, or `traj:<path>.trajectory.pkl` from a SONIC
    eval. Trajectory pkls carry no scene, so they borrow `scene_dir`'s table -
    which is correct, since the policy was rolled out in exactly that scene."""
    if spec.startswith("traj:"):
        import pickle
        with open(spec[5:], "rb") as fh:
            T = pickle.load(fh)
        m = mujoco.MjModel.from_xml_path(
            _feas.build_scene(glob.glob(f"{scene_dir}/objects/*.pkl")[0]))
        assert T.get("quat_format", "wxyz") == "wxyz", T.get("quat_format")
        dof = np.asarray(T["dof_pos"])[:, isaac_to_mjcf(m)]
        return (m, mujoco.MjData(m),
                (np.asarray(T["root_pos_w"]), np.asarray(T["root_quat_w"]), dof),
                float(T["fps"]))
    M = list(joblib.load(glob.glob(f"{spec}/robot/*.pkl")[0]).values())[0]
    m = mujoco.MjModel.from_xml_path(
        _feas.build_scene(glob.glob(f"{spec}/objects/*.pkl")[0]))
    return m, mujoco.MjData(m), (np.array(M["root_trans_offset"]),
                                 np.array(M["root_rot"]),
                                 np.array(M["dof"])), float(M["fps"])


# IsaacLab's articulation joint order for the G1 (g1_29dof_textured.usd), dumped
# from robot.data.joint_names in the container. PhysX parses the articulation
# BREADTH-FIRST, while the MJCF lists joints DEPTH-FIRST (whole left leg, whole
# right leg, waist, arms). The two agree only at index 0. SONIC's
# TrajectoryRecorder stores robot.data.joint_pos verbatim and saves no joint
# names, so writing dof_pos straight into MuJoCo's qpos[7:36] silently renders a
# scrambled robot. The repo's own renderer avoids this by replaying into an
# IsaacLab articulation, where the order matches by construction.
ISAAC_JOINT_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def isaac_to_mjcf(model):
    """Index array mapping an IsaacLab-ordered dof vector into MJCF qpos order."""
    mj_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                for j in range(1, model.njnt)]          # skip the free joint
    missing = set(mj_names) - set(ISAAC_JOINT_ORDER)
    if missing:
        raise RuntimeError(f"MJCF joints absent from the IsaacLab order: {missing}")
    return np.array([ISAAC_JOINT_ORDER.index(n) for n in mj_names])


_FONT = None


def label(img, text, h=30):
    """Burn a caption into a dark strip at the top of the frame."""
    global _FONT
    from PIL import Image, ImageDraw, ImageFont
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 17)
        except OSError:
            _FONT = ImageFont.load_default()
    img = img.copy()
    img[:h] = (img[:h] * 0.25).astype(img.dtype)
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((8, 6), text, fill=(255, 255, 255), font=_FONT)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion_dirs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--azimuth", type=float, default=140.0)
    ap.add_argument("--elevation", type=float, default=-12.0)
    ap.add_argument("--distance", type=float, default=4.2)
    ap.add_argument("--fps_out", type=float, default=24.0)
    ap.add_argument("--lookat_z", type=float, default=0.75,
                    help="camera target height; 0.0 puts the floor line on screen")
    ap.add_argument("--track", default="root", choices=["root", "feet"],
                    help="what the camera follows")
    ap.add_argument("--scene_dir", default=None,
                    help="motion_lib dir supplying the table for any traj: panel")
    ap.add_argument("--pad", action="store_true",
                    help="hold the last frame of shorter panels instead of "
                         "cutting every panel to the shortest. Use this when a "
                         "policy rollout terminated early - cutting would hide "
                         "the very truncation you want to see.")
    ap.add_argument("--track_panel", type=int, default=0,
                    help="which panel's motion the shared camera follows")
    ap.add_argument("--fixed_lookat", type=float, nargs=3, default=None,
                    help="pin the camera target instead of following the robot. "
                         "Use this for reference-vs-policy: if each panel chases "
                         "its own root the two views drift apart exactly when "
                         "they diverge, which is the moment you want to compare.")
    a = ap.parse_args()

    scenes = [load(d, a.scene_dir) for d in a.motion_dirs]
    names = a.labels or [os.path.basename(d.rstrip("/")).replace("traj:", "")
                         for d in a.motion_dirs]
    lens = [len(s[2][0]) for s in scenes]
    T = max(lens) if a.pad else min(lens)

    # Panels can run at different rates - references are 24 fps, SONIC eval
    # trajectories are 25 - so index each panel by TIME, not by frame number,
    # or the two drift apart over the clip.
    base_fps = scenes[a.track_panel][3]

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = a.azimuth, a.elevation, a.distance

    frames = []
    renderers = [mujoco.Renderer(m, a.height, a.width) for m, _, _, _ in scenes]
    for f in range(T):
        panels = []
        for (m, d, (rt, rr, dof), fps), r, nm in zip(scenes, renderers, names):
            g = min(int(round(f * fps / base_fps)), len(rt) - 1)
            if not a.pad:
                g = min(g, T - 1)
            d.qpos[0:3], d.qpos[3:7], d.qpos[7:36] = rt[g], rr[g], dof[g]
            mujoco.mj_forward(m, d)
            if a.fixed_lookat is not None:
                cam.lookat[:] = a.fixed_lookat
            elif a.track == "feet":
                B = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
                C = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
                ctr = 0.5 * (d.xpos[B] + d.xpos[C])
                cam.lookat[:] = [ctr[0], ctr[1], ctr[2] + a.lookat_z]
            else:
                cam.lookat[:] = [rt[g][0], rt[g][1], a.lookat_z]
            r.update_scene(d, cam)
            # only flag a panel that genuinely ran out of frames early
            tag = "  [TERMINATED]" if (a.pad and g == len(rt) - 1
                                       and len(rt) / fps < 0.95 * T / base_fps) else ""
            panels.append(label(r.render(), f"{nm}  f{f:03d}{tag}"))
        frames.append(np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    imageio.mimsave(a.out, frames, fps=a.fps_out, quality=8, macro_block_size=1)
    print(f"wrote {a.out}  ({T} frames, {len(scenes)} panel(s), "
          f"{frames[0].shape[1]}x{frames[0].shape[0]})")


if __name__ == "__main__":
    main()
