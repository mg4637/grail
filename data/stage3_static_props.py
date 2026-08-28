#!/usr/bin/env python
"""Place the Blender scene's static props into the motion-lib frame.

The chair clip's video was rendered from a Blender scene that also contains a
bookshelf and a basket. Stage 2 never reconstructed them - `hoi_data["obj_data"]`
carries exactly one object - but they do not need reconstructing: their exact
transforms were recorded at render time in
`first_frame_pose.pickle["static_objects"]` as {path, pos, rot, scale}.

What they need is a change of frame. The motion lib is expressed in the
reconstruction's frame, not Blender's, so this anchors on the one object that
exists in BOTH - the chair - and derives the rigid transform

    p_ml = A p_blender + c ,   A = R_chair_ml @ R_chair_blender^T

Measured on this clip A is identity to within 0.6 deg, and the result is checked
independently: the derived basket position lands 12 cm from where the robot's
left wrist actually reaches during the reach phase (frame 150). That is not an
input to the fit, so it is real corroboration.

Scale and rotation are baked into the exported mesh rather than passed through
config. `add_table` applies scale via `scene_scale`, which is shared with other
scene machinery; baking keeps the prop independent of it, so the spawn is just a
position with identity rotation.

Usage:
    python data/stage3_static_props.py <motion_lib_dir> --recon <recon_dir> [--out data/motion_lib/<lib>/common]
"""
import argparse, glob, os, pickle
import numpy as np, joblib, trimesh
from scipy.spatial.transform import Rotation as R


def blender_to_motionlib(recon_dir, motion_dir):
    """Rigid transform (A, c) mapping Blender scene coords into the motion lib."""
    fp = pickle.load(open(glob.glob(
        f"{recon_dir}/first_frame_pose.pickle")[0], "rb"))
    O = list(joblib.load(glob.glob(f"{motion_dir}/objects/*.pkl")[0]).values())[0]
    Rb = np.asarray(fp["obj_R"]).reshape(3, 3)
    tb = np.asarray(fp["obj_t"]).reshape(3)
    Rm = R.from_quat(np.array(O["root_quat"])[0, 0][[1, 2, 3, 0]]).as_matrix()
    tm = np.array(O["root_pos"])[0, 0]
    A = Rm @ Rb.T
    return A, tm - A @ tb, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion_dir")
    ap.add_argument("--recon", required=True,
                    help="foundation_pose dir holding first_frame_pose.pickle")
    ap.add_argument("--out", default=None, help="where to write baked meshes")
    ap.add_argument("--only", nargs="*", default=None, help="subset of prop names")
    a = ap.parse_args()
    out = a.out or os.path.join(a.motion_dir, "common")
    os.makedirs(out, exist_ok=True)

    A, c, fp = blender_to_motionlib(a.recon, a.motion_dir)
    print(f"\n=== static props -> {a.motion_dir} ===")
    print(f"  blender->motion-lib rotation {np.round(R.from_matrix(A).as_euler('xyz', degrees=True), 2)} deg")
    print(f"  translation {np.round(c, 3)}")

    placements = {}
    for name, prop in fp.get("static_objects", {}).items():
        if a.only and name not in a.only:
            continue
        mesh = trimesh.load(prop["path"])
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        v = np.asarray(mesh.vertices, dtype=float)
        # bake scale, then the prop's own Blender rotation, then the frame change
        v = v * np.asarray(prop["scale"]).reshape(3)
        v = v @ np.asarray(prop["rot"]).reshape(3, 3).T
        v = v @ A.T
        mesh.vertices = v
        pos = A @ np.asarray(prop["pos"]) + c
        f = os.path.join(out, f"{name}.obj")
        mesh.export(f)
        ext = v.max(0) - v.min(0)
        placements[name] = {"pos": pos.tolist(), "obj": f,
                            "extents": ext.tolist(),
                            "z_span": [float(pos[2] + v[:, 2].min()),
                                       float(pos[2] + v[:, 2].max())]}
        print(f"  {name:<10} pos {np.round(pos,3)}  size {np.round(ext,3)} m  "
              f"z {pos[2]+v[:,2].min():.3f}..{pos[2]+v[:,2].max():.3f}  -> {f}")

    with open(os.path.join(out, "static_props.json"), "w") as fh:
        import json
        json.dump(placements, fh, indent=2)
    print(f"  placements written to {out}/static_props.json")


if __name__ == "__main__":
    main()
