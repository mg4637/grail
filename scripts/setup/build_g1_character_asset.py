#!/usr/bin/env python3
"""
Build a static Unitree G1 robot mesh for use as a GRAIL Blender "character"
asset (grail/pipelines/blender/render_scene.py, character_name="g1_robot").

GRAIL's 2D-HOI render step only ever renders the character in a single
static pose (all motion is later hallucinated by the video-gen model), so
this script only needs to produce one plausible standing-pose mesh -- not a
rig. It forward-kinematics the Unitree G1 MJCF model
(imports/GMR/assets/unitree_g1/g1_mocap_29dof.xml) at its default (all-zero)
joint configuration, which is the robot's neutral standing pose, and merges
every visual link mesh into a single watertight-adjacent OBJ so that
grail/pipelines/blender/assets.py's load_obj_asset() -- which only keeps
`bpy.context.selected_objects[0]` after import -- sees exactly one object.

Usage:
    .venv/bin/python scripts/setup/build_g1_character_asset.py
"""

import os

import mujoco
import trimesh

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G1_XML = os.path.join(_REPO_ROOT, "imports/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
OUT_DIR = os.path.join(_REPO_ROOT, "data/characters/g1_robot")
OUT_OBJ = os.path.join(OUT_DIR, "g1_robot.obj")

# Matte dark grey/black, matching the real G1's dominant plastic/metal finish
# (the MJCF's own "dark" material is rgba 0.2 0.2 0.2; we go flat single-tone
# since merging into one Blender-importable object means one material).
ROBOT_DIFFUSE_RGBA = [38, 38, 40, 255]


def main():
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # zero qpos -> G1's neutral standing pose

    parts = []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_group[geom_id] != 1:
            # The MJCF defines two geoms per link (a group=1 "visual" copy
            # and an ungrouped collision copy using the same full mesh,
            # since this is the "custom_collision" URDF variant) -- keep
            # only the visual one to avoid doubling every mesh.
            continue

        mesh_id = model.geom_dataid[geom_id]
        v_start, v_count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        f_start, f_count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]

        verts_local = model.mesh_vert[v_start : v_start + v_count]
        faces = model.mesh_face[f_start : f_start + f_count]

        xpos = data.geom_xpos[geom_id]
        xmat = data.geom_xmat[geom_id].reshape(3, 3)
        verts_world = verts_local @ xmat.T + xpos

        parts.append(trimesh.Trimesh(vertices=verts_world, faces=faces, process=False))

    if not parts:
        raise RuntimeError("No visual mesh geoms found -- check geom_group filter")
    print(f"Merging {len(parts)} link meshes...")

    merged = trimesh.util.concatenate(parts)
    merged.visual = trimesh.visual.texture.TextureVisuals(
        material=trimesh.visual.material.SimpleMaterial(
            diffuse=ROBOT_DIFFUSE_RGBA,
            ambient=ROBOT_DIFFUSE_RGBA,
            specular=[70, 70, 70, 255],
            glossiness=0.4,
        )
    )

    extents = merged.bounding_box.extents
    print(f"Merged mesh: {len(merged.vertices)} verts, {len(merged.faces)} faces")
    print(f"Bounding box extents (m): {extents} (expect ~0.7 x 0.5 x 1.3 for standing G1)")

    os.makedirs(OUT_DIR, exist_ok=True)
    merged.export(OUT_OBJ, file_type="obj")
    print(f"Wrote {OUT_OBJ}")


if __name__ == "__main__":
    main()
