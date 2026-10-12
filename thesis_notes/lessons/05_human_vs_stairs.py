"""Lesson 5: is the HUMAN (reconstruction) hovering above the stairs, or only the retargeted ROBOT?

Runs in the container's `sonic` env (needs torch + smplx), CPU only, no GPU:
  python docs_learning/lessons/05_human_vs_stairs.py   (cwd = GRAIL root = /workspace/grail)

It follows GRAIL's own retargeting path: G1-proportioned SMPL-X (`G1ProportionSMPLX`), forward pass with
`forward_smplx` on the parameters stored in hoi_data.pkl. Foot vertices come from GRAIL's SMPL-X segmentation
file. Then, per frame, gap = (lowest foot vertex) - (height of floor/step directly under it), in the
reconstruction's own world frame, with the stairs pose stored in the same file (obj_t, obj_R).
gap > 0 hovering, ~0 touching, < 0 penetrating.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# GRAIL root: $GRAIL_ROOT if set, else the current directory (the container is started with --pwd /workspace/grail).
# Do NOT derive it from __file__: this script is copied to different places (Mac: docs_learning/lessons, cluster: thesis/).
ROOT = Path(os.environ.get("GRAIL_ROOT", os.getcwd())).resolve()
assert (ROOT / "grail").is_dir(), f"{ROOT} does not look like the GRAIL root (no grail/ folder)"
sys.path.insert(0, str(ROOT))
from grail.models.g1_smplx_model import G1ProportionSMPLX  # noqa: E402
from grail.retargeting.retarget import _pickle_load_compat, forward_smplx, tensor_to  # noqa: E402

# optional argument: path to another hoi_data.pkl (e.g. the terrain-loss run), relative to the GRAIL root or absolute
HOI = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/generation/4dhoi_recon_smplx_valid/SynStairsV1/stairs_wide3/"
                                                         "kid_indoor1-syn-stairs_rand00001/hoi_data/hoi_data.pkl")
HOI = HOI if HOI.is_absolute() else ROOT / HOI
print("reading", HOI)
BODY_MODELS = ROOT / "imports/GEM-SMPL/inputs/checkpoints/body_models"
SEG = ROOT / "grail/models/smplx_vert_segmentation_contact.json"
N, RISE, TREAD, WIDTH = 3, 0.12, 0.35, 1.5      # stairs_wide3 dimensions (lessons/02_wide_stairs.py)

hoi = _pickle_load_compat(str(HOI))
hd = hoi["human_data"]
params = {k: hd[k] for k in ("poses", "betas", "trans", "left_hand_pose", "right_hand_pose")}
obj_R = np.asarray(hoi["obj_data"]["obj_R"][0], dtype=np.float64)     # local -> world
obj_t = np.asarray(hoi["obj_data"]["obj_t"][0], dtype=np.float64)
print("stairs pose in the reconstruction frame: t =", np.round(obj_t, 3), "\nR =\n", np.round(obj_R, 3))

g1 = G1ProportionSMPLX.from_g1_smplx_dir(str(BODY_MODELS), str(ROOT / "data/g1_smplx"))
with torch.no_grad():
    out = forward_smplx(g1.model, tensor_to(params), return_mesh=True)
V = out.vertices.detach().cpu().numpy()           # (T, 10475, 3)
pelvis = out.joints[:, 0].detach().cpu().numpy()
T = V.shape[0]

seg = json.load(open(SEG))
idx = {s: np.array(seg[f"{s}Foot"] + seg[f"{s}ToeBase"]) for s in ("left", "right")}


def surface_height(pw):
    pm = (pw - obj_t) @ obj_R                      # world -> stairs frame
    x, y = pm[:, 0], pm[:, 1]
    total = 2 * N * TREAD
    h = np.full(len(pw), obj_t[2])                 # floor = base of the stairs
    inside_y = np.abs(y) <= WIDTH / 2
    for i in range(N):
        cover = inside_y & (x >= TREAD * i) & (x <= total - TREAD * i)
        h = np.where(cover, np.maximum(h, obj_t[2] + RISE * (i + 1)), h)
    return h


gap = {s: np.zeros(T) for s in idx}; low = {s: np.zeros(T) for s in idx}; under = {s: np.zeros(T) for s in idx}
for t in range(T):
    for s, ii in idx.items():
        P = V[t, ii]; g = P[:, 2] - surface_height(P); k = int(np.argmin(g))
        gap[s][t], low[s][t], under[s][t] = g[k], P[k, 2], surface_height(P[k:k + 1])[0]

print("\nHUMAN (G1-proportioned SMPL-X) foot vs surface. frame | pelvis z | left low/under/gap | right low/under/gap | min gap")
for t in list(range(0, T, 20)) + [T - 1]:
    print(f"{t:5d} | {pelvis[t,2]:.3f} | {low['left'][t]:.3f} {under['left'][t]:.3f} {gap['left'][t]*100:+6.1f}cm"
          f" | {low['right'][t]:.3f} {under['right'][t]:.3f} {gap['right'][t]*100:+6.1f}cm"
          f" | {min(gap['left'][t], gap['right'][t])*100:+6.1f}cm")
mg = np.minimum(gap["left"], gap["right"])
print(f"\nlower-foot gap over {T} frames: median {np.median(mg)*100:+.1f} cm, mean {mg.mean()*100:+.1f}, min {mg.min()*100:+.1f}, max {mg.max()*100:+.1f}")
for thr in (0.02, 0.05, 0.10):
    print(f"frames where neither foot is within {thr*100:.0f} cm of a surface: {int((mg > thr).sum())}/{T}")
print("frames with a foot more than 2 cm inside a surface:", int((mg < -0.02).sum()))
print(f"pelvis z: start {pelvis[0,2]:.3f}, end {pelvis[-1,2]:.3f}, rise {pelvis[-1,2]-pelvis[0,2]:.3f} m (stairs height {RISE*N:.3f} m)")
