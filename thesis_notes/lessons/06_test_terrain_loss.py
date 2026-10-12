"""Lesson 6: test the new `terrain_support` loss BEFORE spending GPU time on it.

Runs in the container's `grail` env (needs pytorch3d), CPU only. cwd must be the GRAIL root.
It rebuilds the optimizer's inputs from the existing reconstruction (hoi_data.pkl), calls the real
`LossComputer.terrain_foot_gaps` / `_terrain_support_loss`, and checks
  1. the gaps agree with the independent measurement of lessons/05_human_vs_stairs.py (median lower-foot gap ~ +5.4 cm)
  2. the gradient w.r.t. the foot height pushes hovering feet DOWN (loss grows with height when gap > 0)
  3. an artificial "perfectly grounded" body gives a loss near zero.
"""
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import trimesh

ROOT = Path(os.environ.get("GRAIL_ROOT", os.getcwd())).resolve()
assert (ROOT / "grail").is_dir(), f"{ROOT} is not the GRAIL root"
sys.path.insert(0, str(ROOT))
from grail.models.g1_smplx_model import G1ProportionSMPLX  # noqa: E402
from grail.models.smplx_model import get_smplx_verts_segment  # noqa: E402
from grail.optimization.loss_computer import LossComputer  # noqa: E402

REC = ROOT / "results/generation/4dhoi_recon_smplx_valid/SynStairsV1/stairs_wide3/kid_indoor1-syn-stairs_rand00001"
# NOTE: do not import grail.retargeting.retarget here: it imports `mink`, which exists only in the `sonic` env.
with open(REC / "hoi_data/hoi_data.pkl", "rb") as f:
    hoi = pickle.load(f)
hd = hoi["human_data"]
params = {k: torch.as_tensor(np.asarray(hd[k])) for k in ("poses", "betas", "trans", "left_hand_pose", "right_hand_pose")}
g1 = G1ProportionSMPLX.from_g1_smplx_dir(str(ROOT / "imports/GEM-SMPL/inputs/checkpoints/body_models"), str(ROOT / "data/g1_smplx"))
with torch.no_grad():
    verts = g1._forward_smplx(params, return_mesh=True).vertices.detach().float()   # (T, V, 3), same math as retarget.forward_smplx
T = verts.shape[0]

mesh = trimesh.load(str(REC / "mesh_data/model.obj"), force="mesh", process=False)
R0 = torch.tensor(np.asarray(hoi["obj_data"]["obj_R"][0]), dtype=torch.float32)
t0 = torch.tensor(np.asarray(hoi["obj_data"]["obj_t"][0]), dtype=torch.float32)

data = SimpleNamespace(obj=SimpleNamespace(verts=torch.tensor(mesh.vertices, dtype=torch.float32),
                                           faces=torch.tensor(mesh.faces, dtype=torch.long)))


class HM:  # minimal stand-in for the optimizer's human model: only what the loss calls
    @staticmethod
    def get_verts_segment(v, names):
        return get_smplx_verts_segment(v, segment_name=names)


lc = LossComputer(cameras=None, human_model=HM(), device="cpu", get_contact_labels_for_frame_fn=lambda i: None,
                  num_body_joints=22, logger=None)


def make_pred(v):
    return SimpleNamespace(human=SimpleNamespace(verts_seq=v), obj=SimpleNamespace(R=R0.expand(T, 3, 3), trans=t0.expand(T, 3)))


left, right = lc.terrain_foot_gaps(data, make_pred(verts), {})
lower = torch.minimum(left, right)
print(f"1) lower-foot gap: median {lower.median()*100:+.1f} cm, mean {lower.mean()*100:+.1f}, min {lower.min()*100:+.1f}, max {lower.max()*100:+.1f}")
print("   (independent measurement in lesson 05: median +5.4, mean +4.3, min -2.0, max +9.5)")
print("   frame 0 / 120 / 239 gaps L,R in cm:",
      [(round(float(left[i]) * 100, 1), round(float(right[i]) * 100, 1)) for i in (0, 120, 239)], "(05: -0.2,-0.6 / +8.5,+8.5 / +5.5,+5.1)")

v = verts.clone().requires_grad_(True)
loss = lc._terrain_support_loss(data, make_pred(v), {"penetration_scale": 1.0}, weight=1000.0)
loss.backward()
gz = v.grad[..., 2]
hover = lower > 0.02
print(f"2) loss at weight 1000: {loss.item():.3f}. gradient of loss w.r.t. vertex z: sum over frames with hover>2cm = {gz[hover].sum().item():+.3f} "
      f"(positive = loss increases when the body goes up => optimizer moves it down)")

grounded = verts.clone()
grounded[..., 2] -= lower.detach().clamp(min=0)[:, None]          # shift each frame down by its lower-foot hover
l2 = lc._terrain_support_loss(data, make_pred(grounded), {"penetration_scale": 1.0}, weight=1000.0)
g2 = torch.minimum(*lc.terrain_foot_gaps(data, make_pred(grounded), {}))
print(f"3) after shifting each frame down by its hover: lower-foot gap median {g2.median()*100:+.2f} cm, loss {l2.item():.4f} (was {loss.item():.3f})")
