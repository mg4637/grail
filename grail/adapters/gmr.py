"""Adapter for the public ``general_motion_retargeting`` (GMR) package.

GRAIL's retargeting pipeline diverges from public GMR in two places. Rather
than editing files inside the submodule (which forces ``imports/GMR`` to a
"dirty" working tree and complicates submodule bumps), we monkey-patch the
public package at import time. Importing this module is enough to activate
the patches; downstream code does:

    from grail.adapters.gmr import GMR, ROBOT_XML_DICT, ...

and gets the patched GMR transparently.

Patches applied:

1. ``GeneralMotionRetargeting.__init__`` — public GMR multiplies
   ``ik_config["human_scale_table"][k]`` by a per-instance ``ratio``. GRAIL's
   SMPL-X inputs already provide the correct ratio embedded in the SMPL-X
   betas, so we want identity (1.0). After the public init runs, every value
   in ``self.human_scale_table`` is reset to 1.0.

2. ``general_motion_retargeting.utils.smpl.load_smplx_file`` — public GMR
   only knows how to load ``.npz`` SMPL-X dumps. GRAIL's
   ``grail.pipelines.recon_4dhoi`` writes ``.pkl`` files with SMPL-X data wrapped
   under a ``human_data`` key. The replacement function transparently
   handles both ``.pkl`` (GRAIL) and ``.npz`` (public) formats; it also
   (a) truncates ``betas`` to the first 10 dims (public GMR keeps all 16),
   (b) zeroes out root translation before body-model evaluation. Public
   ``.npz`` callers see no change.
"""

from __future__ import annotations

import logging
import pickle

import general_motion_retargeting as _gmr
import general_motion_retargeting.utils.smpl as _gmr_smpl
import numpy as np
import torch

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch 1: human_scale_table -> identity (override public per-joint ratio).
# ---------------------------------------------------------------------------
_orig_gmr_init = _gmr.GeneralMotionRetargeting.__init__


def _patched_gmr_init(self, *args, **kwargs):
    _orig_gmr_init(self, *args, **kwargs)
    if hasattr(self, "human_scale_table"):
        for key in list(self.human_scale_table.keys()):
            self.human_scale_table[key] = 1.0


_gmr.GeneralMotionRetargeting.__init__ = _patched_gmr_init


# ---------------------------------------------------------------------------
# Patch 2: smpl.load_smplx_file — accept GRAIL .pkl + truncated betas.
# ---------------------------------------------------------------------------
import smplx as _smplx  # noqa: E402  (imported here so the patch is self-contained)


def _grail_load_smplx_file(smplx_file, smplx_body_model_path):
    """Drop-in replacement that supports GRAIL ``.pkl`` SMPL-X dumps.

    For ``.npz`` inputs the behavior matches public GMR except that we use
    the first 10 betas (public uses all 16) and zero out root translation
    before body-model forward — both of which GRAIL relies on.
    """
    if smplx_file.endswith(".pkl"):
        with open(smplx_file, "rb") as f:
            pkl_data = pickle.load(f)["human_data"]
        gender = "neutral"
        smplx_data = {
            "pose_body": pkl_data["poses"][..., 3:66],
            "root_orient": pkl_data["poses"][..., :3],
            "betas": pkl_data["betas"],
            "trans": pkl_data["trans"],
            "mocap_frame_rate": torch.tensor(30),
        }
        scale = torch.tensor(pkl_data.get("scale", 1.0))
    else:
        smplx_data = np.load(smplx_file, allow_pickle=True)
        gender = str(smplx_data["gender"])
        scale = torch.tensor(1.0)

    body_model = _smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=gender,
        use_pca=False,
    )
    num_frames = smplx_data["pose_body"].shape[0]
    transl = (
        smplx_data["trans"].copy()
        if hasattr(smplx_data["trans"], "copy")
        else np.array(smplx_data["trans"]).copy()
    )
    transl[..., :3] = 0.0

    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"][..., :10]).float().view(1, -1),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(transl).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float() * 10,
        right_hand_pose=torch.zeros(num_frames, 45).float() * 10,
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )
    smplx_output.vertices *= scale
    smplx_output.joints *= scale

    if len(smplx_data["betas"].shape) == 1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]

    return smplx_data, body_model, smplx_output, human_height


_gmr_smpl.load_smplx_file = _grail_load_smplx_file

_logger.debug("Applied GRAIL GMR runtime patches (scale=1.0 + .pkl SMPL-X loader).")


# ---------------------------------------------------------------------------
# Public re-exports — callers do `from grail.adapters.gmr import GMR, ...`.
# ---------------------------------------------------------------------------
from general_motion_retargeting import (  # noqa: E402
    IK_CONFIG_DICT,
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
    GeneralMotionRetargeting as GMR,
    RobotMotionViewer,
)
from general_motion_retargeting.robot_motion_viewer import draw_frame  # noqa: E402

__all__ = [
    "GMR",
    "IK_CONFIG_DICT",
    "ROBOT_BASE_DICT",
    "ROBOT_XML_DICT",
    "VIEWER_CAM_DISTANCE_DICT",
    "RobotMotionViewer",
    "draw_frame",
]
