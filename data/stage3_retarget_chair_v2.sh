#!/usr/bin/env bash
# Stage 3 re-retarget of the chair carry+climb into data/motion_lib/grail_chair_v2.
#
# Differences from the existing grail_chair / grail_chair_ha:
#
#  1. Carries the three Stage 3 bug fixes proven on the table clip:
#     - g1_mocap_29dof.xml joint ranges restored to the URDF (hip pitch was
#       clamped to +/-1.57 against a real -2.5307..2.8798, which pinned the joint
#       through the whole mount).
#     - mink.solve_ik called with limits= instead of positionally, so the
#       VelocityLimit actually reaches the solver.
#     - configuration.update() after the per-frame root write, so the first solve
#       of each frame is not scored against stale FK.
#
#  2. NO --zero_out_wrist. The table clip froze the wrists because the terrain
#     policy has no fingers and nothing was being held. This clip is a CARRY: the
#     wrist orientation is what puts the palms on the chair rail, and the robot
#     model for this task (g1_43dof) has actuated hands.
#
#  3. Produces object_usd/, which grail_chair_ha lacks - terrain_motion_dir
#     requires it paired 1:1 with robot/*.pkl.
#
# The reference stays 29 body DoF. SONIC's g1_43dof drives the 14 finger joints
# from a 2-scalar open/close PRIMITIVE (use_finger_primitive +
# use_motion_hand_actions), not from tracked finger angles, so no finger IK is
# needed. That grasp signal is generated separately by
# data/stage3_grasp_signal.py - the one shipped in grail_chair_ha is wrong
# (Stage 2's contact detection loses the left hand for the entire carry, so the
# left hand is commanded OPEN from f24-84 while it is holding the chair).
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail
export DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=Yes HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

OUT=/workspace/grail/data/motion_lib/grail_chair_v2
mkdir -p "${OUT}"
exec >> "${OUT}/retarget.log" 2>&1
echo "############ stage3 chair retarget v2 started $(date -u) ############"

python -u -m grail.retargeting.retarget \
    --file results/generation/4dhoi_recon_smplx_valid/ComAsset/chair/kid_indoor2-robot-manipulation_rand00001 \
    --robot unitree_g1 \
    --output_dir "${OUT}" \
    --no_viewer \
    --fps 24
echo "=== retarget exited rc=$? at $(date -u) ==="
