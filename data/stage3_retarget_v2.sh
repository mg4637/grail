#!/usr/bin/env bash
# ############################################################################
# SUPERSEDED by data/stage3_retarget_v3.sh. Change 4 below - dropping the wrist
# position cost from 600 to 100 - was justified by a shoulder-torque argument
# that data/stage3_mujoco_feasibility.py --pin_base then refuted (worst shoulder
# saturates for 1.4% of steps, not 100%), and it measurably worsened hand
# placement: penetration into the tabletop went from 27 mm to 52 mm. v3 keeps
# changes 1-3 and reverts 4.
# ############################################################################
# Stage 3 re-retarget of the table climb into data/motion_lib/grail_table_v2.
#
# Changes relative to the original grail_table run:
#   1. imports/GMR/assets/unitree_g1/g1_mocap_29dof.xml - 15 joint ranges restored
#      to the URDF values. The hip pitch was range="-1.57 1.57" (the URDF says
#      -2.5307 2.8798), which mink.ConfigurationLimit pinned at exactly -1.5700
#      for frames 72-95 of the old output - the whole mount.
#   2. imports/GMR/general_motion_retargeting/motion_retarget.py - solve_ik was
#      called positionally, so ik_limits bound to `safety_break` and the
#      VelocityLimit never reached the solver. Now passed as limits=.
#   3. grail/retargeting/retarget.py - configuration.update() after the per-frame
#      root write, so the first solve of each frame does not use stale FK.
#   4. data/g1_smplx/gmr_smplx_to_g1.json - ik_match_table2 wrist position cost
#      600 -> 100. The target arm (shoulder->wrist 0.340-0.375 m) is longer than
#      the real G1 arm (0.292 m), so an overweighted wrist task made the IK chase
#      unreachable hand targets by sacrificing torso posture.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail
export DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=Yes HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

OUT=/workspace/grail/data/motion_lib/grail_table_v2
mkdir -p "${OUT}"
exec >> "${OUT}/retarget.log" 2>&1
echo "############ stage3 retarget v2 started $(date -u) ############"

python -u -m grail.retargeting.retarget \
    --file results/generation/4dhoi_recon_smplx_valid/ComAsset/table/kid_indoor2-climb-table_rand00001 \
    --robot unitree_g1 \
    --output_dir "${OUT}" \
    --no_viewer \
    --zero_out_wrist \
    --fps 24
echo "=== retarget exited rc=$? at $(date -u) ==="
