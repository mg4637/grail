#!/usr/bin/env bash
# Stage 3 re-retarget of the table climb into data/motion_lib/grail_table_v3.
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
#
# NOT changed, though v2 tried it: the ik_match_table2 wrist position cost stays
# at its shipped 600. v2 lowered it to 100 to stop the IK chasing hand targets
# that lie beyond the G1's 0.292 m arm, on the theory that full extension was
# overloading a 25 N.m shoulder. data/stage3_mujoco_feasibility.py --pin_base
# refuted the premise: driving the reference through MuJoCo with SONIC's own
# gains and effort limits saturates the worst shoulder for 2.5% of steps, not
# 100%. Torque was never the constraint, and the looser wrist task measurably
# hurt - hand penetration into the tabletop went from 27 mm (v1) to 52 mm (v2).
# Hand placement is worth more than a torque margin that was never at risk.
#
# Penetration is dealt with afterwards, by data/stage3_resolve_penetration.py:
# GMR retargets against the robot MJCF alone, so nothing in Stage 3 knows the
# table exists and no IK weighting can fix a target that is inside it.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail
export DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=Yes HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

OUT=/workspace/grail/data/motion_lib/grail_table_v3
mkdir -p "${OUT}"
exec >> "${OUT}/retarget.log" 2>&1
echo "############ stage3 retarget v3 started $(date -u) ############"

python -u -m grail.retargeting.retarget \
    --file results/generation/4dhoi_recon_smplx_valid/ComAsset/table/kid_indoor2-climb-table_rand00001 \
    --robot unitree_g1 \
    --output_dir "${OUT}" \
    --no_viewer \
    --zero_out_wrist \
    --fps 24
echo "=== retarget exited rc=$? at $(date -u) ==="
