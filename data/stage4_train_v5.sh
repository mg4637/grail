#!/usr/bin/env bash
# Stage 4 table climb on the v5 reference, with a feet-only tracking reward.
#
# WHAT CHANGED SINCE data/stage4_train/table_v3 (reference grail_table_v3f)
# ------------------------------------------------------------------------
# Two changes, one to the reference and one to the reward. Both target the
# failure the user identified by watching the rollout: THE ROBOT CANNOT PLACE
# ITS FOOT ON TOP OF THE TABLE. Measured on that rollout, the left foot was
# outside the tabletop footprint on 24 of 42 mount frames, worst -0.41 m, with
# both feet 12-19 cm below the surface.
#
# B. Reference: grail_table_v3f -> grail_table_v5 (data/stage3_build_v5.sh)
#
#    v5 = re-retarget with the wrists FREE (v3's --zero_out_wrist froze wrist
#    dofs 19-21/26-28 at zero, so no IK could orient the palms), then
#    penetration-resolve, foot-planting, and the new
#    data/stage3_fix_hand_contact.py.
#
#    The hand fix addresses the second thing the user saw - the robot planting
#    its hands "upside down" where the child's are flat. Palm flatness
#    (|palm_normal . world_up|, measured over each hand's own contact window):
#
#                        left            right
#      grail_table_v3f   0.18 (w 0.01)   0.17 (w 0.01)
#      grail_table_v5    0.94 (w 0.91)   0.82 (w 0.72)
#
#    A measured limit worth stating plainly: from f70 the robot's shoulder sits
#    0.43-0.59 m above the tabletop while its arm is 0.284 m, so its hands
#    CANNOT reach the table there, however the child's could. That is a body-
#    proportion difference, not a retargeting defect, so past the reach envelope
#    the fix only rotates the palm flat and holds the hand on its existing
#    trajectory. Hand-to-table distance is therefore ~unchanged (0.081 -> 0.083 m
#    left, 0.126 -> 0.124 m right); the palms are what got fixed.
#
# D. Reward: added tracking_feet_pos (weight 2.0, std 0.20, ankles only).
#
#    tracking_vr_5point_local averages all five points' squared error before the
#    exponential, so a foot 0.41 m off scored as healthy alongside four tracked
#    points. The new term scores the two ankles alone. See the comment block in
#    exp/manager/universal_token/scene/terrain_tracking_grail.yaml.
#
# Iteration count matches table_v3 exactly so the comparison is like-for-like.
# Baselines to beat: progress 0.644 / mpjpe 114.4 (table), and table_v3.
#
# Launch once, detached:
#     docker exec -d grail_dev bash /workspace/grail/data/stage4_train_v5.sh
set -uo pipefail

NUM_ENVS=2048
ITERS=6000           # ~4.3 s/iter at 2048 envs on the 4090 -> ~7 h
SAVE_EVERY=2000

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail/imports/SONIC
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMNI_KIT_ACCEPT_EULA=Yes

D=/workspace/grail/data/motion_lib/grail_table_v5
K=table_kid_indoor2-climb-table_rand00001
OUT=/workspace/grail/data/stage4_train/table_v5

# Warm start from the v3 run: same task, same observation space, and the reward
# set only gained a term, so the weights transfer. experiment_dir differs from
# dirname(checkpoint), making this a warm_resume - weights carry, optimizer
# state and step counter reset, which is what we want since both the reference
# and the reward changed.
W=/workspace/grail/data/stage4_train/table_v3/last.pt
[ -f "${W}" ] || W=/workspace/grail/data/stage4_train/table/last.pt
[ -f "${W}" ] || W=/workspace/grail/imports/SONIC/models/terrain_release/last.pt

mkdir -p "${OUT}"
exec >> "${OUT}/train.log" 2>&1
echo "############ stage4 v5 training started $(date -u) ############"
[ -f "${W}" ] || { echo "!! no warm-start checkpoint"; exit 1; }
echo "warm start: ${W}"
echo "reference : ${D}"

python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/scene/terrain_tracking_grail \
    num_envs=${NUM_ENVS} headless=True \
    ++use_wandb=False ++multi_gpu=False \
    ++resume=True ++checkpoint=${W} \
    experiment_dir=${OUT} \
    ++algo.config.num_learning_iterations=${ITERS} \
    ++callbacks.model_save.save_frequency=${SAVE_EVERY} \
    ++manager_env.config.gpu_collision_stack_size_exp=28 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${D}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${D}/objects \
    ++manager_env.config.terrain_motion_dir=${D} \
    ++manager_env.config.flat_to_terrain_ratio=0 \
    ++manager_env.config.flat_usd_path=${D}/object_usd/${K}.usd
echo "=== training exited rc=$? at $(date -u) ==="
echo "now run: docker exec -d grail_dev bash /workspace/grail/data/stage4_post_train_eval_v5.sh"
