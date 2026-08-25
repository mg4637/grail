#!/usr/bin/env bash
# ############################################################################
# DO NOT RUN. Superseded by data/stage4_post_train_eval_v3.sh.
#
# This script rests on a diagnosis that is false: "the policy never experiences
# the last 36% of the motion, because every episode terminates at ~64%".
# terrain_tracking has reference-state initialization on by default -
# use_paired_motions: true selects the branch at commands.py:3204-3210, which
# draws a UNIFORM RANDOM START FRAME over the whole clip (motion_lib_base.py:
# 2353-2367). The policy has trained on the stand-up from the first iteration.
# That is why loosening the thresholds, and then disabling them, changed
# nothing. Relaxing them further only removes the detector.
#
# The real defect was upstream, in the retargeted reference; see the header of
# data/stage4_train_v3.sh. Kept for the record only.
# ############################################################################
# Evaluates the phase-2 policy from stage4_train_curriculum.sh.
#
# THE THRESHOLDS ARE NOT OVERRIDDEN HERE, DELIBERATELY. The config's shipped
# 0.15 / 0.15 / 0.2 apply - terrain_release's own values, the same gate the
# 0.644 baseline was measured against. The curriculum relaxed them for training
# only; if they were relaxed here too, a better number would mean nothing.
#
# Numbers to beat, same motion, same gate:
#   zero-shot   progress 0.308 | mpjpe 388.2 | reason ee_body_pos
#   finetuned   progress 0.644 | mpjpe 114.4 | reason anchor_pos
# Export requires progress 1.0 (export_successful_rollouts --min_progress 1.0).
#
# Read mpjpe as carefully as progress. The 0.644 run reached its height target
# while belly-flopping onto the table, at 114 mm mean joint error - a pelvis
# that is high enough is not the same as a climb. If progress rises but mpjpe
# stays near 100 mm, the curriculum bought reach rather than quality, and the
# next lever is upstream: split the clip so the climb trains as its own short
# task instead of as the tail of a walk-up.
set -uo pipefail

E=/workspace/grail/data/stage4_train/table_curr_p2
S=/workspace/grail/data/stage4_eval/table_curr_p2
mkdir -p "${S}"
exec >> "${S}/post_train_eval.log" 2>&1

echo "=== waiting for any trainer to exit ($(date -u)) ==="
while pgrep -f "[t]rain_agent_trl" >/dev/null; do sleep 60; done
echo "=== trainer gone, settling 60s before eval ($(date -u)) ==="
sleep 60

if [ ! -f "${E}/last.pt" ]; then
    echo "!! no checkpoint at ${E}/last.pt - did phase 2 finish?"; exit 1
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail/imports/SONIC
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMNI_KIT_ACCEPT_EULA=Yes

D=/workspace/grail/data/motion_lib/grail_table
K=table_kid_indoor2-climb-table_rand00001

echo "=== evaluating ${E}/last.pt at SHIPPED thresholds ($(date -u)) ==="
python -u gear_sonic/eval_agent_trl.py \
    checkpoint=${E}/last.pt \
    ++headless=True ++use_wandb=False ++multi_gpu=False \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++eval_output_dir=${S} \
    ++num_envs=8 \
    ++algo.config.eval.num_eval_episodes=8 \
    ++manager_env.config.gpu_collision_stack_size_exp=28 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${D}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${D}/objects \
    ++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[${K}] \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
    ++manager_env.config.terrain_motion_dir=${D} \
    ++manager_env.config.flat_to_terrain_ratio=0 \
    ++manager_env.config.flat_usd_path=${D}/object_usd/${K}.usd \
    ++manager_env.config.render_results=True \
    ++manager_env.config.save_rendering_dir=${S} \
    ++manager_env.config.env_spacing=10.0 \
    ++manager_env.config.max_render_envs=8 \
    ++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg \
    ++manager_env.recorders.render_envs.video_save_path=${S} \
    ++manager_env.recorders.render_envs.video_quality=5 \
    ++manager_env.recorders.trajectory._target_=gear_sonic.envs.manager_env.mdp.recorders.TrajectoryRecorderCfg \
    ++manager_env.recorders.trajectory.save_path=${S}

echo "=== eval finished rc=$? at $(date -u) ==="
grep -E "MOTION_RESULT|Success Rate|Progress Rate" "${S}/post_train_eval.log" | tail -5
