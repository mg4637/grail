#!/usr/bin/env bash
# Stage 4 table finetune, retried at HALF the reference playback speed.
#
# Why: the 6000-iteration run in data/stage4_train/table reached progress 0.644
# and died on `anchor_pos` — the |ref pelvis z - robot pelvis z| > 0.15 m check.
# Measured on the recorded rollout (data/stage4_eval/table_ft/000000.trajectory.pkl),
# the tracking error rode at 0.10-0.147 m for the whole climb against a 0.15 m
# ceiling: mean 0.059, max 0.147. The robot never fell, never flailed - it was
# chronically ~4 frames behind a reference that climbs at up to 0.893 m/s.
#
#   0.15 m budget / 0.893 m/s = 0.168 s = 4 source frames of slack.
#
# motion_fps_scale=0.5 reinterprets the same 121 keyframes as 12 fps instead of
# 24 (motion_lib_base.py:249,1463). Duration 5.04 s -> 10.08 s, every reference
# velocity halves, peak climb 0.893 -> 0.45 m/s, slack 0.168 -> 0.336 s. The
# termination thresholds are untouched - those are terrain_release's own values
# and loosening them would move the goalposts rather than fix anything.
#
# experiment_dir differs from dirname(checkpoint), so train_agent_trl treats
# this as a WARM resume: policy weights carry over, optimizer state and step
# counter start fresh. That is what we want, because the task's timing changed.
#
# save_frequency=2000 (not 1000): checkpoints are 388 MB and the shared root
# filesystem is nearly full. 3 checkpoints + last.pt ~ 1.6 GB.
set -uo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail/imports/SONIC

export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMNI_KIT_ACCEPT_EULA=Yes

D=/workspace/grail/data/motion_lib/grail_table
K=table_kid_indoor2-climb-table_rand00001
E=/workspace/grail/data/stage4_train/table_slow

# Warm-start weights. Prefer our own 6000-iteration finetune (already knows this
# table and this climb, reached progress 0.644). That file is 388 MB, over
# GitHub's 100 MB per-file limit, so it is NOT in the repo - on a fresh machine
# it will be missing and we fall back to the released terrain policy, which
# scripts/setup/download_checkpoints.sh fetches. The fallback costs the 6 hours
# of table-specific training but still starts from a policy that can walk and
# balance, so it is a warm start either way.
W=/workspace/grail/data/stage4_train/table/last.pt
if [ ! -f "${W}" ]; then
    W=/workspace/grail/imports/SONIC/models/terrain_release/last.pt
fi

mkdir -p "${E}"
exec >> "${E}/train.log" 2>&1
echo "=== motion_fps_scale=0.5 warm-start from ${W} started $(date -u) ==="
if [ ! -f "${W}" ]; then
    echo "!! no warm-start checkpoint found - run scripts/setup/download_checkpoints.sh first"
    exit 1
fi

python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/scene/terrain_tracking \
    num_envs=2048 headless=True \
    ++resume=True \
    ++checkpoint=${W} \
    experiment_dir=${E} \
    ++use_wandb=False ++multi_gpu=False \
    ++algo.config.num_learning_iterations=6000 \
    ++callbacks.model_save.save_frequency=2000 \
    ++manager_env.config.gpu_collision_stack_size_exp=28 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_fps_scale=0.5 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${D}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${D}/objects \
    ++manager_env.config.terrain_motion_dir=${D} \
    ++manager_env.config.flat_to_terrain_ratio=0 \
    ++manager_env.config.flat_usd_path=${D}/object_usd/${K}.usd

echo "=== trainer exited rc=$? at $(date -u) ==="
