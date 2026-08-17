#!/usr/bin/env bash
# Detached resume of the Stage 4 table finetune. Runs INSIDE the grail_dev
# container via `docker exec -d`, so there is no host-side client process:
# nothing breaks when the user logs off.
#
# Why this exists: the first launch used `nohup docker exec ... &` from the
# host. That leaves the trainer's stdout/stderr as pipes to the host-side
# docker exec client. When that client dies at logoff, the read end closes and
# the next per-iteration print raises BrokenPipeError, which would kill a run
# that logs every iteration. Here stdout is redirected to a file on the mounted
# volume from inside the container instead.
#
# resume=True with checkpoint == <experiment_dir>/last.pt is a TRUE resume
# (step counter + optimizer state restored), not a warm-resume: train_agent_trl
# only sets warm_resume when dirname(checkpoint) != experiment_dir.
set -uo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail/imports/SONIC

export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMNI_KIT_ACCEPT_EULA=Yes

D=/workspace/grail/data/motion_lib/grail_table
K=table_kid_indoor2-climb-table_rand00001
E=/workspace/grail/data/stage4_train/table

exec >> "${E}/train.log" 2>&1
echo "=== detached resume started $(date -u) from ${E}/last.pt ==="

python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/scene/terrain_tracking \
    num_envs=2048 headless=True \
    ++resume=True \
    ++checkpoint=${E}/last.pt \
    experiment_dir=${E} \
    ++use_wandb=False ++multi_gpu=False \
    ++algo.config.num_learning_iterations=6000 \
    ++callbacks.model_save.save_frequency=1000 \
    ++manager_env.config.gpu_collision_stack_size_exp=28 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${D}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${D}/objects \
    ++manager_env.config.terrain_motion_dir=${D} \
    ++manager_env.config.flat_to_terrain_ratio=0 \
    ++manager_env.config.flat_usd_path=${D}/object_usd/${K}.usd

echo "=== trainer exited rc=$? at $(date -u) ==="
