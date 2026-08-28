#!/usr/bin/env bash
# 5-iteration smoke test of chair_grail + grail_chair_split.
#
# Validates the whole chain before any real GPU spend: g1_43dof loads, both
# object USDs resolve through multi_object_per_env, the per-motion meta supplies
# table_pos (parked for __carry, in place for __basket), the finger primitive
# reads hand_action_left/right, and the BPS lookup finds the chair.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail/imports/SONIC
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMNI_KIT_ACCEPT_EULA=Yes PWD=/workspace/grail
D=/workspace/grail/data/motion_lib/grail_chair_split
OUT=/workspace/grail/data/stage4_train/chair_smoke
rm -rf "${OUT}"; mkdir -p "${OUT}"
exec >> "${OUT}/smoke.log" 2>&1
echo "############ chair smoke test $(date -u) ############"
python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/hoi/chair_grail \
    num_envs=64 headless=True \
    ++use_wandb=False ++multi_gpu=False \
    experiment_dir=${OUT} \
    ++algo.config.num_learning_iterations=5 \
    ++callbacks.model_save.save_frequency=100000 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${D}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${D}/objects \
    ++manager_env.commands.motion.motion_lib_cfg.bps_dir=${D}/bps \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
    ++manager_env.config.object_usd_path=${D}/object_usd
echo "=== smoke exited rc=$? at $(date -u) ==="
