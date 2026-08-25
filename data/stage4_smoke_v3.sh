#!/usr/bin/env bash
# 5-iteration smoke test of terrain_tracking_grail + grail_table_v3f.
# Confirms the new undesired_contacts regex matches real body names, that
# is_terminated instantiates, and that the physics_material event runs, before
# stage4_train_v3.sh spends ~7 GPU-hours finding out.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail/imports/SONIC
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMNI_KIT_ACCEPT_EULA=Yes
D=/workspace/grail/data/motion_lib/grail_table_v3f
K=table_kid_indoor2-climb-table_rand00001
OUT=/workspace/grail/data/stage4_train/table_v3_smoke
W=/workspace/grail/data/stage4_train/table/last.pt
rm -rf "${OUT}"; mkdir -p "${OUT}"
exec >> "${OUT}/smoke.log" 2>&1
echo "############ smoke test started $(date -u) ############"
python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/scene/terrain_tracking_grail \
    num_envs=64 headless=True \
    ++use_wandb=False ++multi_gpu=False \
    ++resume=True ++checkpoint=${W} \
    experiment_dir=${OUT} \
    ++algo.config.num_learning_iterations=5 \
    ++callbacks.model_save.save_frequency=100000 \
    ++manager_env.config.gpu_collision_stack_size_exp=28 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${D}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${D}/objects \
    ++manager_env.config.terrain_motion_dir=${D} \
    ++manager_env.config.flat_to_terrain_ratio=0 \
    ++manager_env.config.flat_usd_path=${D}/object_usd/${K}.usd
echo "=== smoke exited rc=$? at $(date -u) ==="
