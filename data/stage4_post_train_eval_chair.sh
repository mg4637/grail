#!/usr/bin/env bash
# Scored eval of the chair checkpoint, PER MOTION, at shipped thresholds.
#
# NOTE: no +exp= here. eval_agent_trl.py auto-loads checkpoint.parent/config.yaml
# (the saved training config), so re-specifying +exp duplicates a default group
# ("manager_env/recorders appears more than once in the final defaults list") -
# caught on the first attempt, which failed before running anything.
#
# Run separately per motion (filter_motion_keys) rather than mixed, because
# training's own log showed object_pos_deviation pinned at EXACTLY 0.500 for
# all 6000 iterations while envs alternate carry/basket 50/50 by construction -
# strong evidence one whole motion fails identically every episode with zero
# improvement, hidden by the other motion's numbers when pooled. This is what
# actually answers which one, and by how much.
set -uo pipefail
E=/workspace/grail/data/stage4_train/chair
S=/workspace/grail/data/stage4_eval/chair
mkdir -p "${S}"
exec >> "${S}/post_train_eval.log" 2>&1

while pgrep -f "[t]rain_agent_trl|[e]val_agent_trl" >/dev/null; do sleep 30; done
sleep 30

if [ ! -f "${E}/last.pt" ]; then
    echo "!! no checkpoint at ${E}/last.pt"; exit 1
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail/imports/SONIC
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMNI_KIT_ACCEPT_EULA=Yes PWD=/workspace/grail

D=/workspace/grail/data/motion_lib/grail_chair_split
K=chair_kid_indoor2-robot-manipulation_rand00001

for motion in "${K}__carry" "${K}__basket"; do
    SM="${S}/${motion##*__}"
    mkdir -p "${SM}"
    echo "=== evaluating ${motion} at shipped thresholds ($(date -u)) ==="
    python -u gear_sonic/eval_agent_trl.py \
        checkpoint=${E}/last.pt \
        ++headless=True ++use_wandb=False ++multi_gpu=False \
        ++eval_callbacks=im_eval \
        ++run_eval_loop=False \
        ++eval_output_dir=${SM} \
        ++num_envs=8 \
        ++algo.config.eval.num_eval_episodes=8 \
        ++manager_env.commands.motion.motion_lib_cfg.motion_file=${D}/robot \
        ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${D}/objects \
        ++manager_env.commands.motion.motion_lib_cfg.bps_dir=${D}/bps \
        ++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[${motion}] \
        ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
        ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
        ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
        ++manager_env.config.object_usd_path=${D}/object_usd \
        ++manager_env.config.render_results=True \
        ++manager_env.config.save_rendering_dir=${SM} \
        ++manager_env.config.env_spacing=10.0 \
        ++manager_env.config.max_render_envs=8 \
        ++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg \
        ++manager_env.recorders.render_envs.video_save_path=${SM} \
        ++manager_env.recorders.render_envs.video_quality=5 \
        ++manager_env.recorders.trajectory._target_=gear_sonic.envs.manager_env.mdp.recorders.TrajectoryRecorderCfg \
        ++manager_env.recorders.trajectory.save_path=${SM}
    echo "=== ${motion} eval exited rc=$? at $(date -u) ==="
done
echo "--- results ---"
grep -E "MOTION_RESULT|Progress Rate|Success Rate" "${S}/post_train_eval.log" | tail -10
