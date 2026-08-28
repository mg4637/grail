#!/usr/bin/env bash
# Untruncated rollout of the chair checkpoint, FOR VISUALISATION ONLY, per motion.
#
# THE NUMBERS THIS PRINTS ARE MEANINGLESS. With every position/orientation/
# contact termination raised to 100 (only motion_time_out can still end an
# episode), progress is 1.0 by construction - the same artifact flagged
# repeatedly on the table task. Read the video, never the score.
# The real scores are the scored eval's: carry 0.014/11.0 (object_pos_deviation),
# basket 0.147/120.2 (ee_body_pos).
set -uo pipefail
E=/workspace/grail/data/stage4_train/chair
S=/workspace/grail/data/stage4_eval/chair_notrunc
mkdir -p "${S}"
exec >> "${S}/notrunc.log" 2>&1

while pgrep -f "[t]rain_agent_trl|[e]val_agent_trl" >/dev/null; do sleep 30; done
sleep 20

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
    echo "=== untruncated rollout (VISUALISATION ONLY) ${motion} $(date -u) ==="
    python -u gear_sonic/eval_agent_trl.py \
        checkpoint=${E}/last.pt \
        ++headless=True ++use_wandb=False ++multi_gpu=False \
        ++eval_callbacks=im_eval ++run_eval_loop=False \
        ++eval_output_dir=${SM} ++num_envs=8 \
        ++algo.config.eval.num_eval_episodes=8 \
        ++manager_env.commands.motion.motion_lib_cfg.motion_file=${D}/robot \
        ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${D}/objects \
        ++manager_env.commands.motion.motion_lib_cfg.bps_dir=${D}/bps \
        ++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[${motion}] \
        ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
        ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
        ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
        ++manager_env.config.object_usd_path=${D}/object_usd \
        ++manager_env.terminations.anchor_pos.params.threshold=100.0 \
        ++manager_env.terminations.anchor_ori_full.params.threshold=100.0 \
        ++manager_env.terminations.ee_body_pos.params.threshold=100.0 \
        ++manager_env.terminations.ee_body_pos.params.down_threshold=100.0 \
        ++manager_env.terminations.object_pos_deviation.params.threshold=100.0 \
        ++manager_env.terminations.hand_table_contact_termination.params.force_threshold=1000000.0 \
        ++manager_env.config.render_results=True \
        ++manager_env.config.save_rendering_dir=${SM} \
        ++manager_env.config.env_spacing=10.0 \
        ++manager_env.config.max_render_envs=8 \
        ++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg \
        ++manager_env.recorders.render_envs.video_save_path=${SM} \
        ++manager_env.recorders.render_envs.video_quality=5 \
        ++manager_env.recorders.trajectory._target_=gear_sonic.envs.manager_env.mdp.recorders.TrajectoryRecorderCfg \
        ++manager_env.recorders.trajectory.save_path=${SM}
    echo "=== ${motion} notrunc exited rc=$? at $(date -u) ==="
done
