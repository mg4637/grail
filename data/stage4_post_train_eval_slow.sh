#!/usr/bin/env bash
# Evaluates the motion_fps_scale=0.5 finetune once its trainer exits.
#
# CRITICAL: this eval passes the SAME motion_fps_scale=0.5. A policy trained on
# the half-speed reference must be scored against the half-speed reference;
# evaluating it at 1.0 would measure it on a task it was never trained for.
#
# Numbers to beat, same motion at full speed:
#   zero-shot   progress 0.308 | mpjpe 388.2 | reason ee_body_pos
#   finetuned   progress 0.644 | mpjpe 114.4 | reason anchor_pos
# Export requires progress 1.0 (export_successful_rollouts --min_progress 1.0).
#
# Note when reading the result: progress 1.0 here means the robot completed the
# climb performed over 10.08 s rather than 5.04 s. That is a real, usable
# result - a slower climb - not the original clip's timing.
set -uo pipefail

E=/workspace/grail/data/stage4_train/table_slow
S=/workspace/grail/data/stage4_eval/table_slow
mkdir -p "${S}"
exec >> "${S}/post_train_eval.log" 2>&1

echo "=== waiting for trainer to exit ($(date -u)) ==="
while pgrep -f "[t]rain_agent_trl" >/dev/null; do sleep 60; done
echo "=== trainer gone, settling 60s before eval ($(date -u)) ==="
sleep 60

if [ ! -f "${E}/last.pt" ]; then
    echo "!! no checkpoint at ${E}/last.pt — nothing to evaluate"; exit 1
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail/imports/SONIC
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMNI_KIT_ACCEPT_EULA=Yes

D=/workspace/grail/data/motion_lib/grail_table
K=table_kid_indoor2-climb-table_rand00001

echo "=== evaluating ${E}/last.pt at motion_fps_scale=0.5 ($(date -u)) ==="
python -u gear_sonic/eval_agent_trl.py \
    checkpoint=${E}/last.pt \
    ++headless=True ++use_wandb=False ++multi_gpu=False \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++eval_output_dir=${S} \
    ++num_envs=8 \
    ++algo.config.eval.num_eval_episodes=8 \
    ++manager_env.config.gpu_collision_stack_size_exp=28 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_fps_scale=0.5 \
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
