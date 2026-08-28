#!/usr/bin/env bash
# Untruncated rollout of the 0.644 BASELINE policy, FOR VISUALISATION ONLY.
#
# This is data/stage4_train/table/last.pt - the earlier finetune, trained on the
# ORIGINAL grail_table reference (not grail_table_v3). Pairing matters: this
# checkpoint must be evaluated against the reference it trained on.
#
# NOTE its scored failure mode differs from the 0.772 run: it terminates on
# anchor_pos (root height diverging), not ee_body_pos (foot placement). Expect a
# different-looking failure, not the same foot-placement miss.
#
# The scored eval (stage4_post_train_eval_v3.sh) terminates at progress 0.604,
# so its video stops mid-mount and you cannot see what the policy would have
# done next. This runs the same checkpoint with the termination thresholds set
# absurdly high, so only motion_time_out can end the episode and the rollout
# covers the whole clip.
#
# THE NUMBERS THIS PRINTS ARE MEANINGLESS. With no active termination term,
# progress is 1.0 by construction - that is exactly the artifact that made the
# previous "it works now" run look successful. Read the video, never the score.
# The real score is progress 0.644 / mpjpe 114.4 (reason=anchor_pos), from the scored eval.
set -uo pipefail
E=/workspace/grail/data/stage4_train/table
S=/workspace/grail/data/stage4_eval/table_ft_notrunc
mkdir -p "${S}"
exec >> "${S}/notrunc.log" 2>&1

while pgrep -f "[t]rain_agent_trl|[e]val_agent_trl" >/dev/null; do sleep 30; done
sleep 20

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail/imports/SONIC
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMNI_KIT_ACCEPT_EULA=Yes

D=/workspace/grail/data/motion_lib/grail_table
K=table_kid_indoor2-climb-table_rand00001

echo "=== untruncated rollout (VISUALISATION ONLY) $(date -u) ==="
python -u gear_sonic/eval_agent_trl.py \
    checkpoint=${E}/last.pt \
    ++headless=True ++use_wandb=False ++multi_gpu=False \
    ++eval_callbacks=im_eval ++run_eval_loop=False \
    ++eval_output_dir=${S} ++num_envs=8 \
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
    ++manager_env.terminations.anchor_pos.params.threshold=100.0 \
    ++manager_env.terminations.anchor_ori_full.params.threshold=100.0 \
    ++manager_env.terminations.ee_body_pos.params.threshold=100.0 \
    ++manager_env.terminations.foot_pos_xyz.params.threshold=100.0 \
    ++manager_env.config.render_results=True \
    ++manager_env.config.save_rendering_dir=${S} \
    ++manager_env.config.env_spacing=10.0 \
    ++manager_env.config.max_render_envs=8 \
    ++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg \
    ++manager_env.recorders.render_envs.video_save_path=${S} \
    ++manager_env.recorders.render_envs.video_quality=5 \
    ++manager_env.recorders.trajectory._target_=gear_sonic.envs.manager_env.mdp.recorders.TrajectoryRecorderCfg \
    ++manager_env.recorders.trajectory.save_path=${S}
echo "=== notrunc eval exited rc=$? at $(date -u) ==="
