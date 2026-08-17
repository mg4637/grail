#!/usr/bin/env bash
# Stage 4 table climb, two-phase termination curriculum.
#
# WHY THIS EXISTS
# ---------------
# The 6000-iteration finetune in data/stage4_train/table stopped at progress
# 0.644 on `anchor_pos`. The important consequence is not the score, it is this:
# because every episode terminates at ~64%, the policy has NEVER EXPERIENCED THE
# LAST 36% OF THE MOTION - the part where the robot gets its feet under it and
# stands up. It cannot learn a phase it never reaches. Watch
# data/stage4_eval/table_ft/000000.mp4 and this is visible: the robot hauls its
# torso onto the table in a belly-flop, which is what maximises pelvis height
# (and so keeps the episode alive) for a policy that has only ever been scored
# on the first two thirds.
#
# So phase 1 loosens the termination thresholds far enough that the robot runs
# to the end of the clip and gets gradient signal from the stand-up. The motion
# it produces will be ugly - that is expected, phase 1 buys EXPOSURE, not
# quality. Phase 2 then restores the real thresholds and refines a motion whose
# shape the policy already knows.
#
# This is curriculum learning, not moving the goalposts: the thresholds are
# relaxed for TRAINING only. Evaluation (stage4_post_train_eval_curriculum.sh)
# always runs at the shipped 0.15 / 0.15 / 0.2, which are terrain_release's own
# values. If you relax the numbers at eval time you have not fixed anything, you
# have only removed the detector - and export_successful_rollouts feeds Stage 5,
# so a motion that passes a widened gate becomes a bad motion on real hardware.
#
# Measured margin against every termination term on the failing rollout:
#   anchor_pos    limit 0.15   worst 0.147   <- the only one at its limit (98%)
#   ee_body_pos   limit 0.15   worst <=0.128
#   foot_pos_xyz  limit 0.20   worst <=0.252 (upper bound; the env compares in
#                                             body_pos_relative_w, which removes
#                                             root drift my measurement kept)
# There is real headroom behind anchor_pos, so relaxing it does not simply move
# the failure to the next term.
#
# Runs both phases sequentially in one process, so the two never contend for the
# GPU. Launch once, detached:
#     docker exec -d grail_dev bash /workspace/grail/data/stage4_train_curriculum.sh
set -uo pipefail

# --- knobs -------------------------------------------------------------------
# Phase 1 thresholds. SONIC's own unused defaults in
# gear_sonic/config/manager_env/terminations/terms/*.yaml are 0.25 / 0.25 / 0.5;
# these go a little looser still, because the robot has to survive an entirely
# unpractised stand-up. If phase 1 still dies short of the end, raise these
# further - phase 1 has no standard to uphold, its only job is to reach the end.
P1_ANCHOR=0.35
P1_EE=0.35
P1_FOOT=0.5
P1_ITERS=3000        # exposure phase; ~3 h at 2048 envs on a 4090
P2_ITERS=6000        # refinement phase; ~6 h

NUM_ENVS=2048
SAVE_EVERY=2000      # checkpoints are 388 MB and the host disk is nearly full

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail/imports/SONIC
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMNI_KIT_ACCEPT_EULA=Yes

D=/workspace/grail/data/motion_lib/grail_table
K=table_kid_indoor2-climb-table_rand00001
P1=/workspace/grail/data/stage4_train/table_curr_p1
P2=/workspace/grail/data/stage4_train/table_curr_p2
EV1=/workspace/grail/data/stage4_eval/table_curr_p1

# Warm-start weights for phase 1. Prefer our own finetune - it already knows
# this table and this climb. That file is 388 MB, over GitHub's per-file limit,
# so it is not in the repo; on a fresh clone we fall back to the released
# terrain policy that scripts/setup/download_checkpoints.sh fetches.
W=/workspace/grail/data/stage4_train/table/last.pt
[ -f "${W}" ] || W=/workspace/grail/imports/SONIC/models/terrain_release/last.pt

mkdir -p "${P1}" "${P2}" "${EV1}"
exec >> "${P1}/curriculum.log" 2>&1

echo "############ termination curriculum started $(date -u) ############"
if [ ! -f "${W}" ]; then
    echo "!! no warm-start checkpoint - run scripts/setup/download_checkpoints.sh first"
    exit 1
fi
echo "warm start: ${W}"

COMMON=(
    +exp=manager/universal_token/scene/terrain_tracking
    num_envs=${NUM_ENVS} headless=True
    ++use_wandb=False ++multi_gpu=False
    ++callbacks.model_save.save_frequency=${SAVE_EVERY}
    ++manager_env.config.gpu_collision_stack_size_exp=28
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${D}/robot
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${D}/objects
    ++manager_env.config.terrain_motion_dir=${D}
    ++manager_env.config.flat_to_terrain_ratio=0
    ++manager_env.config.flat_usd_path=${D}/object_usd/${K}.usd
)

# ---------------------------------------------------------------- phase 1 ----
echo
echo "=== PHASE 1: loose thresholds ${P1_ANCHOR}/${P1_EE}/${P1_FOOT}, ${P1_ITERS} iters ($(date -u)) ==="
python -u gear_sonic/train_agent_trl.py "${COMMON[@]}" \
    ++resume=True \
    ++checkpoint=${W} \
    experiment_dir=${P1} \
    ++algo.config.num_learning_iterations=${P1_ITERS} \
    ++manager_env.terminations.anchor_pos.params.threshold=${P1_ANCHOR} \
    ++manager_env.terminations.ee_body_pos.params.threshold=${P1_EE} \
    ++manager_env.terminations.foot_pos_xyz.params.threshold=${P1_FOOT}
echo "=== phase 1 exited rc=$? at $(date -u) ==="

if [ ! -f "${P1}/last.pt" ]; then
    echo "!! phase 1 produced no checkpoint - stopping before phase 2"
    exit 1
fi

# --------------------------------------------- diagnostic: did it reach the end?
# Cheap (~5 min) and the whole point of phase 1. Evaluated at the LOOSE
# thresholds, because that is the task phase 1 was actually trained on. If
# progress here is not close to 1.0 the robot still never sees the stand-up, and
# phase 2 will refine a motion that is still missing its ending - raise
# P1_ANCHOR/P1_EE/P1_FOOT and rerun phase 1 rather than continuing.
echo
echo "=== PHASE 1 CHECK: eval at loose thresholds ($(date -u)) ==="
sleep 30
python -u gear_sonic/eval_agent_trl.py \
    checkpoint=${P1}/last.pt \
    ++headless=True ++use_wandb=False ++multi_gpu=False \
    ++eval_callbacks=im_eval ++run_eval_loop=False \
    ++eval_output_dir=${EV1} ++num_envs=8 \
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
    ++manager_env.terminations.anchor_pos.params.threshold=${P1_ANCHOR} \
    ++manager_env.terminations.ee_body_pos.params.threshold=${P1_EE} \
    ++manager_env.terminations.foot_pos_xyz.params.threshold=${P1_FOOT} \
    ++manager_env.config.render_results=True \
    ++manager_env.config.save_rendering_dir=${EV1} \
    ++manager_env.config.env_spacing=10.0 \
    ++manager_env.config.max_render_envs=8 \
    ++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg \
    ++manager_env.recorders.render_envs.video_save_path=${EV1} \
    ++manager_env.recorders.render_envs.video_quality=5
echo "--- phase 1 result (want progress near 1.0 AT LOOSE thresholds) ---"
grep -E "MOTION_RESULT|Progress Rate" "${P1}/curriculum.log" | tail -3

# ---------------------------------------------------------------- phase 2 ----
# Strict thresholds restored by omission - the config's own 0.15/0.15/0.2 apply.
# experiment_dir != dirname(checkpoint), so this is a warm_resume: weights carry
# over, optimizer state and step counter reset. Correct here, since the
# termination structure - and therefore the task - has changed.
echo
echo "=== PHASE 2: shipped thresholds 0.15/0.15/0.2, ${P2_ITERS} iters ($(date -u)) ==="
sleep 30
python -u gear_sonic/train_agent_trl.py "${COMMON[@]}" \
    ++resume=True \
    ++checkpoint=${P1}/last.pt \
    experiment_dir=${P2} \
    ++algo.config.num_learning_iterations=${P2_ITERS}
echo "=== phase 2 exited rc=$? at $(date -u) ==="

echo
echo "############ curriculum finished $(date -u) ############"
echo "now run: docker exec -d grail_dev bash /workspace/grail/data/stage4_post_train_eval_curriculum.sh"
