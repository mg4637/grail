#!/usr/bin/env bash
# Stage 4 chair task: carry -> place -> step up -> take basket -> step down.
#
# Two motions, one dynamic object each (data/stage3_split_chair.py):
#   __carry   f0-135   object=chair, table_usd_path parked underground
#   __basket  f128-240 object=basket, table_usd_path = bookshelf + placed chair
# MultiUsdFileCfg(random_choice=False) pairs env k with USD[k % 2] and
# use_paired_motions pairs env k with motion[k % 2], so odd/even envs each train
# on one full motion, and commands.py itself refuses to start unless the USD
# stems equal the motion keys in order (verified in the smoke test's own log:
# "[OK] Motion-USD pairing verified: 2 motions <-> 2 USDs, all stems match").
#
# What went into the reference before this run, briefly (full detail in
# IMPLEMENTATION_HISTORY.md):
#   - three Stage 3 bug fixes carried over from the table task (hip-pitch
#     range, mink.solve_ik keyword arg, stale-FK write)
#   - a geometry-derived grasp signal per object, replacing Stage 2's contact
#     annotation only where it was actually wrong (basket) - validated by a
#     physical constraint (a raised object can't have both hands open), 0
#     violations on both objects
#   - the chair's 9.8 cm float after placement, regrounded
#   - a genuine foot-planting defect - the right foot never touched the floor
#     for the ~4s carry walk, and both feet hovered 2-4cm off the floor after
#     the step-down - fixed per-leg, per-footfall, against a real reference
#     first (an ungated version of this same fix produced an anatomically
#     broken pose - knee snapped straight, hip twisted 75 deg - on one
#     candidate frame; caught before writing it, see the script's own comment)
#
# stage3_chair_feasibility.py confirms the fix, not just the geometry: severe
# infeasible frames (>100N of base wrench no push-only contact can supply)
# dropped 11->6 in __carry and 13->8 in __basket, and every frame that dropped
# out of the severe list lines up with a specific corrected footfall. A
# residual remains during the continuous single-support transitions in both -
# possibly the same quasi-static-test limitation noted for the table task
# (a real gait uses momentum this test can't credit), not confirmed as a
# genuine remaining defect.
#
# Warm start: pnp_table_warmstart, the only released checkpoint at this
# architecture (g1_43dof, dex hands, BPS observations) - matches
# chair_grail.yaml's actor/critic and observation configs since chair_grail.yaml
# is derived from pnp_table.yaml.
set -uo pipefail

NUM_ENVS=2048
ITERS=6000           # matches the table task's budget; ~5-7h on this 4090
SAVE_EVERY=2000

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail/imports/SONIC
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMNI_KIT_ACCEPT_EULA=Yes PWD=/workspace/grail

D=/workspace/grail/data/motion_lib/grail_chair_split
OUT=/workspace/grail/data/stage4_train/chair

W=/workspace/grail/imports/SONIC/models/pnp_table_warmstart/last.pt

mkdir -p "${OUT}"
exec >> "${OUT}/train.log" 2>&1
echo "############ chair training started $(date -u) ############"
[ -f "${W}" ] || { echo "!! no warm-start checkpoint at ${W}"; exit 1; }
echo "warm start: ${W}"

python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/hoi/chair_grail \
    num_envs=${NUM_ENVS} headless=True \
    ++use_wandb=False ++multi_gpu=False \
    ++resume=True ++checkpoint=${W} \
    experiment_dir=${OUT} \
    ++algo.config.num_learning_iterations=${ITERS} \
    ++callbacks.model_save.save_frequency=${SAVE_EVERY} \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${D}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${D}/objects \
    ++manager_env.commands.motion.motion_lib_cfg.bps_dir=${D}/bps \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
    ++manager_env.config.object_usd_path=${D}/object_usd
echo "=== training exited rc=$? at $(date -u) ==="
