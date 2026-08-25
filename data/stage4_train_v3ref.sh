#!/usr/bin/env bash
# Stage 4 table climb, retrained on the corrected reference.
#
# WHAT CHANGED SINCE data/stage4_train/table (progress 0.644, mpjpe 114.4)
# -----------------------------------------------------------------------
# Both the reference and the reward config. The reference is the bigger change,
# and the previous session's diagnosis was wrong about where the problem was:
#
#   * "the policy never experiences the last 36% of the motion" is false.
#     terrain_tracking has reference-state initialization on by default
#     (use_paired_motions: true -> commands.py:3204-3210 -> motion_lib_base.py:
#     2353-2367 draws a uniform random start frame over the whole clip). The
#     policy has trained on the stand-up all along, which is why loosening and
#     then disabling the 0.15 threshold changed nothing. data/stage4_train_slow.sh
#     and data/stage4_train_curriculum.sh both rest on that refuted premise.
#
#   * "the arms cannot hold the robot up" is also false. Driving the reference
#     through MuJoCo with SONIC's own gains and effort limits, root pinned to the
#     reference path (data/stage3_mujoco_feasibility.py --pin_base), saturates
#     the worst shoulder for 1.4% of steps and lands both feet on the tabletop.
#     Torque was never the constraint.
#
# What WAS wrong with the reference, all of it upstream in Stage 3:
#
#   a. GMR's g1_mocap_29dof.xml declared hip pitch as range="-1.57 1.57" against
#      a URDF limit of -2.5307 2.8798, and mink.ConfigurationLimit pinned it at
#      exactly -1.5700 for frames 72-95 - the entire mount. 15 joint ranges were
#      truncated; all restored.
#   b. GMR called mink.solve_ik positionally, so ik_limits bound to
#      `safety_break` and the VelocityLimit never reached the solver.
#   c. retarget.py wrote the root into qpos without configuration.update(), so
#      the first solve of every frame scored its tasks against stale FK.
#   d. Nothing in Stage 3 knows the scene exists - GMR retargets against the
#      robot MJCF alone. The reference walked 53 mm THROUGH the floor and put
#      the hands 27 mm inside the tabletop. SONIC's terminations measure exactly
#      those bodies at 0.15 m, against targets inside a rigid body, and on the
#      tabletop it asks the foot to stand below the surface - which the contact
#      solver answers by ejecting the foot. That is the slip in the eval video.
#      I "fixed" this with data/stage3_resolve_penetration.py and it was a
#      MISTAKE - see below.
#
# WHY THIS TRAINS ON grail_table_v3, NOT grail_table_v3f
# ------------------------------------------------------
# stage3_resolve_penetration.py enforces penetration <= 2 mm, which is an
# INEQUALITY - and a robot floating in mid-air satisfies it perfectly. Measured
# over the mount (reference frames 50-99), it converted a motion with 47/50
# frames on >=2 contacts into one with 21/50 frames touching NOTHING AT ALL and
# only 3 frames on >=2 contacts. Pushing out far enough to clear the deepest
# body lifts the shallower ones off the surface entirely.
#
# By contact-force feasibility (data/stage3_contact_feasibility.py), counting
# frames whose required base wrench no push-only friction-bounded contact force
# can supply by more than 100 N:
#
#     grail_table      (v1, shipped)              45/70 infeasible
#     grail_table_v3   (Stage 3 bugs fixed)       33/70,  8 severe
#     grail_table_v3f  (+ penetration "fix")      70/70, 52 severe   <- trained on this
#
# So the 6000-iteration run that scored progress 0.604 was trained on by far the
# worst reference of the three. v3 keeps every genuine Stage 3 fix and drops the
# damaging one. Read the residuals as comparative, not absolute: a kinematic
# retarget is never exactly dynamically consistent, and even plain walking in
# this clip shows 50-100 N of residual.
#
# So this trains on data/motion_lib/grail_table_v3, and on the reward changes
# documented in exp/manager/universal_token/scene/terrain_tracking_grail.yaml
# (vr5point std 0.1 -> 0.25, is_terminated -10, undesired_contacts exempting the
# bodies the climb actually needs, narrowed friction randomisation, small reset
# noise). Termination thresholds stay at the shipped 0.15/0.15/0.20.
#
# Launch once, detached:
#     docker exec -d grail_dev bash /workspace/grail/data/stage4_train_v3.sh
set -uo pipefail

NUM_ENVS=2048
ITERS=6000           # ~4.3 s/iter at 2048 envs on the 4090 -> ~7 h
SAVE_EVERY=2000      # checkpoints are 388 MB

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail/imports/SONIC
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMNI_KIT_ACCEPT_EULA=Yes

D=/workspace/grail/data/motion_lib/grail_table_v3
K=table_kid_indoor2-climb-table_rand00001
OUT=/workspace/grail/data/stage4_train/table_v3ref

# Warm start. Our own finetune already knows this table and this climb; it is
# 388 MB so it is not in the repo, and a fresh clone falls back to the released
# terrain policy from scripts/setup/download_checkpoints.sh. experiment_dir
# differs from dirname(checkpoint), so this is a warm_resume: weights carry over
# while the optimizer state and step counter reset. That is what we want - the
# reference and the reward function have both changed.
W=/workspace/grail/data/stage4_train/table/last.pt
[ -f "${W}" ] || W=/workspace/grail/imports/SONIC/models/terrain_release/last.pt

mkdir -p "${OUT}"
exec >> "${OUT}/train.log" 2>&1
echo "############ stage4 v3 training started $(date -u) ############"
[ -f "${W}" ] || { echo "!! no warm-start checkpoint"; exit 1; }
echo "warm start: ${W}"
echo "reference : ${D}"

python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/scene/terrain_tracking_grail \
    num_envs=${NUM_ENVS} headless=True \
    ++use_wandb=False ++multi_gpu=False \
    ++resume=True ++checkpoint=${W} \
    experiment_dir=${OUT} \
    ++algo.config.num_learning_iterations=${ITERS} \
    ++callbacks.model_save.save_frequency=${SAVE_EVERY} \
    ++manager_env.config.gpu_collision_stack_size_exp=28 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${D}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${D}/objects \
    ++manager_env.config.terrain_motion_dir=${D} \
    ++manager_env.config.flat_to_terrain_ratio=0 \
    ++manager_env.config.flat_usd_path=${D}/object_usd/${K}.usd
echo "=== training exited rc=$? at $(date -u) ==="
echo "now run: docker exec -d grail_dev bash /workspace/grail/data/stage4_post_train_eval_v3.sh"
