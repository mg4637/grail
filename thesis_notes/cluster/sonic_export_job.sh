#!/bin/bash
#SBATCH --job-name=sonic_export
#SBATCH --account=<PROJECT-ID>
#SBATCH --partition=qgpu
#SBATCH --gpus=1
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/project/<project-id>/<cluster-user>/thesis/sonic_export_%j.log
# v2: adds the passive termination logger (term_logger.py in $WORK/extra_py) and an EXTRA variable for ONE experiment override.
# usage: CKPT=<checkpoint.pt> OUT=<folder> [ENVS=4] [STEPS=500] [EXTRA="++some.hydra.override=value"] sbatch sonic_export_job.sh

WORK=/scratch/project/<project-id>/<cluster-user>/thesis
CKPT=${CKPT:?set CKPT=/path/to/checkpoint.pt}
OUT=${OUT:?set OUT=/path/to/output/folder}
ENVS=${ENVS:-4}
STEPS=${STEPS:-500}                      # policy steps at 50 Hz: 500 = 10 s; trajectory recorded every 2nd step (25 fps)
EXTRA=${EXTRA:-}                         # experiment override(s); empty = unchanged evaluation
KITC=/root/miniconda3/envs/sonic/lib/python3.11/site-packages/isaacsim/kit
DATA=/workspace/grail/data/motion_lib/stairs_wide3_terrain
USD=$DATA/object_usd/stairs_wide3_kid_indoor1-syn-stairs_rand00001.usd
mkdir -p $WORK/fakehome $OUT
ml -f purge; ml apptainer
hostname; date; echo "CKPT=$CKPT OUT=$OUT ENVS=$ENVS STEPS=$STEPS EXTRA='$EXTRA'"
apptainer exec --nv --cleanenv --home $WORK/fakehome --pwd /workspace/grail/imports/SONIC \
  --bind $WORK/grail:/workspace/grail --bind $WORK:$WORK \
  --bind $WORK/kit_rw/data:$KITC/data --bind $WORK/kit_rw/cache:$KITC/cache \
  --bind /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json:/etc/vulkan/icd.d/nvidia_icd.json \
  --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  --env PYTHONPATH=$WORK/extra_py \
  --env ACCEPT_EULA=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PRIVACY_CONSENT=Y \
  --env HYDRA_FULL_ERROR=1 --env PYTHONUNBUFFERED=1 --env WANDB_MODE=offline --env WANDB_DIR=$WORK/wandb \
  --env XDG_CACHE_HOME=$WORK/cache --env HF_HOME=$WORK/hf_cache --env TORCH_HOME=$WORK/torch_cache \
  --env TORCH_EXTENSIONS_DIR=$WORK/torch_ext --env TORCH_CUDA_ARCH_LIST=8.0 \
  $WORK/containers/grail_sandbox \
  /root/miniconda3/envs/sonic/bin/python -u gear_sonic/eval_agent_trl.py \
    +checkpoint=$CKPT +headless=True \
    ++num_envs=$ENVS ++max_render_steps=$STEPS \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=$DATA/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=$DATA/objects \
    ++manager_env.config.terrain_motion_dir=$DATA \
    ++manager_env.config.flat_usd_path=$USD \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
    ++manager_env.recorders.trajectory._target_=term_logger.TrajectoryTermLoggerCfg \
    ++manager_env.recorders.trajectory.save_path=$OUT \
    $EXTRA
echo "finished: $(date)"; ls -la $OUT
echo "--- terminations (first 12 lines) ---"; head -12 $OUT/terminations.jsonl 2>/dev/null
