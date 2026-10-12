#!/bin/bash
#SBATCH --job-name=sonic_train
#SBATCH --account=<PROJECT-ID>
#SBATCH --partition=qgpu
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/project/<project-id>/<cluster-user>/thesis/sonic_train_%j.log
# SONIC terrain tracking finetuned from the pretrained stairs policy.
# usage: RUN=run2 ENVS=512 ITERS=1000 [SAVE=100] [CKPT=...] sbatch --time=02:30:00 sonic_train_job.sh
WORK=/scratch/project/<project-id>/<cluster-user>/thesis
RUN=${RUN:-run1}                                   # -> training/stairs_$RUN
ITERS=${ITERS:-6000}
ENVS=${ENVS:-1024}
CKPT=${CKPT:-models/terrain_stairs_warmstart/last.pt}   # relative to imports/SONIC, or an absolute path
KITC=/root/miniconda3/envs/sonic/lib/python3.11/site-packages/isaacsim/kit
DATA=/workspace/grail/data/motion_lib/stairs_wide3_terrain
USD=$DATA/object_usd/stairs_wide3_kid_indoor1-syn-stairs_rand00001.usd
mkdir -p $WORK/fakehome $WORK/training $WORK/wandb
ml -f purge; ml apptainer
hostname; date; echo "RUN=$RUN ITERS=$ITERS ENVS=$ENVS CKPT=$CKPT"
apptainer exec --nv --cleanenv --home $WORK/fakehome --pwd /workspace/grail/imports/SONIC \
  --bind $WORK/grail:/workspace/grail --bind $WORK:$WORK \
  --bind $WORK/kit_rw/data:$KITC/data --bind $WORK/kit_rw/cache:$KITC/cache \
  --bind /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json:/etc/vulkan/icd.d/nvidia_icd.json \
  --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  --env ACCEPT_EULA=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PRIVACY_CONSENT=Y \
  --env HYDRA_FULL_ERROR=1 --env PYTHONUNBUFFERED=1 --env WANDB_MODE=offline --env WANDB_DIR=$WORK/wandb \
  --env XDG_CACHE_HOME=$WORK/cache --env HF_HOME=$WORK/hf_cache --env TORCH_HOME=$WORK/torch_cache \
  --env TORCH_EXTENSIONS_DIR=$WORK/torch_ext --env TORCH_CUDA_ARCH_LIST=8.0 \
  $WORK/containers/grail_sandbox \
  /root/miniconda3/envs/sonic/bin/python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/scene/terrain_tracking \
    num_envs=$ENVS headless=True \
    ++resume=True ++checkpoint=$CKPT \
    experiment_dir=$WORK/training/stairs_$RUN \
    ++algo.config.num_learning_iterations=$ITERS \
    ++callbacks.model_save.save_frequency=${SAVE:-100} \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=$DATA/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=$DATA/objects \
    ++manager_env.config.terrain_motion_dir=$DATA \
    ++manager_env.config.flat_usd_path=$USD
echo "finished: $(date)"
