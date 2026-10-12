#!/bin/bash
#SBATCH --job-name=recon_terrain
#SBATCH --account=<PROJECT-ID>
#SBATCH --partition=qgpu
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/project/<project-id>/<cluster-user>/thesis/recon_terrain_%j.log
# 4D HOI reconstruction (trial 2: + terrain_support loss; reuses cached steps 1-3). Manual-LLM wrapper answers the interaction check.
WORK=/scratch/project/<project-id>/<cluster-user>/thesis
ml -f purge; ml apptainer
hostname; date
apptainer exec --nv --cleanenv --pwd /workspace/grail \
  --bind $WORK/grail:/workspace/grail --bind $WORK:$WORK \
  --env PYTHONPATH=$WORK/pylibs:/workspace/grail \
  --env HF_HOME=$WORK/hf_cache --env TORCH_HOME=$WORK/torch_cache \
  --env XDG_CACHE_HOME=$WORK/cache --env TORCH_EXTENSIONS_DIR=$WORK/torch_ext \
  --env TORCH_CUDA_ARCH_LIST=8.0 \
  --env MANUAL_LLM_ANSWERS=$WORK/manual_llm/interaction_answers.json \
  --env MANUAL_LLM_LOG=$WORK/manual_llm/calls.log \
  $WORK/containers/grail_sandbox \
  /root/miniconda3/envs/grail/bin/python $WORK/manual_llm/run_recon_manual_llm.py \
    --dataset SynStairsV1 --category stairs_wide3 --results_dir results \
    --config configs/recon_4dhoi/loco_smplx_terrain.yaml \
    --skip_step1 --skip_step2 --skip_step3 --verbose
echo "exit code: $?"; date
