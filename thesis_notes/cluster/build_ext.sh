#!/bin/bash
# Rebuild nvdiffrast for the A100 (sm_80) into $WORK/pylibs and build FoundationPose mycpp. CPU only (login node).
# usage: WORK=<scratch folder> nohup ./build_ext.sh > build_ext.log 2>&1 &
set -euo pipefail
ml -f purge; ml apptainer
SB=$WORK/containers/grail_sandbox
[ -d $WORK/nvdiffrast_src ] || git clone --depth 1 https://github.com/NVlabs/nvdiffrast $WORK/nvdiffrast_src
mkdir -p $WORK/pylibs
echo "=== nvdiffrast (sm_80) $(date)"
apptainer exec --cleanenv --bind $WORK:$WORK --env WORK=$WORK --env TORCH_CUDA_ARCH_LIST=8.0 --env CUDA_HOME=/usr/local/cuda $SB bash -c '
  /root/miniconda3/envs/grail/bin/pip install --isolated --no-build-isolation --no-deps --no-cache-dir --upgrade --target $WORK/pylibs $WORK/nvdiffrast_src' 2>&1 | tail -15
echo "=== mycpp $(date)"
apptainer exec --cleanenv --bind $WORK/grail:/workspace/grail $SB bash -c '
  P=/root/miniconda3/envs/grail; cd /workspace/grail/imports/FoundationPose/mycpp
  rm -rf build; mkdir build; cd build
  cmake .. -DPYTHON_EXECUTABLE=$P/bin/python -Dpybind11_DIR=$($P/bin/python -c "import pybind11;print(pybind11.get_cmake_dir())") 2>&1 | tail -5
  make -j8 2>&1 | tail -8; ls -la'
echo "=== done $(date)"
