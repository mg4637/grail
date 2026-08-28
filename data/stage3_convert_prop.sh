#!/usr/bin/env bash
# Convert a baked static prop mesh to USD for SONIC's `add_table` slot.
# Scale is already baked into the OBJ by data/stage3_static_props.py, so --scale 1.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail
export OMNI_KIT_ACCEPT_EULA=Yes PYTHONUNBUFFERED=1
D=/workspace/grail/data/motion_lib/grail_chair_v2/common
exec >> "${D}/convert_prop.log" 2>&1
echo "############ prop conversion started $(date -u) ############"
for name in bookshelf; do
    echo "--- ${name} ---"
    python -u grail/retargeting/convert_mesh.py \
        "${D}/${name}.obj" "${D}/${name}.usd" \
        --headless --mass 1.0 --scale 1.0 \
        --collision-approximation meshSimplification
    echo "  rc=$?"
done
echo "=== done $(date -u) ==="
