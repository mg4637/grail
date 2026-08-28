#!/usr/bin/env bash
# USDs for the split chair task.
#   static_scene : bookshelf + chair-at-placed-pose, the `add_table` prop.
#                  World coordinates are baked in, so table_pos is (0,0,0) for
#                  the basket motion and (0,0,-100) parks it for the carry.
#   basket_object: the dynamic object of the basket motion. Origin-centred - its
#                  pose comes from objects/basket.pkl, not from the mesh.
# Scales are already baked, hence --scale 1.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
cd /workspace/grail
export OMNI_KIT_ACCEPT_EULA=Yes PYTHONUNBUFFERED=1
D=/workspace/grail/data/motion_lib/grail_chair_split/common
exec >> "${D}/convert.log" 2>&1
echo "############ started $(date -u) ############"
for name in static_scene basket_object; do
    echo "--- ${name} ---"
    python -u grail/retargeting/convert_mesh.py "${D}/${name}.obj" "${D}/${name}.usd" \
        --headless --mass 1.0 --scale 1.0 --collision-approximation meshSimplification
    echo "  rc=$?"
done
echo "=== done $(date -u) ==="
