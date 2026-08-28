#!/usr/bin/env bash
# Build grail_table_v5 from the raw v4w retarget (wrists free, --zero_out_wrist
# dropped). Order is load-bearing:
#   1. penetration resolve pushes the pose out of geometry GMR never saw, but it
#      pushes the WHOLE body out, so it lifts the hands off the table too;
#   2. foot planting puts the soles back on the floor/tabletop;
#   3. the hand fix runs LAST, because (1) would otherwise undo it.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=data/motion_lib/grail_table_v4w
DST=data/motion_lib/grail_table_v5

rm -rf "${DST}"
echo "### 1/4 penetration resolve"
python data/stage3_resolve_penetration.py "${SRC}" "${DST}"
echo "### 2/4 foot planting"
for S in right left; do
  python data/stage3_fix_foot_planting.py "${DST}" --side "${S}" --write
done
echo "### 3/4 hand contact + palm flatness"
python data/stage3_fix_hand_contact.py "${DST}" --max_joint_change 1.2 --write
# The hand fix runs after the resolver and can push a palm a few mm back into
# the tabletop (measured: 9.6 mm worst). A final pass clears it, and it is safe
# to run last because at this magnitude it moves joints by <=0.006 rad - it
# cannot undo the flatness correction.
echo "### 4/4 final penetration pass"
python data/stage3_resolve_penetration.py "${DST}" "${DST}_tmp"
rm -rf "${DST}" && mv "${DST}_tmp" "${DST}"

echo "### verify"
python data/stage3_verify_reference.py "${DST}"
