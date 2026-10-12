"""Lesson 2: build a custom wide 3-step staircase in GRAIL's asset format.

Same box-stacking geometry as grail.pipelines.gen_terrain.generate_stair_pieces,
but with fixed, generous dimensions instead of random ones.

Run from the GRAIL root:
  docs_learning/.venv/bin/python docs_learning/lessons/02_wide_stairs.py
"""
import random
import sys
from pathlib import Path

# Python puts this script's folder on sys.path, not the current directory, so
# add the GRAIL repo root (two levels up) to make `import grail` work.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from grail.pipelines.gen_terrain import _box_mesh, _random_color, export_terrain

NUM_STEPS = 3
RISE = 0.12      # height of each step (m)
TREAD = 0.35     # depth of each step going up AND coming down (m); top platform = 2*TREAD
WIDTH_Y = 1.5    # staircase width (m)
OUT = str(_ROOT / "data/syn_stairs/stairs_wide3")

random.seed(0)  # only affects the colours
rises = [RISE] * NUM_STEPS
treads_up = [TREAD] * NUM_STEPS
treads_down = [TREAD] * NUM_STEPS
total_length = sum(treads_up) + sum(treads_down)

pieces, start_up = [], 0.0
for i, rise in enumerate(rises):
    x_min = start_up
    x_max = total_length - sum(treads_down[:i])
    sx = (x_max - x_min) / 2.0
    verts, quads = _box_mesh(x_min + sx, 0.0, sum(rises[:i]) + rise / 2.0, sx, WIDTH_Y / 2.0, rise / 2.0)
    pieces.append((verts, quads, _random_color(), True))
    start_up += treads_up[i]

export_terrain(pieces, OUT, "stairs_wide3")
print("wrote", OUT, "| total length", total_length, "| top platform depth", treads_up[-1] + treads_down[-1])
