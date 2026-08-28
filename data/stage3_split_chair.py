#!/usr/bin/env python
"""Split the chair clip into two single-dynamic-object motions.

WHY SPLIT
---------
The clip manipulates two objects - the chair (carried f12-96) and the basket
(taken off the shelf at f152) - but SONIC drives exactly one. The motion lib
format carries an object axis and observations are shaped
(num_envs, max_num_objects, 3), yet at reset `commands.py` positions ONE object
and parks every other prim 100 m underground, and every reward, termination and
observation hardcodes `object_root_pos[:, 0]`.

The way through with no engine changes is a structural fact about the clip: the
two objects are never manipulated at the same time. So each phase becomes its
own motion with its own single dynamic object:

    carry   f0..135    object = chair    (pick up, carry 1.96 m, place, step up)
    basket  f128..240  object = basket   (reach, take it, step down)

WHAT MAKES THIS FIDDLY
----------------------
`table_pos`/`table_quat` are per-motion (read from meta/<key>.pkl), but
`table_usd_path` is GLOBAL - one static mesh for the whole experiment. The two
phases need different static geometry: the carry needs an empty floor, while the
basket phase needs the chair present as something to stand on.

Resolved by baking ONE static USD holding the bookshelf AND the chair frozen at
its placed pose, then using the per-motion position to switch it in and out:

    basket  table_pos = (0,0,0)      -> shelf + chair exactly where they belong
    carry   table_pos = (0,0,-100)   -> parked underground, out of the way

The carry phase therefore has no bookshelf, which is fine: the robot walks
toward it and sets the chair down 0.44 m in front, never touching it. Parking is
required, not merely tidy - a static chair sitting at the placement target would
collide with the chair being carried into it.

Usage:
    python data/stage3_split_chair.py <src_lib> <dst_lib> [--carry_end 135] [--basket_start 128]
"""
import argparse, glob, json, os, shutil
import numpy as np, joblib, trimesh
from scipy.spatial.transform import Rotation as R

PARK = [0.0, 0.0, -100.0]


def slice_motion(M, lo, hi):
    out = {}
    for k, v in M.items():
        a = np.asarray(v)
        out[k] = a[lo:hi].copy() if (a.ndim >= 1 and a.shape[0] == M["dof"].shape[0]) else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--carry_end", type=int, default=135)
    ap.add_argument("--basket_start", type=int, default=128)
    ap.add_argument("--chair_frozen_at", type=int, default=120,
                    help="frame whose chair pose is baked into the static USD")
    a = ap.parse_args()

    rp = glob.glob(f"{a.src}/robot/*.pkl")[0]
    base_key = list(joblib.load(rp))[0]
    M = list(joblib.load(rp).values())[0]
    chair = list(joblib.load(glob.glob(f"{a.src}/objects/*.pkl")[0]).values())[0]
    basket = list(joblib.load(glob.glob(f"{a.src}/objects_basket/*.pkl")[0]).values())[0]
    props = json.load(open(f"{a.src}/common/static_props.json"))
    T = len(M["dof"])

    keys = {"carry": f"{base_key}__carry", "basket": f"{base_key}__basket"}
    spans = {"carry": (0, a.carry_end), "basket": (a.basket_start, T)}

    for sub in ("robot", "objects", "meta", "common"):
        os.makedirs(os.path.join(a.dst, sub), exist_ok=True)

    for name, (lo, hi) in spans.items():
        key = keys[name]
        rm = slice_motion(M, lo, hi)
        # Each phase grips a different object, so each needs its own finger
        # signal. stage3_grasp_signal.py writes the basket's as
        # hand_action_*_basket; promote it for the basket motion and drop the
        # unused pair from both, so what SONIC reads is unambiguous.
        if name == "basket":
            for side in ("left", "right"):
                rm[f"hand_action_{side}"] = rm.pop(f"hand_action_{side}_basket")
        else:
            for side in ("left", "right"):
                rm.pop(f"hand_action_{side}_basket", None)
        # The FILENAME matters, not just the key inside. modular_tracking_env_cfg
        # derives the object-USD list from robot/*.pkl filenames
        # (motion_stems = splitext(f)[0]) and commands.py then aborts unless
        # those stems equal the motion keys in order. So robot/, objects/ and
        # object_usd/ must all be named for the motion key.
        joblib.dump({key: rm}, os.path.join(a.dst, "robot", f"{key}.pkl"))
        src_obj = chair if name == "carry" else basket
        om = {}
        for k, v in src_obj.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
                om[k] = v[lo:hi].copy()
            elif isinstance(v, dict) and set(v) >= set(range(T)):
                # per-frame dict (e.g. contact_points_{left,right}_hand, keyed
                # 0..T-1 globally) - re-key to the LOCAL 0..(hi-lo-1) range the
                # sliced motion uses. Missed originally: only ndarray fields were
                # resliced, so the basket motion shipped with contact_points
                # still keyed 0..239 against a 112-frame motion - every lookup
                # at the actual playback frames (0..111) missed, which is what
                # sample_before_contact's own "no contact data available" error
                # was catching.
                om[k] = {i - lo: v[i] for i in range(lo, hi)}
            else:
                om[k] = v
        joblib.dump({key: om}, os.path.join(a.dst, "objects", f"{key}.pkl"))
        meta = {"object_name": "chair" if name == "carry" else "basket",
                "scene_scale": 1.0, "smpl_height": 1.66,
                "table_pos": [0.0, 0.0, 0.0] if name == "basket" else PARK,
                "table_quat": [1.0, 0.0, 0.0, 0.0]}
        joblib.dump(meta, os.path.join(a.dst, "meta", f"{key}.pkl"))
        print(f"  {name:<7} frames {lo:3d}..{hi:3d} ({hi-lo:3d})  object={meta['object_name']:<6}"
              f"  table_pos={meta['table_pos']}")

    # --- the one static USD: bookshelf + chair frozen at its placed pose ---
    shelf = trimesh.load(props["bookshelf"]["obj"])
    if isinstance(shelf, trimesh.Scene):
        shelf = shelf.dump(concatenate=True)
    shelf.vertices = np.asarray(shelf.vertices) + np.asarray(props["bookshelf"]["pos"])

    cm = trimesh.load("results/generation/mesh/ComAsset/chair/model.obj")
    if isinstance(cm, trimesh.Scene):
        cm = cm.dump(concatenate=True)
    sc = np.asarray(chair["scale"]).reshape(-1)
    cv = np.asarray(cm.vertices) * (sc[:3] if sc.size >= 3 else float(sc[0]))
    f0 = a.chair_frozen_at
    cq = np.asarray(chair["root_quat"])[f0, 0]
    cp = np.asarray(chair["root_pos"])[f0, 0]
    cm.vertices = cv @ R.from_quat(cq[[1, 2, 3, 0]]).as_matrix().T + cp

    static = trimesh.util.concatenate([shelf, cm])
    sp = os.path.join(a.dst, "common", "static_scene.obj")
    static.export(sp)
    e = np.asarray(static.vertices)
    print(f"\n  static scene -> {sp}")
    print(f"    bookshelf + chair@f{f0} (chair at {np.round(cp,3)})")
    print(f"    bbox x{e[:,0].min():6.2f}..{e[:,0].max():5.2f} y{e[:,1].min():6.2f}..{e[:,1].max():5.2f}"
          f" z{e[:,2].min():6.2f}..{e[:,2].max():5.2f}")

    for f in glob.glob(f"{a.src}/common/*"):
        if os.path.isfile(f):
            shutil.copy(f, os.path.join(a.dst, "common", os.path.basename(f)))
    if os.path.isdir(f"{a.src}/bps"):
        shutil.copytree(f"{a.src}/bps", os.path.join(a.dst, "bps"), dirs_exist_ok=True)
    print(f"  written {a.dst}")


if __name__ == "__main__":
    main()
