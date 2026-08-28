# Implementation History

Newest session last. Times are UTC.

---

## Session: 2026-08-23 → 2026-08-25 — Stage 3 re-retarget + Stage 4 retrain (table climb)

**Starting state.** A Stage 4 SONIC finetune reached progress 0.644 / mpjpe 114.4 on the
table climb, terminating on `anchor_pos`. A later run on another machine loosened and then
disabled the 0.15 threshold, reported progress 1.0, but the rendered rollout showed the
robot slipping off the table.

**Goal.** Make the Unitree G1 climb the table. Compute: one local RTX 4090.

### Timeline

| UTC | Event |
|---|---|
| 08-23 ~17:00 | Built `data/stage3_mujoco_feasibility.py`; its stance control collapsed. Diagnosed: open-loop joint PD has no balance strategy, so a free-base humanoid topples whatever you feed it. Added `--pin_base`. |
| 08-23 ~18:00 | Found explicit PD damping unstable at MuJoCo's 2 ms step on low-inertia wrists (fake 94 % saturation, gone at 0.5 ms). Switched to `IMPLICITFAST` + `dof_damping`, matching IsaacLab's implicit actuators. |
| 08-23 ~18:30 | Pinned-base test showed 1.4 % worst-joint saturation → concluded (**wrongly**, see Corrections) that torque was never the constraint. Reverted the v2 wrist IK cost change (100 → shipped 600). |
| 08-23 19:28 | Stage 3 re-retarget → `data/motion_lib/grail_table_v3`, rc=0. |
| 08-23 ~19:35 | Measured reference penetration: soles travel **53 mm below the floor**, 116/121 frames penetrate >5 mm. |
| 08-23 19:41 | Wrote `data/stage3_resolve_penetration.py` → `grail_table_v3f`. Penetration 76.6 → 2.0 mm. |
| 08-23 19:44 | Wrote `terrain_tracking_grail.yaml` (Stage 4 reward/event changes). |
| 08-23 19:46 | 5-iteration smoke test passed. |
| 08-23 19:47 | Stage 4 training on `v3f` started (2048 envs, 6000 iters). |
| 08-24 00:56 | Training finished rc=0 (5 h 09 m, 2.9 s/iter). |
| 08-24 00:58 | Eval: **progress 0.604, mpjpe 92.2, `anchor_pos`** — worse than the 0.644 baseline. |
| 08-25 07:27 | Untruncated rollout for video (terminations off). Reports progress 1.000 / mpjpe 485.8 — the same artifact as the other machine. |
| 08-25 ~07:40 | Found and fixed a joint-order bug in the trajectory renderer (see Corrections). |
| 08-25 ~08:00 | Contact re-plant attempt → `grail_table_v4`. Unusable. |
| 08-25 08:12 | Wrote `data/stage3_contact_feasibility.py`; fixed three physics bugs in it; validated it. |
| 08-25 08:15 | Stage 4 training on `grail_table_v3` started (the reference that should have been used first). Eval armed to auto-run on exit. |
| 08-25 13:49 | Training finished rc=0 (5 h 35 m, 3.2 s/iter, 6000 iters). |
| 08-25 13:53 | Eval: **progress 0.772, mpjpe 112.2, `ee_body_pos`** — best result of the project, and it clears the pre-committed 0.644 bar. |
| 08-26 13:18 | Untruncated rollout for video. Table work stopped here; moving to the chair clip. |

### Genuine Stage 3 bugs found and fixed

1. **`imports/GMR/assets/unitree_g1/g1_mocap_29dof.xml`** — 15 joint ranges truncated.
   Hip pitch declared `-1.57 1.57` against a URDF limit of `-2.5307 2.8798`;
   `mink.ConfigurationLimit` pinned it at exactly −1.5700 for frames 72–95, the entire
   mount. All restored to URDF values.
2. **`imports/GMR/general_motion_retargeting/motion_retarget.py`** — `mink.solve_ik` called
   positionally, so `ik_limits` bound to `safety_break` and the `VelocityLimit` never
   reached the solver. Now passed as `limits=`.
3. **`grail/retargeting/retarget.py`** — root written into `qpos` without
   `configuration.update()`, so the first solve of every frame used stale FK. Also added
   `usd_scale` so the exported USD tracks the world scale.

Result: hip pin gone; worst `joint_limit` cost if tracked perfectly −2.86 → −1.09.

### Corrections — things I got wrong and later disproved

1. **"The policy never sees the last 36 % of the motion."** False. `terrain_tracking` has
   reference-state initialization on by default (`use_paired_motions: true` →
   `commands.py:3204-3210` → `motion_lib_base.py:2353-2367` draws a uniform random start
   frame). The stand-up was trained from iteration 1. This refutes
   `stage4_train_curriculum.sh`; both it and `stage4_train_slow.sh` are marked DO NOT RUN.
2. **Scaling the scene to make the climb easier is inert.** Sweeping `scene_scale` over
   {0.90…0.75} produced byte-identical joint angles — scaling the pelvis path while keeping
   bone offsets is a per-frame rigid translation.
3. **"The pinned-base test proves the reference is executable."** No. Pinning the base means
   an external constraint holds the pelvis on the reference path, so the pin does the work
   of lifting the robot. It proves the limbs can reach the poses, nothing more.
4. **The penetration fix was the main damage.** `stage3_resolve_penetration.py` enforces
   penetration ≤ 2 mm, an *inequality* that a robot floating in mid-air satisfies perfectly.
   Over the mount (frames 50–99) it turned 47/50 frames on ≥2 contacts into **21/50 frames
   touching nothing at all**. The 6000-iteration run was trained on this.
5. **Renderer joint order.** SONIC's `TrajectoryRecorder` stores `robot.data.joint_pos` in
   IsaacLab's **breadth-first** articulation order and saves no joint names; the MJCF is
   **depth-first**. Writing it into `qpos[7:36]` scrambled 27 of 29 joints. Permutation
   `[0,3,6,9,13,17,1,4,…]` now in `stage3_render_reference.py`. Validated: corrected order
   puts the soles at exactly 0.000 m while standing.
6. **Three bugs in the feasibility test itself**, each of which changed the answer:
   `mj_rne` excludes gravity (returns 0 for a static robot); `mj_forward` overwrites `qacc`;
   `mj_inverse` subtracts contact forces already resolved.
7. **"The reference is physically impossible" was too strong.** Validating the test on plain
   walking — obviously feasible — still showed 50–150 N residual. A kinematic retarget is
   never exactly dynamically consistent, so only the *comparison* is meaningful.

### Key measurements

**G1 vs the table**
| | |
|---|---|
| G1 height / mass | 1.372 m / 33.3 kg (weight 327 N) |
| pelvis / shoulder | 0.793 m / 1.085 m; arm reach 0.284 m |
| tabletop | 0.905 m = 66 % of height, **0.180 m below the shoulders** |
| work to raise COM | 147 J |

The tabletop being below shoulder height means the G1 can plant both hands on top with
shoulders above them, so the arms act as compression struts — the 25 N·m shoulder limit is
not what binds. The hardware can do this climb.

**Reference quality** (contact-force feasibility, frames 30–99; severe = >100 N of base
wrench no push-only friction-bounded contact can supply)

| reference | infeasible | severe | note |
|---|---|---|---|
| `grail_table` (v1, shipped) | 45/70 | — | soles 53 mm through the floor |
| `grail_table_v3` | 33/70 | 8 (f31, 37, **68–75**) | Stage 3 bugs fixed |
| `grail_table_v3f` | 70/70 | **52** | penetration "fix" — **trained on this** |
| `grail_table_v4` | 15/70 | 3 | feasible only because hips are 123 mm inside the table, 5× jerk — unusable |

The severe block at f68–75 is where the reference is supported **only by hip links** pressed
against the table's vertical side rail. Root cause: the source is a kid who hooks fingers
over the edge (left wrist sits 1–11 cm *below* the top surface) and braces a hip on the side
rail. The G1 has no fingers, `--zero_out_wrist` freezes the wrists, and a vertical rail gives
a horizontal contact normal that cannot carry 327 N. GMR matches joint *positions* with no
scene, no contact and no dynamics.

### Results

| run | reference | progress | mpjpe | termination |
|---|---|---|---|---|
| `terrain_release` zero-shot | `grail_table` | 0.308 | 388.2 | `ee_body_pos` |
| baseline finetune | `grail_table` | 0.644 | 114.4 | `anchor_pos` |
| 6000 iters on the damaged ref | `grail_table_v3f` | 0.604 | 92.2 | `anchor_pos` |
| **6000 iters, final** | **`grail_table_v3`** | **0.772** | 112.2 | **`ee_body_pos`** |
| untruncated (video only) | `grail_table_v3` | 1.000 *(artifact)* | 210.4 | — |

Training on `v3f`: mean reward 3.14 → 5.83, but `error_body_pos` moved only 89 → 87 mm. The
reward gain came from survival time, not tracking accuracy.

**The two 6000-iteration runs differ only in the reference** — same reward config, same
hyperparameters, same warm start. Dropping `stage3_resolve_penetration.py` moved progress
0.604 → 0.772, which confirms that pass was the damage, and matches the ordering the
contact-feasibility test predicted (v3 ≫ v3f) before either run existed.

**What the final policy does.** The mount is solved: the pelvis reaches **1.312 m** at frame
86 against a 0.905 m tabletop and holds ~1.24 m for ~15 frames. (The `v3f` policy peaked at
1.103 m and was on the floor by f110.) It then fails to **stand up** — the torso stays
pitched forward with the hands on the surface through f100, then drifts sideways and comes
off the far edge by f124, ending at 0.795 m. The termination reason moving from `anchor_pos`
to `ee_body_pos` says the same thing: the root now tracks, the limbs do not.

Success rate is still 0 — the motion is not completed — and mpjpe 112.2 is essentially the
114.4 baseline, so this is more progress at unchanged tracking quality.

### Files

**Created**
- `data/stage3_mujoco_feasibility.py` — PD replay; use `--pin_base` only
- `data/stage3_resolve_penetration.py` — penetration push-out **(harmful; see Correction 4)**
- `data/stage3_enforce_contacts.py` — contact re-plant (produced unusable `v4`)
- `data/stage3_contact_feasibility.py` — contact-force feasibility; the gate
- `data/stage3_render_reference.py` — MuJoCo renderer for motion_lib and trajectory pkls
- `data/stage3_retarget_v3.sh`, `data/stage3_verify_reference.py`
- `data/stage4_train_v3.sh`, `data/stage4_train_v3ref.sh`, matching eval scripts,
  `data/stage4_smoke_v3.sh`, `data/stage4_eval_notrunc_v3.sh`
- `imports/SONIC/.../scene/terrain_tracking_grail.yaml` — vr5point σ 0.1→0.25,
  `is_terminated` −10, `undesired_contacts` exempting climb contacts, friction
  randomisation [0.7, 1.3], small reset noise. Shipped 0.15/0.15/0.20 thresholds untouched.

**Modified** — the three GMR/retarget fixes above; `stage4_train_curriculum.sh`,
`stage4_train_slow.sh`, `stage4_post_train_eval_curriculum.sh` marked DO NOT RUN.

**Reverted** — `data/g1_smplx/gmr_smplx_to_g1.json` wrist cost back to the shipped 600.

**Motion libs** — `grail_table_v2` (superseded), `v3` (best), `v3f` (harmful), `v4` (unusable).
`grail_table` never modified.

### Outcome and stopping point

The pre-committed decision rule was: if `v3` does not clear 0.644, conclude the kid's
strategy does not transfer. It cleared it — 0.772 — so the Stage 3 bug fixes were both
necessary and productive, and the reference, not the RL setup, was the binding problem
throughout.

What remains unsolved is the stand-up on top of the table, and success rate is still 0.
**Table work stopped here by decision, to move to the chair clip.** If it is resumed, the
open leads are, in order:
1. The remaining failure is in a window the feasibility test never flagged as severely
   infeasible, so it is plausibly a training problem rather than a broken reference — more
   iterations, or higher sample density on frames 85–121, are the cheap things to try first.
2. `grail_table_v3` still penetrates the scene by up to 77 mm. The correct repair is an
   *equality* on a contact set (bodies meant to bear load sit at distance 0), not the
   inequality `stage3_resolve_penetration.py` implements. Nobody has built that.
3. Do not retry a geometric edit without running `stage3_contact_feasibility.py` on the
   result first — it correctly predicted the ranking of every reference tried here.

**Known about the chair before starting:** it is a compound carry-then-climb that no single
released policy covers. The object translates 1.96 m in xy (correlation 0.967 with the
pelvis) and its z goes 0.90 → 1.27 → 0.97 — the chair is carried, set down, and only then
stood on. Terrain tracking treats the object as static scenery; the pick-up policies cover
the carry but terminate once the chair is stood on. Zero-shot: `terrain_release` 0.116 /
122.2 (`foot_pos_xyz`), `pnp_ground_release` 0.024 / 26.3 (`object_pos_deviation`).
`grail_chair_ha` has no `object_usd/`, which `terrain_motion_dir` requires paired 1:1.

### Ops notes

`docker exec -d` with the script doing its own `exec >> logfile 2>&1`; never wrap container
runs in `timeout` (SIGTERM deadlocks Isaac Sim); one Isaac job at a time on the single GPU;
poll with `[t]rain_agent_trl`. MuJoCo rendering on the host needs `MUJOCO_GL=glfw DISPLAY=:3`
(EGL is broken — PyOpenGL pinned at 3.1.0; osmesa not installed). Container-created files are
root-owned; `chmod -R a+rX` inside the container before reading them from the host.

---

## Session: 2026-08-26 → 2026-08-27 — Chair task: carry + place + step up + basket + step down

**Decision.** Table work stopped by decision at 0.772 progress (see previous session). Moved to
the chair clip, this time asked for the full compound task: carry the chair to a bookshelf,
place it, step up, take a basket off the shelf, step down.

### Two structural blockers found before any building

1. **Two objects, one dynamic slot.** SONIC's scene builder spawns exactly one dynamic object
   prim; every reward/termination/observation hardcodes `object_root_pos[:, 0]`. Chose
   **Option A**: split the clip into two motions, each with a single dynamic object, since the
   chair and the basket are never manipulated at the same time.
2. **Two robot models, no bridge.** Terrain tracking uses `g1_model_12` (no hands); pick-and-
   place uses `g1_43dof` (dex hands). No shipped config combines them, and 29 vs 43 DoF can't be
   warm-started across. Committed to `g1_43dof` for the whole task.

### Built

- **`grail_chair_split`** — `__carry` (f0–135, object=chair) and `__basket` (f128–240,
  object=basket), each with a per-motion grasp signal and its own `meta/table_pos` (parked at
  (0,0,−100) for `__carry`, so a static chair at the placement target doesn't collide with the
  one being carried in; in place for `__basket`).
- **Bookshelf and basket placed with no new reconstruction.** Stage 2 only tracked the chair,
  but the Blender scene's `first_frame_pose.pickle["static_objects"]` already recorded both
  props' exact transforms. Derived the Blender→motion-lib frame by anchoring on the chair (the
  one object in both frames); validated independently twice (the robot's own reach lands within
  12 cm of the derived basket position; the baked basket's underside sits at the shelf's top to
  three decimals).
- **`data/stage3_basket_trajectory.py`** — the basket has no tracked motion, so its trajectory
  is derived by welding it to the grasping hand from the measured take frame (f152, 3.8 cm palm
  gap) onward.
- **`data/stage3_chair_feasibility.py`** — contact-force feasibility adapted for carried
  objects: a payload contributes an external wrench at its own COM regardless of which link
  holds it, so no mesh decomposition of the payload is needed; the chair-as-*support* case
  (once stood on) is voxelised instead, since no convex-decomposition backend (coacd/vhacd) is
  installed on this host or in the container.
- **`imports/SONIC/.../hoi/chair_grail.yaml`** — `g1_43dof`, one dynamic object via
  `MultiUsdFileCfg` one-per-env pairing (not `multi_object_per_env` — see Corrections), 3 kg
  chair, both hands in the finger contact sensors, `use_finger_primitive` +
  `use_motion_hand_actions`.

### Corrections — repeated mistakes, each fixed by measuring the actual contact surface

The basket's grasp and orientation were revised three times in this session, twice on direct,
correct user pushback:

1. **Nearest-vertex distance is not contact distance.** On these low-poly meshes (chair
   0.90 m² / ~1400 faces) a palm resting flat on the seat can be 8–11 cm from the nearest
   *vertex* while touching the surface. This produced two wrong conclusions in sequence — first
   "the carry is one-handed" (it isn't), then, after over-correcting, "the feet float 9 cm above
   the chair" (they don't). Fixed by sampling the mesh *surface* (~4 mm spacing) and querying a
   KD-tree; both errors resolved to ~0 with the same code, no threshold tuning.
2. **Basket orientation, three attempts.** (a) Derived yaw from the noisy hand-to-hand line —
   spun the basket through two spurious 90° turns, compounded by a leftover code path that wrote
   an xyzw quaternion into a wxyz field (a literal 180° flip on one frame). (b) Held the
   orientation world-fixed — matched the "no visible rotation" read from the source video, but
   the robot itself yaws 80° during this phase, so a world-fixed basket would end up pointing
   the wrong way relative to the body. (c) **Correct**: hold the basket's orientation fixed
   *relative to the robot's root yaw* — `quat(f) = Rz(yaw_root(f) − yaw_root(attach)) · quat_on_shelf`.
   Verified: 0.00° deviation from body yaw, 0.00° tilt, frame-to-frame rotation matches only the
   robot's own turn rate. The user's framing — "it needs to rotate but with the robot, not
   itself" — is what identified the right frame after two wrong ones.
3. **The two-handed basket grasp is asserted, not measured.** With orientation now correct, the
   robot's hand line runs along −X while the basket's long axis is +Y, so the hands bracket its
   short side and the right wrist sits 8–16 cm off — the same reach shortfall seen throughout
   this project. The right hand's closure in the grasp signal is a declared exception (wider
   thresholds, documented in the script), not a geometric finding.
4. **`multi_object_per_env` is incompatible with this config.** It renames prims to
   `Object_<name>`, but the grasp contact sensors and hand-frame transformer are hardcoded to
   `{ENV_REGEX_NS}/Object`; the scene failed to build. **One-per-env** `MultiUsdFileCfg` is not
   just the workaround but the better fit: `random_choice=False` gives env *k* USD[k % N] while
   `use_paired_motions` gives env *k* motion[k % N], the prim stays `/Object` so every sensor
   binds, and `commands.py` itself refuses to start unless the USD stems equal the motion keys
   in order — which also caught that the **object-USD list is derived from `robot/*.pkl`
   filenames**, not the keys inside them; `carry.pkl`/`basket.pkl` had to be renamed to the full
   motion key.
5. **The chair floats 9.8 cm after placement.** Same defect class as the table clip — measured,
   not assumed: placed-chair low point +0.092 m against a robot sole at −0.007…+0.021 on the
   floor either side of it. Fixed by dropping the chair (and the robot with it, weighted by how
   far its feet are off the floor, only after both hands release at f96) rather than the root
   alone, since the robot is standing on the seat that's moving.

### The feasibility diagnosis found a real, unresolved defect

Running `stage3_chair_feasibility.py` on both split motions surfaced the same asymmetric fault
in two places:

| phase | finding |
|---|---|
| `__carry` walk (f0–96) | right sole never within 2 cm of the floor (0/24 sampled frames); left plants normally (21/24) |
| `__basket` step-down (f62–104) | both soles airborne 5–14 cm at f77–91 (0 contacts even at 5 cm tolerance); single-foot contact elsewhere |

Checked against the source video at the corresponding frames: the kid's carry and step-down are
both ordinary, feet-planted, no-airtime motions. So this is a retargeting or reconstruction
defect, not real content — the same shape as the L/R keypoint swap documented for an earlier
clip ([[grail-keypoint-lr-swap]]).

**Root-cause investigation is open, not resolved.** Was checking Stage 2's own
`foot_contact_probs` and ViTPose ankle confidence against the retargeted foot heights to
localize the fault to reconstruction vs. Stage 3 IK, and had started an SMPL-X forward-kinematics
check of the *pre-retargeting* human mesh to settle it independently, when interrupted (blocked
on a `models/smplx/smplx` path typo, not yet fixed). This must be resolved, and the standing
one-legged defect fixed, before any Stage 4 training — no reward tuning can compensate for a
reference where one foot never touches the ground.

### Files

**Created:** `data/stage3_split_chair.py`, `data/stage3_static_props.py`,
`data/stage3_basket_trajectory.py`, `data/stage3_reground_chair.py`,
`data/stage3_chair_feasibility.py`, `data/stage3_convert_chair_props.sh`,
`data/stage4_smoke_chair.sh`, `imports/SONIC/.../hoi/chair_grail.yaml`.
Extended `data/stage3_grasp_signal.py` (surface-sampled distance, per-object signals via
`--object_dir`/`--suffix`) and `data/stage3_render_reference.py` (moving/mocap objects, static
props with position, motion-key selection for multi-motion libraries).

**Motion libs:** `grail_chair_v2` (single-clip, 43-DoF-ready retarget with the three Stage 3
fixes and a corrected grasp signal), `grail_chair_split` (the two training motions).
`grail_chair` / `grail_chair_ha` untouched.

### Open at end of session

- Root-cause the one-legged defect (Stage 2 reconstruction vs Stage 3 retargeting) before
  fixing it — the fix differs completely depending on which.
- The bookshelf's mesh is a low-poly proxy with no open shelves — fine for collision, would
  matter if a future pass needs the robot to reach *into* it.
- No Stage 4 training run yet on the chair task; the feasibility gate has not been cleared.

### Ops notes (additional)

Voxelising a support mesh (`trimesh.voxelized`) is a usable stand-in for convex decomposition
when neither `coacd` nor `vhacd` is installed (confirmed absent on host and in container): 426
boxes at 4 cm pitch reproduces a chair's seat/legs/gap well enough for contact-force
feasibility. `ElementTree`'s `Element.find(...) or default` is a bug, not an idiom — an
`Element` with no children is falsy, so an existing tag is silently shadowed by a duplicate.

---

## Session: 2026-08-27 — Table video: two corrected claims, foot-placement diagnostic built

Two things the user caught, both worth recording precisely rather than quietly fixed.

**"Fingerless" was imprecise.** The table robot (`g1_model_12`, MJCF `g1_29dof_rev_1_0.xml`) has
zero finger joints past the wrist — the arm ends in one rigid mesh, `left/right_rubber_hand`,
sculpted to look hand-shaped but with nothing that articulates. Correct description: **rigid,
non-articulated hand, no grasping capability** — not "fingerless" (which reads as "no hand at
all" against a video that visibly shows hand-shaped geometry). Confirmed by contrast against
`g1_43dof` (used for the chair task), which has 14 real finger joints on the actual training
robot (`g1_43dof_s3.usda`), driven by a 2-scalar open/close primitive
(`use_finger_primitive` + `use_motion_hand_actions`).

**The claimed root cause of the table failure was wrong, on two successive attempts, until
actually checked against footage and geometry:**
1. First claim: "the kid hooks fingers over the table edge; a rigid hand can't hook an edge."
   Built from a geometric inference (retargeted wrist sitting below the tabletop surface), never
   from watching the source video. Watching it: **the mount is plain palms-flat climbing, no
   finger use at all** — flat hands pushing up, legs swinging up one at a time, kneel, stand.
2. Second claim: "the policy mounts fine but never completes the stand-up (weight stuck on the
   hands)." Also unverified speculation.
3. **What the user identified, and measurement confirmed:** the policy's feet are placed at the
   wrong location relative to the table — short of the surface and/or off its edge — not stuck
   mid-recovery. Measured against the table's real footprint (x∈[−1.00,1.15], y∈[−0.65,0.67],
   top z=0.905): the left foot sits at y=0.62–0.88 against an edge at 0.67 (**outside the
   footprint** for much of the window), and even when horizontally over the table, both feet
   hover 12–19 cm below the surface. The reference's own feet land cleanly at 0.85–0.89 vs 0.905.
   This is the real, load-bearing cause of the slide-off in the video.

**Clarified precisely what the "notrunc" video is.** Same checkpoint, same reference
(`grail_table_v3`) as the scored eval (progress 0.772 / mpjpe 112.2, shipped 0.15/0.20
thresholds, unmodified). The video is a *second* eval of that same checkpoint with
`anchor_pos`/`anchor_ori_full`/`ee_body_pos`/`foot_pos_xyz` all raised to 100.0 so the episode
doesn't truncate at the point the scored run would have ended it. `progress=1.0` in that run is
an artifact of no termination being active, not evidence of success — same artifact flagged
earlier in this project. The 0.15 threshold was never removed from training or from the number
that matters.

**Built `data/stage4_foot_placement_diagnostic.py`.** Reproduces the exact
`tracking_vr_5point_local` reward (pelvis-anchor local frame, 5 body points, std=0.25,
`terrain_tracking_grail.yaml`'s value) per point, next to a table-footprint horizontal-margin
and vertical-gap measure for each foot, averaged across all 8 scored-eval trajectories (verified
near-identical, max 2 cm apart — a deterministic policy, not per-env noise).

Findings:
- Confirms the geometric failure with numbers: left foot outside the table footprint on 24/42
  sampled frames (worst margin −0.41 m), right foot on 14/42 (−0.27 m); once nominally over the
  table, both feet sit 0.12–0.19 m below the surface against the reference's 0.04–0.06 m.
- **Refutes a reward-dilution hypothesis proposed in this same investigation.** Expected the
  5-point tracking reward's per-point average to be hiding bad foot tracking behind good
  torso/wrist tracking. Measured instead: the **wrists** are tracked worse than either foot for
  most of the window (mean squared error 0.037–0.048 vs 0.024–0.027) — the whole body is
  imprecisely tracked during the mount, not just the feet, so dilution isn't the mechanism.
  Retracted rather than kept because it sounded plausible.
- What the diagnostic does support: the reward has **no notion of the table's physical
  geometry** anywhere in the stack — it only ever compares to the reference's own foot position,
  never to the table's edges or surface — and `std=0.25` (loosened earlier in this project to
  give gradient at climb-scale error) still pays ~0.53 reward for a 20 cm single-axis miss, so it
  doesn't sharply punish a placement error of the size actually observed.

**No fix built.** This was diagnosis only, as asked. A candidate fix — an edge-margin or
contact-aware foot-placement term, since nothing today measures "is this foot on the usable
part of the surface it's meant to land on" — is proposed, not implemented or tested. Whether
that or simply more training iterations would close the gap is unknown; no ablation has been run.

---

## Session: hands flat on the table (Option B) + a feet-only reward (Option D)

Two fixes built in one pass, both aimed at what the user identified by watching the rollout:
the robot cannot place its foot on top of the table, and it plants its hands "upside down"
where the child's are flat.

### B. `grail_table_v5` — a reference with flat palms

Pipeline in `data/stage3_build_v5.sh`, run over the raw `grail_table_v4w` retarget (produced by
`data/stage3_retarget_v4w.sh`, which is v3 with `--zero_out_wrist` dropped — that flag froze
wrist dofs 19–21/26–28 at zero, so **no IK could orient the palms at all**). Order is
load-bearing and was established by measurement, not guessed:

1. `stage3_resolve_penetration.py` — pushes the pose out of geometry GMR never saw. It pushes
   the *whole body* out, so it also lifts the hands off the table (root moved 50.4 mm mean).
2. `stage3_fix_foot_planting.py` (both legs) — puts the soles back down.
3. `stage3_fix_hand_contact.py` (**new**) — the palm fix; must run after (1), which would
   otherwise undo it.
4. A final `resolve_penetration` pass — (3) reintroduces up to 9.6 mm of penetration, and at
   that magnitude the resolver corrects it while moving joints ≤0.006 rad, so it cannot undo
   the flatness work.

**Result — palm flatness `|palm_normal · world_up|`, each hand over its own contact window:**

| | left | right |
|---|---|---|
| `grail_table_v3f` | 0.18 (worst 0.01) | 0.17 (worst 0.01) |
| `grail_table_v4w` | 0.40 (worst 0.12) | 0.24 (worst 0.00) |
| **`grail_table_v5`** | **0.94 (worst 0.91)** | **0.82 (worst 0.72)** |

Soles −0.002 m; final soles 0.910/0.902 against the 0.905 m tabletop; root path untouched;
peak joint velocity 5.92 → 6.75 rad/s, i.e. the correction stays inside the motion's own
dynamic range.

**A measured limit, stated plainly rather than papered over.** The G1 arm is **0.284 m**
(shoulder→wrist), but from f70 the robot's shoulder sits **0.43–0.59 m** above the tabletop.
Its hands therefore *cannot* reach the table through the stand-up, however the child's could —
the child's shoulder is lower relative to the same table. This is a body-proportion difference,
not a retargeting defect. So past the reach envelope the fix rotates the palm flat and **holds
the hand on its existing trajectory**; hand-to-table distance is essentially unchanged
(0.081 → 0.083 m left, 0.126 → 0.124 m right). **The palms are what got fixed, not the contact.**

**Four wrong turns, each caught by measurement and corrected:**

- *Mis-scaled cost.* First version used raw squared metres, making the flatness term ~50× the
  contact term; it fixed the palms and left the hands off the surface entirely. Both terms are
  now normalised by the tolerance that matters for them (1 cm of height, 0.1 of alignment).
- *All-or-nothing rejection.* A per-frame cap on joint change originally **discarded** any frame
  needing more, leaving uncorrected frames between corrected neighbours — a discontinuity worse
  than the original defect. Now the step is scaled down to the cap and kept **only if FK
  verifies it improves**, since scaling a nonlinear IK solution is not guaranteed to.
- *Post-hoc smoothing made frames worse.* Solving each frame independently then averaging
  neighbouring corrections (`uniform_filter1d`) produced blends worse than either input —
  worst-case left flatness went 0.12 → **0.01**, i.e. the written motion was worse than the
  original. Replaced with temporal coherence *in the solve*: each frame warm-starts from the
  previous frame's accepted correction plus a continuity term. Post-hoc smoothing removed.
- *Unconstrained flatness made the arms drift.* Once the contact term was gated off for
  unreachable frames, nothing held the hand in place and the right hand swung **0.126 → 0.273 m**
  away from the table. Out of reach, the contact term is now replaced by a hold-in-place term.

The reporting was also changed to measure **the motion actually written** — after blending and
joint-limit clipping — rather than the optimiser's own per-frame numbers. That change is what
exposed the smoothing regression; the optimiser's numbers had looked fine.

### D. `tracking_feet_pos` — a feet-only tracking term

`tracking_local_point_subset_error` in `rewards.py` (plus a `RewardsCfg` field — that class is a
dataclass with declared fields, which is why the first launch died with
`TypeError: RewardsCfg.__init__() got an unexpected keyword argument`). It is
`tracking_local_vr_5point_error` restricted to a subset of the reward points; wired up in
`terrain_tracking_grail.yaml` as change **6**, weight 2.0, `std=0.20`, `point_indices: [3, 4]`
(the two `ankle_roll_link`s).

Rationale: the 5-point term averages all five points' squared error *before* the exponential, so
the feet are one fifth of one term — which is why the diagnostic found a foot up to 0.41 m
outside the footprint on frames the combined term scored as healthy. `std=0.20` is picked for
the failure's scale: `exp(−d²/σ²)` is 0.37 at 20 cm and still 0.015 at 41 cm, so there is signal
across the whole observed miss (at the shipped σ=0.1, a 20 cm error scores 0.018 and 41 cm is
numerically flat).

Note this needs **no** table geometry, which is a correction to the fix proposed at the end of
the previous session ("an edge-margin or contact-aware foot-placement term"). The reference's own
feet land inside the footprint, so tracking them tightly *is* the edge-awareness.

### Training

`data/stage4_train_v5.sh` — 2048 envs, 6000 iterations (matching `table_v3` exactly so the
comparison is like-for-like), warm-started from `table_v3/last.pt`. Baselines to beat:
progress 0.644 / mpjpe 114.4.

**Not yet known:** whether either change moves the result. Nothing about B or D has been
validated by a trained policy — B is verified only as reference geometry, D only as a reward
function that loads. The claim being tested is that flat palms plus a feet-specific gradient
improve foot placement on the tabletop; that is a hypothesis until the eval is rendered.

**Status at end of session.** `stage4_train_v5.sh` launched detached and confirmed running:
`tracking_feet_pos` registered at weight 2.0 and earning reward (0.042 → 0.248 over the first
iterations), ETA ~5.7 h for 6000 iterations. `stage4_post_train_eval_v5.sh` is also launched and
parked, waiting for the trainer to exit before evaluating at the shipped 0.15/0.15/0.20
thresholds and rendering the video.

Two launch bugs worth recording, both mine, both caught: `RewardsCfg` is a dataclass with
declared fields, so a new reward term needs a field added there as well as a config entry (the
first launch died on `TypeError: unexpected keyword argument 'tracking_feet_pos'`); and
`pgrep -c`/`pgrep -q` without `-f` match the process *name* only, which twice produced a false
"the training died" reading against a perfectly healthy run.

### Result: the v5 run REGRESSED, and change D is the likely cause

Training ran all 6000 iterations; the chained eval and a notrunc render both completed.

| run | progress | mpjpe | reason |
|---|---|---|---|
| `table_ft` (original baseline) | 0.644 | 114.4 | anchor_pos |
| `table_v3` | 0.604 | 92.2 | anchor_pos |
| `table_v3ref` (best so far) | **0.772** | 112.2 | ee_body_pos |
| **`table_v5`** | **0.280** | 184.2 | ee_body_pos |

Notrunc mpjpe: `table_v3ref` 210.4 → `table_v5` **606.4**.

**The v5 policy never climbs.** From the recorded eval trajectories:

| | root z start → end | max | x travel |
|---|---|---|---|
| reference | 0.779 → 1.692 | 1.693 | +1.288 m |
| `table_v3ref` policy | 0.762 → 1.240 | 1.312 | +1.165 m |
| **`table_v5` policy** | 0.778 → **0.780** | 0.780 | +0.551 m |

Its pelvis never leaves standing height. It walks about half the approach and stops.

**Change B is not responsible, and this was measured rather than assumed.** Comparing the two
references by FK frame-by-frame, `grail_table_v5` and `grail_table_v3f` are nearly identical in
*position*: root differs 0.001 m mean / 0.010 m max, and **no end-effector differs by more than
0.15 m on any frame** (left wrist 0.009 m mean, right wrist 0.015 m, ankles 0.002/0.009 m). The
difference is almost entirely wrist *rotation* — `wrist_roll`/`wrist_yaw` by ~0.5 rad mean —
which is what flattening the palms does, and which barely moves the wrist-link origin that
`ee_body_pos` measures. A reference this close to v3f cannot explain 0.772 → 0.280.

**Change D is the probable cause, and the training log shows the mechanism.** Final-iteration
termination rates:

| | time_out | anchor_pos | ee_body_pos | foot_pos_xyz |
|---|---|---|---|---|
| `table_v3` | 0.415 | 0.360 | 0.471 | 0.374 |
| `table_v5` | **0.955** | 0.005 | 0.027 | 0.016 |

v5's episodes almost never terminate, so they run roughly twice as long, which is why *every*
episode-sum reward roughly doubled (mean return 5.92 → 12.96; `action_rate_l2` doubled in
magnitude too, which is the giveaway that this is episode length rather than better tracking).

`tracking_feet_pos` at weight 2.0 is dense, always earnable by keeping the feet near the
reference, and — unlike the mount — costs nothing to collect while standing on the floor.
Together with `is_terminated: -10` it raised the value of simply surviving above the value of
attempting the climb. The policy took the safe option. **This is the same class of failure that
`is_terminated` was added to prevent, arriving from the opposite direction:** the earlier config
made quitting competitive, and this one makes never-starting competitive.

Worth noting the training metrics look *excellent* in isolation — near-zero terminations, double
the return, every tracking term up. Read alone they would have been reported as a success. The
eval is what exposed it.

**Recommended next step, not yet run:** re-train on `grail_table_v5` with `tracking_feet_pos`
either removed entirely (isolating B, which is the clean A/B against `table_v3ref`'s 0.772) or
gated to the mount phase so it cannot be farmed while walking on flat ground. Keeping B and
dropping D is the first thing to try, since B is verified reference geometry and D is the
change with a measured mechanism for the regression.
