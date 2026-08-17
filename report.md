# GRAIL Replication — Status Report

**Date:** 2026-08-14
**Repo:** `/home/mert/PycharmProjects/GRAIL`
**Machine:** RTX 4090 (24.5 GB), i7-13700K (24 threads), 125 GB RAM
**Source material:** two Kling-AI generated clips, 24 fps

| Stage | Name | Status |
|---|---|---|
| 0–1 | Asset prep / character fitting | ✅ done |
| 2 | `recon_4dhoi` — 4D HOI reconstruction | ✅ **both clips valid** |
| 3 | `retargeting` — SMPL-X → Unitree G1 | ✅ **3 motion libs built** |
| 4 | `tracking` — RL in Isaac Lab on SONIC | ⚠️ **large improvement, still below export threshold** |
| 5 | Sim-to-real | ⛔ not started (needs real G1 hardware) |

---

## 1. Where everything lives

### Stage 2 outputs — reconstruction

| Path | What it is |
|---|---|
| `results/generation/4dhoi_recon_smplx_valid/ComAsset/<obj>/<clip>/hoi_data/hoi_data.pkl` | **The Stage 2 deliverable.** SMPL-X body + hands + object 6-DoF, world space. |
| `.../<clip>/result_vis/recon_result.mp4` | Reconstruction rendered on its own. |
| `.../<clip>/result_vis/recon_comparison.mp4` | **Side-by-side vs. source video — the thing to actually watch.** |
| `.../<clip>/result_vis/recon_result_top_view.mp4` | Top-down view; good for spotting drift. |
| `.../<clip>/result_vis/recon_result.html` | Interactive 3D viewer (18–36 MB, open in browser). |
| `.../<clip>/mesh_data/` | Object mesh + textures the reconstruction was solved against. |
| `results/generation/4dhoi_recon_smplx/.../log.txt` | **Validity-check log** — the six pass/fail thresholds. |
| `results/generation/4dhoi_recon_cache/` | 1.3 GB of intermediates: depth, SAM2 masks, contact labels, interaction. |
| `results/generation/foundation_pose_output/` | 273 MB of per-frame object-pose tracking debug renders. |
| `results/generation/videos_kling/ComAsset/<obj>/` | The original input videos. |

The two clips are:
- `table/kid_indoor2-climb-table_rand00001` — 121 frames, 5.04 s
- `chair/kid_indoor2-robot-manipulation_rand00001` — 240 frames, 10.0 s

### Stage 3 outputs — retargeted G1 motion

| Path | What it is |
|---|---|
| `data/motion_lib/grail_table/` | Table climb library (63 MB — includes USD + textures) |
| `data/motion_lib/grail_chair/` | Chair library (172 KB) |
| `data/motion_lib/grail_chair_ha/` | Chair **with hand actions** (276 KB) |

Inside each library:
- `robot/*.pkl` — **the Stage 3 deliverable.** G1 kinematic reference: `root_trans_offset`, `root_rot`, `dof (N,29)`, `pose_aa (N,30,3)`, `fps`.
- `objects/*.pkl` — object trajectory in the same frame.
- `meta/*.pkl` — per-motion metadata SONIC reads.
- `object_usd/*.usd` — the object as Isaac Sim scene geometry. **Required, paired 1:1 by filename stem with `robot/*.pkl`.**
- `bps/*.npy` — Basis Point Set 10-D object encoding. Needed by the `pnp_*` policies, unused by terrain.

> ⚠️ `grail_chair_ha` has **no `object_usd/`**, so it cannot be used with the terrain path at all.
> Load these with `joblib.load()`, not `pickle.load()` — they are joblib-compressed and plain pickle raises `invalid load key`.

### Stage 4 outputs — tracking

| Path | What it is |
|---|---|
| `data/stage4_train/table/` | 2.6 GB finetune run directory |
| `data/stage4_train/table/train.log` | 23 MB training log (**append mode — see caveat §5**) |
| `data/stage4_train/table/model_step_00{1000..6000}.pt` | Checkpoints every 1000 iters, 388 MB each |
| `data/stage4_train/table/last.pt` | Rolling checkpoint, rewritten every 50 iters |
| `data/stage4_train/table/config.yaml` | Fully resolved Hydra config for the run |
| `data/stage4_eval/table_ft/` | **Post-finetune eval — the Stage 4 result** |
| `data/stage4_eval/table_ft/post_train_eval.log` | Contains the `MOTION_RESULT` verdict line |
| `data/stage4_eval/table_ft/00000{0..7}.mp4` | Rendered rollouts, one per env |
| `data/stage4_eval/table_ft/00000{0..7}.trajectory.pkl` | Recorded trajectories, 81 frames @ 25 fps |
| `data/stage4_eval/table_ft/scene_metadata.json` | Index mapping env → video/trajectory |
| `data/stage4_eval/{table,chair,chair_terrain}/` | Zero-shot baseline rollout videos (see §4.1) |

### Scripts written during this work

| Path | Purpose |
|---|---|
| `data/stage4_train_resume.sh` | Detached trainer. Runs **inside** the container; survives logoff. |
| `data/stage4_post_train_eval.sh` | Watcher that waits for the trainer to exit, then evaluates `last.pt`. |

Both live on the mounted volume (`data/` → `/workspace/grail/data/`) precisely so they can be launched with `docker exec -d` and have no host-side process to die.

### Upstream checkpoints

`imports/SONIC/models/` — `terrain_release`, `terrain_stairs_warmstart`, `pnp_ground_release`, `pnp_ground_warmstart`, `pnp_table_release`, `pnp_table_warmstart`, `sonic_manipulation_base`.

---

## 2. Stage 2 — 4D HOI reconstruction

**What it does:** GENMO estimates per-frame SMPL-X body pose in camera space and transforms it to world space using the *known* camera extrinsics; WiLoR refines the hands; FoundationPose tracks the object 6-DoF from a known initial pose; then a joint optimization refines human + object together against keypoints, projection, MoGe-2 metric depth, VLM contact labels, and smoothness.

The whole method depends on the **privileged 3D configuration** — mesh, intrinsics, extrinsics, metric scale, initial object pose — being correct. When that metadata is wrong the stage produces plausible-looking garbage rather than erroring.

### Results — both clips PASS all six checks

| Check (threshold) | table climb | chair manip |
|---|---|---|
| object mask difference (< 0.3) | **0.038** | **0.156** |
| keypoint tracking (< 2.0) | **0.929** | **0.844** |
| contact (< 5.0) | **3.465** | **2.748** |
| initial human position error (< 0.5 m) | **0.236** | **0.068** |
| initial penetration (≥ 0.02) | 0.872 | 0.055 |
| sufficient motion (avg max vel ≥ 0.01) | 0.051 | 0.046 |

**What these mean.** *Mask difference* compares the FoundationPose render against the SAM2 segmentation — it is the primary failure filter, and passing means the tracked object actually sits where the object is in the video. *Keypoint tracking* is 2D reprojection agreement with ViTPose. *Contact* scores whether VLM-labelled contacts are geometrically satisfied. *Initial penetration* guards against the human starting inside the object — the chair's 0.055 is comfortably above threshold but noticeably tighter than the table's 0.872, which is expected for a clip that begins with hands already on the object.

### Payload integrity

Both `hoi_data.pkl` files carry `poses (N,165)`, `betas (1,10)`, `trans (N,3)`, `left_hand_pose`/`right_hand_pose (N,45)`, `vitpose (N,17,3)`, `foot_contact_probs (N,4)`, and object `obj_R` / `obj_t (N,3)`. **Zero NaNs in any array.**

### Motion content

- **Table:** pelvis z 0.751 → 1.642 m, 1.32 m of xy travel, object **completely static** (z pinned at 0.034, 0.000 m travel). A clean, pure 0.89 m climb.
- **Chair:** pelvis z 0.759 → 0.815 with a 1.221 peak; object z 0.900 → 1.297 → 0.97 and **1.96 m of xy travel**, correlating 0.967 with the pelvis. The chair is *carried*, set down, and only then stepped on.

That chair finding matters and is covered in §4.3.

---

## 3. Stage 3 — retargeting to Unitree G1

**What it does:** GMR maps the SMPL-X motion onto the G1's 29-DoF skeleton, producing the joint-space kinematic reference the RL stage will try to physically realize. Because the character was pre-fitted to G1 proportions in Stage 1, morphology mismatch is reduced and contacts survive better.

| | table | chair | chair_ha |
|---|---|---|---|
| frames / fps / duration | 121 / 24 / 5.04 s | 240 / 24 / 10.0 s | 240 / 24 / 10.0 s |
| DOFs | 29 | 29 | 29 + 2 hand actions |
| root z start → end | 0.759 → 1.651 | 0.764 → 0.820 | same |
| root xy travel | 1.329 m | 1.740 m | same |
| root speed mean / max | 0.62 / 1.42 m/s | 0.54 / 1.35 m/s | same |
| ‖Δdof‖ per frame — mean / p99 / max | 0.025 / 0.141 / **0.201** rad | 0.024 / 0.137 / **0.531** rad | same |
| NaNs | 0 | 0 | 0 |

**Reading the artifact check.** Root trajectories match Stage 2 to within millimetres, so the retarget preserved the motion. The concerning number is the chair's **0.53 rad single-frame joint jump** (≈ 12.7 rad/s at 24 fps) against the table's well-behaved 0.20 max — a snap during the carry phase. Median motion is smooth in both (mean Δ ≈ 0.024 rad/frame); this is one isolated discontinuity, not pervasive jitter.

`smpl_joints (N,24,3)` is all zeros in every file — an unused field in this export path, not a failure.

---

## 4. Stage 4 — task-general tracking

**What it does:** turns the kinematic reference into *physically executable* actions by training in Isaac Lab on top of the SONIC whole-body controller. Two paths exist and the choice is made by the **termination set, not by the object**:

- `pnp_*` (pick-and-place) terminates on `object_pos_deviation`, `object_z_pos_deviation`, `grasp_failure_after_contact`.
- `terrain_release` (scene/terrain) terminates on `anchor_pos`, `anchor_ori_full`, `ee_body_pos`, `foot_pos_xyz`, `time_out` — and has **no object terms at all**, treating the object as static scenery.

### 4.1 Zero-shot baselines — all FAIL

| motion | policy | progress | mpjpe | termination |
|---|---|---|---|---|
| table climb | `terrain_release` | 0.308 | 388.2 | `ee_body_pos` |
| chair | `terrain_release` | 0.116 | 122.2 | `foot_pos_xyz` |
| chair | `pnp_ground_release` | 0.024 | 26.3 | `object_pos_deviation` |

Rollout videos survive in `data/stage4_eval/table/`, `chair_terrain/`, and `chair/` respectively (mapping inferred from directory names). The **logs for these three runs are gone** — they were written to a session scratchpad that was cleared on restart. The numbers above are from notes taken at the time, not re-derivable from disk.

### 4.2 Finetune on the table climb

Warm-started from `terrain_release`, then resumed in-place.

| | |
|---|---|
| iterations | **6000 / 6000 complete** |
| envs | 2048 |
| wall clock | 5 h 51 m (21,078 s) |
| total timesteps | 256,204,800 |
| throughput | ~11,500 steps/s, 4.24–4.50 s/iter |
| VRAM | 12,386 MiB / 24,564 MiB @ 87 % util |
| mean reward | **1.68 → ~5.35** |
| exit | `rc=0`, 2026-08-12 19:36 UTC |

### 4.3 Post-finetune eval — the Stage 4 verdict

```
MOTION_RESULT: table_kid_indoor2-climb-table_rand00001
  status=FAIL  progress=0.644  mpjpe=114.4  reason=anchor_pos
```

**Against the zero-shot baseline this is a large, real gain:**

| metric | zero-shot | finetuned | change |
|---|---|---|---|
| progress | 0.308 | **0.644** | **2.1× better** |
| mpjpe (mm) | 388.2 | **114.4** | **3.4× better** |

`progress` is the fraction of the motion completed before termination — the robot reached ~162 of 250 replay frames. `mpjpe` is mean per-joint position error against the kinematic reference. So the released checkpoints were **not useless** — they were badly out of distribution, and finetuning recovered most of that gap.

**But it is still `FAIL`.** `grail.data_export.export_successful_rollouts` requires `--min_progress 1.0`, so **there is no exportable Stage 4 dataset yet.**

Final termination mix shows where it dies:

| termination | share |
|---|---|
| `time_out` (the *good* outcome — survived the clip) | 0.469 |
| `anchor_pos` | 0.375 |
| `ee_body_pos` | 0.138 |
| `foot_pos_xyz` | 0.122 |

### 4.4 Root cause — a timing budget, not a height budget

`anchor_pos` is the dominant killer, and its rule is specific. From
`imports/SONIC/gear_sonic/envs/manager_env/mdp/terminations.py:104`
(`exceeded_anchor_height`), as configured in `data/stage4_train/table/config.yaml`:

```yaml
threshold: 0.15            # kill if |ref pelvis z − robot pelvis z| > 0.15 m
threshold_adaptive: true
down_threshold: 0.75       # the relaxed budget...
root_height_threshold: 0.5 # ...but only when the reference root is below 0.5 m
```

Measured against `data/motion_lib/grail_table/robot/*.pkl`:

| quantity | value |
|---|---|
| peak reference vertical speed | **0.893 m/s** |
| p95 vertical speed | 0.734 m/s |
| lag budget at peak (0.15 / 0.893) | **0.168 s = 4 source frames** |
| frames where ref root z < 0.5 (relaxation fires) | **0 of 121** |
| steepest 0.5 s window | +0.303 m, from frame 68 |
| failure point (progress 0.644) | frame 77, ref z 1.318 m |

**The robot must track a pelvis rising at ~0.9 m/s to within 15 cm, continuously,
with no relaxation.** The `down_threshold: 0.75` escape hatch exists for
crouching and sitting motions; it never triggers here because our reference
never drops below 0.5 m. The policy dies at frame 77 — immediately after the
steepest ascent window.

So the failure is not "the table is too tall." It is "the reference climbs
faster than the controller can follow, and the tolerance is four frames." Height
is a secondary factor; vertical *speed* is the binding constraint.

> ⚠️ **The 0.15 / 0.15 / 0.2 thresholds are not ours.**
> `imports/SONIC/models/terrain_release/config.yaml` uses identical values. The
> looser 0.25 / 0.25 / 0.5 found in
> `gear_sonic/config/manager_env/terminations/terms/*.yaml` are unused defaults
> that the experiment config overrides. **Relaxing them is therefore moving the
> goalposts** — progress would rise because termination loosened, not because
> tracking improved. Do not treat that as a fix.

### 4.5 The lever that already exists

`imports/SONIC/gear_sonic/utils/motion_lib/motion_lib_base.py:249` and `:1463`:

```python
self.motion_fps_scale = self.m_cfg.get("motion_fps_scale", 1.0)
...
motion_fps = int(curr_motion.fps * self.motion_fps_scale)
curr_dt    = 1.0 / motion_fps
curr_len   = 1.0 / motion_fps * (num_frames - 1)
```

Setting `motion_fps_scale=0.5` reinterprets the same 121 keyframes as 12 fps
instead of 24 — duration 5.04 s → 10.08 s, **every velocity halved**, peak climb
0.893 → 0.45 m/s, lag budget 0.168 s → 0.336 s.

No code change, no re-retargeting, no new motion library — a single Hydra
override. The robot performs the same motion more slowly, which is a legitimate
physical result rather than a loosened standard.

`int()` truncates, so prefer scales that yield whole fps: 0.5 → 12, 0.75 → 18.

### 4.6 Two things that were ruled out

**Why the chair was never finetuned.** It is a compound carry-then-climb task that **no single released policy covers**: terrain tracking treats the object as static so it cannot match a chair that moves 1.96 m; the pick-up policies handle the carry but terminate the moment the chair is stood on. The fix is upstream — split the clip — not more training.

**Why a from-scratch retrain was ruled out.** `terrain_release` was trained on a 17,585-motion library and its base on `combined_bones_18399`. **Neither dataset is on this disk** — our libraries hold one motion each. Independently of data, 3.1e10 env-steps at this card's measured throughput is ≈ 85 days.

### 4.7 Measured scaling on this 4090 (single motion, terrain config)

| envs | s/iter | steps/s | VRAM |
|---|---|---|---|
| 64 | 3.07 | ~500 | 6,069 MiB |
| 512 | 2.86 | ~4,300 | 7,554 MiB |
| 2048 | 4.28 | ~11,500 | 12,386 MiB |
| 4096 (extrapolated) | ~8.5 | — | ~19.4 GB |

VRAM ≈ 5.9 GB base + 3.3 MB/env. The card saturates past ~512 envs.

---

## 5. Caveats and gotchas worth remembering

**`dof_pos` in a Stage 4 `.trajectory.pkl` is in IsaacLab joint order, not URDF order.**
`TrajectoryRecorderTerm` saves `robot.data.joint_pos` raw
(`recorders.py:193`), and IsaacLab orders joints breadth-first —
`left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, …` — whereas the
GMR/MuJoCo model (`g1_mocap_29dof.xml`) and the Stage 3 `robot/*.pkl` use URDF
order, whole left leg then whole right leg then waist then arms. Replaying the
recorded angles in MuJoCo without permuting them puts leg angles on arm joints:
the render looks superficially robot-like, which is what makes it dangerous.
Permute with `G1_ISAACLAB_TO_MUJOCO_DOF` from
`gear_sonic/envs/manager_env/robots/g1.py`:

```python
ISAACLAB_TO_MUJOCO_DOF = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11,
                          15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
dof_mujoco = traj["dof_pos"][:, ISAACLAB_TO_MUJOCO_DOF]
```

`root_pos_w` / `root_quat_w` need no conversion (`quat_format` is already
`wxyz`), so any analysis using only root position — the §4.4 timing work — is
unaffected by this. Always sanity-check a replay against the eval `.mp4` that
Isaac Sim itself rendered.

**Config flags the shipped SONIC release configs omit.** Without these `eval_agent_trl.py` runs *forever* and produces nothing while looking perfectly healthy — GPU pegged, physics stepping. GPU-busy is not evidence of progress; check for the output files.

- `++eval_callbacks=im_eval` — absent from `models/*_release/config.yaml`, so `callbacks={}` and **no metrics are ever computed**.
- `++run_eval_loop=False` — the default rollout loop has no exit condition.
- `++eval_output_dir=<dir>` — the shipped default is an NVIDIA `/mnt/amlfs-05/...` path.
- `++manager_env.config.flat_usd_path=<any .usd>` — the loader raises `FileNotFoundError` on the shipped placeholder path *before* noticing that `flat_to_terrain_ratio=0` means no flat env needs it.

**`train.log` is opened append-mode.** Anchor any measurement to the last `=== detached resume started ===` marker (currently line 32413). Counting matches over the whole file silently reports the *previous* run's numbers.

**Long runs must survive logoff.** `nohup docker exec ... &` is **not** enough — it leaves stdout as a pipe to a host-side client, and when that dies the next print raises `BrokenPipeError`. Use `docker exec -d` with the script doing its own `exec >> logfile 2>&1`. Verify with `readlink /proc/<pid>/fd/1` — it must name a file, not `pipe:[...]`.

**Never poll with `docker exec ... pgrep -f train_agent_trl`** — the pattern matches its own `bash -lc` command line, so the wait loop never exits. Use the bracket trick: `[t]rain_agent_trl`.

**Never wrap container runs in `timeout`** — SIGTERM deadlocks Isaac Sim inside `simulation_app.close()`.

**Never run two Isaac Sim jobs at once** on the single GPU.

**`resume` vs `warm_resume`:** `resume=True` with `checkpoint == <experiment_dir>/last.pt` is a true resume (step counter + optimizer restored). A *different* dirname triggers warm-resume instead.

**Stale `invalid` markers.** `results/generation/4dhoi_recon_smplx/.../invalid` are zero-byte files dated Aug 9, left over from before the fixes. Both clips passed on the final Aug 12 run. **Ignore these files; trust `log.txt`.**

**fps mislabel in Stage 2.** `hoi_data['human_data']['mocap_frame_rate']` reports **30**, but the source Kling videos are **24 fps** (`ffprobe: r_frame_rate=24/1`, 121 frames / 5.04 s). Stage 3 writes 24 and is correct, so nothing downstream is wrong — but reading `mocap_frame_rate` directly gives a value 25 % off.

**Shared machine.** Docker is a shared daemon with other users' images and build cache. Nothing shared was pruned or deleted at any point, per standing instruction. `~/.cache/huggingface` was likewise left alone. Disk is tight (~52 GB free, 98 % full).

---

## 6. Current machine state

The host **rebooted 2026-08-14 14:47**. Container `grail_dev` is `Exited (255)` and the GPU is idle. **Nothing was lost** — training and eval both completed on 2026-08-12.

The container has `RestartPolicy=no`, so a host reboot always stops everything.

```bash
docker start grail_dev          # before any further Stage 4 work
nvidia-smi                      # confirm GPU free
```

Check the results at any time without the container running:

```bash
grep -a MOTION_RESULT data/stage4_eval/table_ft/post_train_eval.log
awk 'NR>32413' data/stage4_train/table/train.log | grep -a "Mean reward" | tail -5
```

---

## 7. What was done in this session

1. Ran **Stage 4 zero-shot** with the released SONIC checkpoints on all three motion libraries. All three failed (§4.1).
2. Diagnosed an eval that appeared healthy but could never terminate — `py-spy` showed it stepping physics inside an exit-conditionless `while True` with `callbacks={}`. Found and applied the three missing launcher flags (§5).
3. Established that policy family is selected by **termination set**, not by object type, and that a `pnp_table` `KeyError` on missing `table_pos` means "not a pick-up motion" rather than "try the sibling policy".
4. Analysed the chair object trajectory and identified it as a **compound carry+climb** task no single released policy can cover (§4.3).
5. Investigated a full from-scratch retrain at the user's request and **ruled it out on evidence** — the training corpora are not on disk, and the compute is ≈ 85 days regardless.
6. Benchmarked env-count scaling on the 4090 and selected **2048 envs** (§4.3 table).
7. Rebuilt the launch mechanism to **survive logoff**: `docker exec -d` + in-container log redirection, replacing the `nohup docker exec &` approach that would have died on a broken pipe.
8. Wrote a **chained post-training eval** that waits for the trainer to exit and then evaluates `last.pt` with baseline-identical settings, recording both video and trajectory pkls — so a result would be waiting whether training finished or died early.
9. Ran the finetune to completion (6000 iters, 5 h 51 m) and collected the result: **progress 0.308 → 0.644, mpjpe 388.2 → 114.4** (§4.3).
10. Audited and re-verified Stage 2 and Stage 3 outputs from disk for this report, including the fps mislabel and the chair's 0.53 rad joint snap.
11. **Diagnosed the actual Stage 4 failure mode** (§4.4): read the `exceeded_anchor_height` termination, measured the reference climb rate against its 0.15 m tolerance, and established that the binding constraint is a four-frame timing budget rather than table height. Found `motion_fps_scale` (§4.5) as the existing one-flag lever, and confirmed the thresholds originate in `terrain_release` itself — so loosening them would be goalpost-moving.

Corrections made along the way, recorded honestly: an initial "eval is healthy" call was wrong and was retracted once `py-spy` showed the real state; a throughput measurement taken over the whole append-mode log reported the previous run's numbers and was retracted and redone against the correct anchor; and **the first draft of §8 in this report recommended lowering the table as the primary fix, which §4.4 supersedes** — the height framing was based on a comparison against the stairs prior rather than on the termination rule that actually fires.

---

## 8. Recommended next steps

> **This section supersedes an earlier draft** which said the fix was "upstream, not
> more compute" and recommended lowering the table first. That was based on a
> height comparison against the stairs prior. The measured cause (§4.4) is the
> reference's vertical *speed* against a four-frame tolerance, and the cheapest
> fix is neither upstream nor more compute — it is one config flag.

Ordered cheapest-first:

1. **Retime the motion — `motion_fps_scale=0.5`.** One Hydra override, no code
   change, no re-retargeting. Halves the peak climb rate to 0.45 m/s and doubles
   the lag budget to 0.336 s, directly attacking the measured failure mode.
   Warm-start from the existing `last.pt` into a **new** experiment directory.
   Cost: ~6 h unattended. **Try this first.**
2. **Lower the table geometry.** A real fix, but it needs a Stage 2 and Stage 3
   re-run to regenerate the reconstruction and the retargeted reference.
3. **Split the clips.** Table into approach + mount; chair into carry
   (→ `pnp_*`) and climb (→ `terrain`). Most work by far, and the only route
   that makes the chair usable at all.

**Not recommended:** loosening the termination thresholds (§4.4 — goalpost-moving,
not a fix), and another 6000 iterations at the current settings (reward had
plateaued at 5.3–5.5 for the last ~1500 iterations, so more of the same buys
little).

Only once `progress = 1.0` can the Stage 4 deliverable be produced:

```bash
python -m grail.data_export.export_successful_rollouts --min_progress 1.0 ...
```

---

## 9. Ready-to-run commands

The two scripts below **are written and committed**; neither has been run yet,
because this workstation's GPU was fully occupied by another user's job
(`breakout`, `train_pdp_tracker.py`, 21.7 GB of 24.5 GB) — see §9.1 for running
them elsewhere.

| script | what it does |
|---|---|
| `data/stage4_train_slow.sh` | warm-start finetune at `motion_fps_scale=0.5`, 6000 iters, 2048 envs, writes to `data/stage4_train/table_slow/` |
| `data/stage4_post_train_eval_slow.sh` | waits for that trainer to exit, then evaluates **at the same 0.5 scale** into `data/stage4_eval/table_slow/` |

```bash
docker start grail_dev

# launch detached, so both survive logoff
docker exec -d grail_dev bash /workspace/grail/data/stage4_train_slow.sh
docker exec -d grail_dev bash /workspace/grail/data/stage4_post_train_eval_slow.sh

# confirm stdout is a FILE, not a pipe (see §5)
docker exec grail_dev bash -lc 'readlink /proc/$(pgrep -f "[t]rain_agent_trl")/fd/1'
```

Because `experiment_dir` differs from the checkpoint's directory, this is a
**warm_resume** — weights only, fresh optimizer and step counter. That is the
correct behaviour here, since retiming changes the task.

Two things that are easy to get wrong:

- **The eval must also pass `motion_fps_scale=0.5`.** Scoring a policy trained on
  the half-speed reference against the full-speed reference measures it on a
  task it never trained for. `stage4_post_train_eval_slow.sh` already does this.
- **`progress = 1.0` here means the 10.08 s climb**, not the source clip's
  5.04 s one. It is a genuine physical result, but it is a slower climb.

### 9.1 Running this on a different machine

`data/` and `results*/` are gitignored, so the following are **force-added** to
the repo: the four `data/*.sh` runners and all of `data/motion_lib/**` (63 MB,
the Stage 3 output Stage 4 consumes). Everything else Stage 4 needs is either
tracked already (`imports/SONIC` is vendored, not a submodule) or downloaded.

```bash
git clone --recursive https://github.com/mg4637/grail.git && cd grail
bash scripts/setup/download_checkpoints.sh    # fetches terrain_release/last.pt
# then the docker commands above
```

**The warm-start checkpoint is not in the repo.** `data/stage4_train/table/last.pt`
is 388 MB, over GitHub's 100 MB per-file limit. `stage4_train_slow.sh` therefore
falls back to `imports/SONIC/models/terrain_release/last.pt` when it is absent.
The consequence is real: the fallback discards the 6 hours of table-specific
training that got progress from 0.308 to 0.644, and restarts from the generic
released policy — on an easier (half-speed) task, but with no memory of this
table. To keep that head start, copy the file across manually:

```bash
scp <thishost>:~/PycharmProjects/GRAIL/data/stage4_train/table/last.pt \
    <otherhost>:~/grail/data/stage4_train/table/last.pt
```

**Baseline to beat: progress 0.644, mpjpe 114.4.**

To re-derive the §4.4 measurements at any time, without the container running:

```bash
.venv/bin/python -c "
import joblib, numpy as np
m = joblib.load('data/motion_lib/grail_table/robot/table_kid_indoor2-climb-table_rand00001.pkl')
m = m[list(m)[0]]
z = np.asarray(m['root_trans_offset'])[:,2]; fps = float(np.asarray(m['fps']))
vz = np.diff(z)*fps
print('peak m/s', vz.max(), '| budget s', 0.15/vz.max(), '| frames below 0.5m', (z<0.5).sum())"
```

Expected: `peak m/s 0.893 | budget s 0.168 | frames below 0.5m 0`.
