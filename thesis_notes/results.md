# Results

Outputs and measurements, in the order they are produced. Nothing here is guessed: if a stage has not run, it says "not run yet".

## Status board

| Stage | Where it runs | Status |
|---|---|---|
| 0. Update GRAIL (`git pull`) | Mac | done: `aa31d82 → 8b9afa5`, README-only change |
| 1. Generate 3-step stairs | Mac (CPU) | done: custom wide stairs_wide3 (see Stage 1-2) |
| 2. 2D HOI: render start frame | Mac | done |
| 3. Video (Kling replacement) | you, web UI | done: 10 s, 24 fps |
| 4. 4D HOI reconstruction | Karolina (GPU) | done: 15.4 min, valid (see Stage 4 result) |
| 5. Retarget to Unitree G1 (GMR) | Karolina login node (CPU) | done: trial 1 and trial 2 (robot gap of trial 2 NOT yet measured) |
| 6. Tracking policy training | Karolina (GPU) | smoke test passed; 6000-iteration run submitted (job 5182169), result pending |
| 7. Post-training in orcs (mjlab) | Karolina (GPU) | not started |
| 8. Push repo to `mg4637/grail` | Mac | done as NEW branch `grail-stairs-snapshot` (ce62787); main untouched |

## Stage 0 result

```
before: aa31d82 feat/fix: honour faceVertexCounts when converting USD meshes ...
after : 8b9afa5 Remove oral presentation note from citation
changed files: README.md only
```

## Stage 1 result: the 3-step staircase

`data/syn_stairs/stairs_007/model.obj` measured with trimesh (script `stairs_info.py`):

| asset | size x,y,z (m) | faces | distinct step-top heights (m) | steps |
|---|---|---|---|---|
| stairs_007 | 0.809 x 1.751 x 0.335 | 36 | 0.127, 0.266, 0.335 | **3** |
| stairs_000 (control) | 2.591 x 1.414 x 0.891 | 96 | 0.114, 0.256, 0.331, 0.427, 0.561, 0.648, 0.744, 0.891 | 8 |

The seed prediction (`random.seed(7070+i); randint(3,8)`) matched the real mesh in both cases.
Rises of stairs_007: 0.127, 0.139, 0.069 m (differences between consecutive step tops; the last step is low). Whether the G1 can climb these is something the training stage will tell us, not something assumed here.

## Lesson 1 result: MuJoCo ball on stairs

```
nbody = 2 | ngeom = 5 | nq (positions) = 7 | nv (velocities) = 6
t= 0.00s  ball xyz = [0.5 0.  1. ]
t= 0.50s  ball xyz = [0.5   0.    0.384]
t= 1.00s  ball xyz = [0.5   0.    0.385]
final ball z = 0.385
contacts at end: 1  (ball_geom <-> step2)
```
Expected value: top of step 2 (0.335) + ball radius (0.05) = 0.385. It matches, so the physics, geometry and contact logic work on your Mac.

## Stage 2 result: start frame rendered on the Mac

| item | value |
|---|---|
| asset | `stairs_wide3` (custom: 3 steps, 12 cm rise, 35 cm tread, 1.5 m wide) |
| Blender | 4.4.3 on M2 (GRAIL targets 4.3.2) |
| render time | 37 s (first stairs_007 render: 2 min, includes first-run overhead) |
| frame | `results/generation/asset_renders/SynStairsV1/stairs_wide3/kid_indoor1-syn-stairs_rand00001.png` (1280x720) |
| other outputs | camera pickle, depth PNG + EXR, object/human masks, FoundationPose inputs (`cam_K.txt`, `first_frame_pose.pickle`) under `results/generation/` |

Visual check (by me, from the image): kid on the left facing the stairs, stairs clearly 3-tiered with treads deep enough for a foot, static framing, kid fully visible. The earlier stairs_007 frame was rejected by you because the treads were too narrow.

Not yet verified: whether Blender 4.4.3 output matches 4.3.2 pixel for pixel (we have no 4.3.2 render to compare).

## Stage 3 result: the generated video (made by you in a web UI)

| property | value |
|---|---|
| file | `results/generation/videos_kling/SynStairsV1/stairs_wide3/kid_indoor1-syn-stairs_rand00001.mp4` |
| video | H.264, 1280x720, 24 fps, 240 frames, 10.0 s |
| audio | AAC track present (ignored by GRAIL) |
| size | 1.3 MB |

Visual check of 9 frames (`docs_learning/frames/contact_sheet.png`), by me:
- camera static, same room as the start frame: yes
- kid walks in from the left, approaches the stairs, climbs all 3 steps one at a time with feet fully on each tread, ends standing on the top platform, facing the camera/right: yes
- stays in frame throughout: yes
- caveats: the kid's body turns toward the camera at the end; the video model slightly changed the kid's appearance (hair/face) vs. the render, which is normal and expected.
Not measured: foot-sliding or exact step timing; reconstruction quality will show that.


## Stage 8 result: GitHub upload

- Target repo `mg4637/grail` was non-empty with unrelated history, so nothing was pushed to `main`.
- Pushed branch `grail-stairs-snapshot` = one commit with upstream GRAIL 8b9afa5 + 2 patches (verified by `git diff --stat`).
- Two earlier push attempts of the full history were rejected by Git LFS (an old LFS object is missing upstream). About 2 GB of LFS objects had been uploaded to the user's account before the rejection; the user should check their GitHub LFS quota.
- Documents were not pushed (public repo, they contain cluster identifiers).

## Karolina facts (verified from the user's terminal output)

| item | value |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB (40 GB), driver 610.43.02, node acn12 |
| compute-node internet | works (huggingface.co HTTP 200; Docker Hub registry reachable, 401 = auth challenge) |
| project | <PROJECT-ID>, Karolina GPU, 800 node-hours, 24.51 used before the test, valid until 2026-10-27 |
| project storage | `/mnt/proj2/<project-id>` (group-writable, shared filesystem 19 TB free; personal quota unknown) |
| Apptainer | modules apptainer/1.1.5, apptainer/1.2.5 (default) |
| usage after a 10-min allocation | `it4ifree` still showed 24.51 (billing granularity not yet determined) |

## Stage 4 prerequisites on Karolina (all verified from the user's terminal output)

| item | result |
|---|---|
| container | `grail_sandbox` (Apptainer, 55 GB, built from `docker://nvgrail/grail:1.1.1`), env `grail`: Python 3.10.20, torch 2.7.1+cu128 |
| GPU test job | A100-SXM4-40GB (cc 8.0, 42.4 GB), matmul OK, 5 s job |
| nvdiffrast | initially FAIL (CUDA 209); rebuilt for sm_80 into `$WORK/pylibs` -> OK |
| FoundationPose mycpp | initially missing; built -> OK |
| `import grail.pipelines.recon_4dhoi` | OK |
| checkpoints | GEM-SMPL (hmr2 2.6 GB, vitpose, vimo, yolo, GENMO last.ckpt), FoundationPose weights, SMPL + SMPL-X body models: all `[OK]`; SONIC, GEM-SOMA, Hunyuan not downloaded (not needed yet) |
| runtime models | MoGe-2 and SAM2 pre-downloaded; `hf_cache` = 17 GB |
| project allocation | 389.64 / 800 node-hours used on 2026-10-11 (24.51 on 2026-10-05; not our usage); valid to 2026-10-27 |


## Stage 4 result: 4D HOI reconstruction (Karolina job 5181637)

| item | value |
|---|---|
| hardware | 1x A100-SXM4-40GB (node acn06) |
| wall time | 927 s (15.4 min); stage 4 optimization 10 min 52 s |
| status | `exit code: 0`, `Videos: 1/1 valid | Frames: 240/240 valid` |
| initial human position error | 0.0931 m (threshold 0.5) |
| object mask difference | 0.0122 (threshold 0.3), aligned at all frames |
| keypoint tracking | 0.480 (threshold 2.0) |
| contact | 0.0 (threshold 5.0; static object, no contact labels) |
| min distance at initial frame | 1.2327 m (threshold 0.02) |
| human motion (avg max velocity) | 0.0360 (threshold 0.01) |
| object motion | static, range 0.0000 m |
| interaction start frame | 210 / 240 (from my manual-LLM answers; see llm_answers.md Entry 2) |

All numbers are copied from the pasted job log. Not yet checked by eye: the rendered reconstruction videos and the 3D trajectory (outputs still on the cluster).

### Stage 4 visual check (by me, 6 of 240 frames; files in `docs_learning/frames/recon_comparison_sheet.png`, `recon_top_sheet.png`)

Outputs copied to the Mac under `results/generation/4dhoi_recon_smplx_valid/SynStairsV1/stairs_wide3/kid_indoor1-syn-stairs_rand00001/`: `hoi_data/hoi_data.pkl` (410 KB), `mesh_data/` (stairs mesh), `result_vis/{input,recon_result,recon_comparison,recon_result_top_view}.mp4` (1280x720, 24 fps, 240 frames each; comparison is 1280x2160) and `recon_result.html` (36 MB interactive viewer).

- Frames 0, 60: reconstructed body stands/walks toward the stairs like the input.
- Frames 110, 150: one foot on the lower steps; overlay follows the input pose.
- Frames 190, 230: standing on the top platform at the same place as the real kid.
- Top view: approach, step onto the stairs, end near the middle of the stair rectangle (where the top platform is).
- The `inter start idx = 210` concern: no visible effect on world alignment in these frames.
- NOT checked: foot sliding, stair penetration, smoothness, the other 234 frames. The interactive HTML viewer was not opened by me.

## Stage 5 result: retargeting to Unitree G1 (login node, CPU, container `sonic` env)

| item | value |
|---|---|
| command | GRAIL terrain recipe: `--zero_out_wrist`, default height_offset/mesh_scale |
| time | ~17 s for the motion |
| outputs | `data/motion_lib/stairs_wide3/{robot,objects,meta}/stairs_wide3_kid_indoor1-syn-stairs_rand00001.pkl`, `object_usd/....usd` + textures |
| stairs USD check | 1 mesh, 18 faces; RigidBody + Mass on `/model`; Collision + MeshCollision on the mesh |
| known error | `RuntimeError: Accessed invalid null prim` in Isaac Lab MeshConverter; handled by GRAIL's repair step; Isaac Sim also had no GPU on this node |
| not yet checked | the robot motion itself (feet on steps? height offset?), SDF collision settings |

### Stage 5 check of the robot motion (by me; scripts in `docs_learning/lessons/03_view_retargeted_motion.py`)

| item | value |
|---|---|
| frames / fps / duration | 240 / 24 / 10 s |
| pelvis start -> end (x,y,z) | (-0.225, 1.213, 0.759) -> (-0.355, -0.991, 1.185) |
| stairs pose (world) | position (0,0,0.006), rotated -90 deg about z; robot ends on the top platform (y in [-1.4,-0.7]) |
| feet at start | on the floor within ~1-3 cm (both methods) |
| feet at end | **~3-6 cm above the top surface** (uncertainty ~2 cm) |
| replay (6 frames viewed) | stands, walks, steps onto first step, climbs, stands on top |
| not checked | foot sliding, individual step contacts (crude detector), the other frames, physical feasibility |


### Stage 5 correction: the retargeted G1 hovers above the stairs (measured; confirms the user's observation)

Script `docs_learning/lessons/04_contact_gap_check.py`. gap = lowest foot point - surface height under it (positive = hovering). Standing on the floor the gap reads about -1.5 cm (foot geometry reaches below z=0), so real hover = gap + ~1.5 cm.

| frame | pelvis (y, z) | left foot gap | right foot gap | note |
|---|---|---|---|---|
| 0 | +1.21, 0.76 | -1.5 cm | -1.4 cm | standing on floor |
| 40 | +0.92, 0.76 | -1.6 | +0.5 | walking |
| 80 | +0.30, 0.79 | +0.3 | +18.8 | right foot swinging |
| 100 | +0.01, 0.94 | +17.8 | +5.5 (over step 1) | lifting onto the first step |
| 120 | -0.13, 0.98 | +7.2 (over step 2) | +6.0 (over step 1) | climbing |
| 140 | -0.37, 1.10 | +7.4 (step 2) | +17.6 (step 1) | climbing |
| 160 | -0.58, 1.12 | +10.8 | +3.0 (top) | climbing |
| 180 | -0.84, 1.21 | +10.7 | +5.0 | on top |
| 200 | -1.04, 1.19 | +3.6 | +5.9 | on top |
| 239 | -0.99, 1.19 | +3.4 | +3.3 | standing on top |

Summary over 240 frames (gap of the lower foot): median +3.5 cm, mean +2.6, min -4.1, max +8.8. Frames where neither foot is within 2 cm of a surface: 150; within 5 cm: 74; within 10 cm: 0. Frames with a foot more than 2 cm inside a surface: 36.

Interpretation (measured): the robot stays a few cm above the steps during the climb and at the top; the pelvis rises 43 cm while the stairs are 36.6 cm. Not interpreted yet: whether the cause is the reconstruction or the retargeting (see implementation_history.md Step 8.3). The first check in this stage ("feet on the floor at the start, 3-6 cm up at the end") was incomplete because it ignored the surface under each foot.

### Stage 4/5 diagnosis: where does the hover come from? (measured with lessons/05_human_vs_stairs.py)

Human (G1-proportioned SMPL-X from `hoi_data.pkl`) foot-to-surface gap, same stairs/floor model as for the robot:

| frame | pelvis z | left gap | right gap | note |
|---|---|---|---|---|
| 0 | 0.753 | -0.2 cm | -0.6 cm | standing on floor (method check) |
| 60 | 0.776 | +3.9 | +0.1 | approaching |
| 100 | 0.943 | +16.9 | +7.0 (over step 1) | lifting |
| 120 | 0.977 | +8.5 (over step 2) | +8.5 (over step 1) | climbing |
| 160 | 1.122 | +11.0 | +4.4 (top) | climbing |
| 180 | 1.207 | +11.4 | +7.7 | on top |
| 239 | 1.179 | +5.5 | +5.1 | standing on top |

Lower-foot gap: median +5.4 cm (robot: +3.5), mean +4.3, min -2.0, max +9.5. Neither foot within 5 cm of a surface in 144/240 frames (robot: 74/240). Pelvis rise 0.426 m vs stairs 0.360 m.

**Conclusion:** the hover is present in the reconstruction (human), not introduced by retargeting. The reconstruction filter reported "Valid" because none of its checks measures foot-to-terrain distance. Cause (inferred from the code, not proven): no term in the optimization ties the feet to the step surfaces; see implementation_history.md Step 8.6.

## Stage 4 re-run with the new `terrain_support` loss (GPU job, A100, 10.6 min, exit 0, valid)

Human foot-to-surface gap (`lessons/05_human_vs_stairs.py` on the new `hoi_data.pkl`; same method as the original measurement):

| measure (240 frames) | original reconstruction | with terrain loss |
|---|---|---|
| lower-foot gap, median / mean | +5.4 / +4.3 cm | **+1.7 / +1.2 cm** |
| lower-foot gap, min / max | -2.0 / +9.5 cm | **-0.7 / +2.8 cm** |
| neither foot within 2 cm of a surface | 166 / 240 | 50 / 240 |
| neither foot within 5 cm of a surface | 144 / 240 | **0 / 240** |
| a foot > 2 cm inside a surface | 5 | **0** |
| pelvis rise (stairs 0.36 m) | 0.426 m | 0.390 m |
| filter: keypoint tracking (limit 2.0) | 0.480 | 0.660 |
| filter: initial position error (limit 0.5 m) | 0.0931 m | 0.0967 m |

Sample frames (L,R foot gap): frame 120 +3.0/+2.6 cm (before +8.5/+8.5); frame 160 +7.8/+0.8 (before +11.0/+4.4); frame 180 +6.0/+1.8 (before +11.4/+7.7); frame 239 +1.9/+2.1 (before +5.5/+5.1). Frame 0 still grounded (+0.2/-0.1).

Caveats: median hover is still +1.7 cm; keypoint error rose (0.48 -> 0.66) because the video's kid was not at exactly the stairs' rendered heights; the optimiser also moved the stairs by 5 mm (t_z 0.006 -> 0.001). Not yet done: retarget the new reconstruction and measure/replay the robot.

## Trials archive (nothing is overwritten; each trial has its own folders)

| Trial | What differs | Reconstruction output | Retargeted motion | Status |
|---|---|---|---|---|
| 1 | original GRAIL optimisation (`loco_smplx.yaml`) | `results/generation/4dhoi_recon_smplx_valid/` | `data/motion_lib/stairs_wide3/` | Mac + cluster; hover median +5.4 cm (human), +3.5 cm (robot) |
| 2 | + `terrain_support` loss weight 1000 (`loco_smplx_terrain.yaml`) | `results/generation/4dhoi_recon_smplx_terrain_valid/` | `data/motion_lib/stairs_wide3_terrain/` (retarget command given; robot gap not yet measured) | cluster (Mac copy pending); hover median +1.7 cm (human) |

SCRATCH note: files not accessed for 90 days are deleted automatically; keep the Mac copies (and optionally copy to `/mnt/proj2/<project-id>/...`) as the lasting archive.

## Stage 6 prerequisite: SONIC training stack runs on Karolina (smoke test, job 5182164)

| item | value |
|---|---|
| command | `gear_sonic/train_agent_trl.py +exp=manager/universal_token/scene/terrain_tracking`, 4 envs, 3 iterations, warm start `models/terrain_stairs_warmstart/last.pt` |
| data | `data/motion_lib/stairs_wide3_terrain` (trial 2, terrain-loss reconstruction, retargeted) |
| result | completed; 288 timesteps; 37 s job; checkpoints in `training/smoke` |
| not measured yet | speed at scale (iteration time with 1024+ envs), reward/tracking quality after 6000 iterations |

## Experiment log: early termination in policy rollouts (one change per row)

| id | checkpoint | change vs previous | result (termination / full clip?) | notes |
|---|---|---|---|---|
| E0 | pretrained warm start | baseline (4 envs, 500 steps) | all envs reset at 5.2-5.6 s (frames 130-140), pelvis max 0.87 m, upright | reason unknown; before any finetuning |
| E1 | run 2, step 1000 | checkpoint only (+ passive logger, no behaviour change) | **no early termination**: `terminations.jsonl` = one line `{"step": 497, "fired": {"time_out": [0,1,2,3]}}` (all 4 envs reach the end of the 498-frame / 9.94 s clip; no failure rule fired); 250 frames recorded per env | job 5188083; motion lib: 1 motion, 9.940 s, 498 frames at 50 Hz |


### E1 detailed report (lessons/08_rollout_report.py on `rollouts/e1_run2_1000`, 4 envs x 250 frames @ 25 fps)

User's visual verdict in MuJoCo (full clip, `03_view_retargeted_motion.py --trajectory`): "the new clip is very good".

| quantity | E0 (pretrained warm start, no finetuning) | E1 (run 2, 1000 iterations) |
|---|---|---|
| early resets | every env at frame 130-140 (5.2-5.6 s); reason never logged | none; only `time_out` at step 497 (clip end 9.94 s / 498 frames) |
| root distance to reference | mean 0.83-0.85 m, max 1.8-2.3 m | **mean 0.065-0.069 m**; max only at the final frame (the natural reset back to the start, 2.25 m) |
| pelvis height | 0.75-0.87 m (never climbs) | 0.76-1.15 m (reference 0.75-1.15 m) |
| lower foot vs stairs surface (physics) | median -0.2 cm, min -11 to -12 cm, 7-18 frames > 2 cm inside | **median -0.1 cm, min -0.5 cm, max 0.0**, 0 frames > 2 cm inside |
| robot at the end | back at the start after the 2nd reset (y ~ +0.1) | on the top platform at y ~ -1.0 (reference -1.00) until the final reset frame |

Frame table env 0 (E1): t = 0 s y +1.21 z 0.76; 4 s y +0.02 z 0.83; 6 s y -0.40 z 1.02; 8 s y -1.05 z 1.14; 9 s y -1.02 z 1.14; dist to reference <= 0.09 m throughout. Swing-foot gaps of +11 to +20 cm at some frames are normal (the other foot is on the ground).

Interpretation: the policy reproduces the clip within ~7 cm on average and keeps the feet on the steps. Because a simulated foot cannot hover, the earlier +1.7 cm reference hover (and the arguable benefit of the terrain loss) was absorbed by physics; still no trial-1 vs trial-2 training comparison exists, so the contribution of the terrain loss to this outcome is unproven. The E0 result and E1 result differ by checkpoint only (run 2 starts from the E0 policy).

## Contacts and forces in the MuJoCo replay (E1 rollout, env 0) — what is real and what is not

Foot collision spheres vs stairs (signed distance, negative = overlap), via `mj_geomDistance`: frame 125: -4.7 mm (step 0); 175: -4.7 mm (step 2); 200: -4.6..-4.8 mm (step 2); 225: -4.1..-6.3 mm (step 2); frame 0: floor -1.3 mm and -0.3 mm. Contacts counted by `mj_forward` after making the stairs collidable at build time: frame 0: 2 (floor); 100: 2; 125: 2; 150: 2; 175: 4; 200: 6; 225: 5 (all stairs). Summed normal force reported by MuJoCo: 24 / 5 / 5 / 5 / 105 / 137 / 153 N vs weight 337 N: **an artefact of a static pose with zero velocity and no actuators, not the real ground reaction**. The real forces come from the PhysX sensors (`forces.npz`, experiment E1b, pending).

### E1b: real PhysX contact forces (`rollouts/e1b_forces`, job with logger v2, same checkpoint/settings as E1; `lessons/09_forces_report.py`)

Passive additions only: trajectories identical in size to E1 (43,407 B per env), only `time_out` at step 497, no `logger_error`.

| quantity (4 envs, all alike) | result |
|---|---|
| robot mass / weight in PhysX | 33.3 kg / 327 N (the MuJoCo model has 34.4 kg / 337 N: robot descriptions differ by ~1 kg) |
| summed vertical foot force, standing on floor (0.2-0.8 s) | mean 320-330 N = **97.8-100.9 %** of weight |
| standing on the top step (8.6-9.6 s) | mean 326-330 N = **99.6-100.8 %** of weight (min 307, max 397) |
| approach (2-3 s) | 335-339 N = 102.5-103.5 % |
| climbing (5-8 s) | mean 324-326 N = 99.0-99.6 %; range 248-459 N |
| frames with NO foot force > 10 N | **0 / 250** |
| frames with a foot pressing on the stairs (>10 N, feet->terrain sensor) | 157 / 250; peak vertical reaction 450-459 N at frame 162 (6.5 s) = ~1.4 x weight |

Conclusion: the physics sanity check passes (static standing force = weight), so `contact_forces` / `feet_terrain_contact_sensor` readings are credible and the viewer's `--forces` arrows show real PhysX forces. Not checked: contact positions (not recorded), horizontal/friction forces, individual foot loads vs the reference.

## Status of the GRAIL stair-climbing pipeline (written 2026-10-12)

Finished end to end on ONE clip: custom wide 3-step staircase -> 2D HOI render -> video (user, web UI) -> 4D reconstruction (+ our terrain loss, manual LLM answers) -> G1 retargeting -> RL tracking policy (finetuned from the pretrained stairs policy, 1000 iterations) -> rollout recorded in Isaac Sim and replayed in MuJoCo. Outcome: the policy follows the clip (mean root distance 6.5-6.9 cm), stays on the steps, never airborne, standing force = weight. NOT done / not claimed: generalisation (trained and tested on the same clip), a trial-1 vs trial-2 training comparison, GRAIL's data export / web visualiser stages, sim-to-real, a multi-clip dataset.
