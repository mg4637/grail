# GRAIL Replication Roadmap — Task Brief for Claude Code

**Repo:** `NVlabs/GRAIL` (working copy `mg4637/grail`)
**Goal:** Take my Blender + Kling-generated interaction videos all the way to retargeted G1 robot motion data, following the GRAIL pipeline.
**Where I am:** I have finished the *video generation* part of Stage 1 (`gen_2dhoi`). I generated the interaction videos using the **Kling AI web page** (manual, not the API). I have videos from my first Blender scene.

> **Claude Code: read this whole file first, then work top-to-bottom. Do NOT skip the "Stage 0" reconciliation — my videos came from the web UI, so they are probably missing the metadata the rest of the pipeline depends on. Confirm each stage's outputs exist before starting the next stage. Ask me before downloading large checkpoints or launching GPU training.**

---

## Mental model of the pipeline (so you know why each step matters)

GRAIL is 5 stages. The whole method depends on a **"privileged 3D configuration"**: object mesh, camera intrinsics `C_K`, camera extrinsics `C_E`, metric scale, and initial object pose `Θ₁^O` are all *known by construction* before the video is generated, and *reused* during reconstruction. If any of that metadata is missing or inconsistent with the video, Stage 2 silently produces garbage instead of erroring out. Keeping that metadata intact is the single most important thing in this whole plan.

Data flows like this:

```
gen_2dhoi   →  video/*.mp4  + camera params + object mesh + Θ₁^O   (Stage 1)
recon_4dhoi →  recon/*.pkl   (SMPL-X body/hands + object 6-DoF)     (Stage 2)
retargeting →  robot/*.pkl   (G1 joint-space kinematic reference)   (Stage 3)
tracking    →  robot/*.pkl + objects/*.pkl  (post-RL physical traj) (Stage 4)
[sim2real]  →  egocentric visual policy (needs real G1 hardware)    (Stage 5)
```

Reference for the expected on-disk layout (from the released HuggingFace dataset — mirror your `results/` dir to this):

```
data/<hoi_category>/
  ├── video/<category>__<object>__<NNN>.mp4     # source synthetic video
  ├── recon/<category>__<object>__<NNN>.pkl     # 4D HOI recon (SMPL-X + object 6-DoF)
  ├── robot/<category>__<object>__<NNN>.pkl     # post-RL G1 robot trajectory (29 body DOFs + hand_dof_pos)
  ├── objects/<category>__<object>__<NNN>.pkl   # post-RL object 6-DoF trajectory
  ├── meta/<category>__<object>__<NNN>.pkl      # per-motion metadata (lengths, contact flags, source IDs)
  └── object_usd/<category>__<object>__<NNN>.usd
checkpoint/
  ├── GEM-SMPL/            # SMPL-X human pose weights (HMR2, ViTPose, VIMO, YOLO, HMR4D)
  ├── FoundationPose/weights/
  └── SONIC/models/
```

There are **three conda environments**: `grail` (main pipeline), `hunyuan` (3D asset generation only), and `sonic` (Isaac Sim/Lab, needed only for Stage 4 tracking). Use `conda activate grail` for everything except asset gen and RL training.

---

## Stage 0 — Environment + video reconciliation (DO THIS FIRST)

### 0.1 Verify install
```bash
# from repo root, inside the docker container or scratch install
source ~/.bashrc
conda activate grail
python -c "import grail; print('grail import OK')"
```
If this fails, the environment isn't built. Build it:
```bash
git submodule update --init --recursive          # MUST have submodules (GEM-SMPL, FoundationPose, etc.)
bash scripts/setup/install_env_docker.sh          # if using docker image
# OR:  bash scripts/setup/install_env_scratch.sh   # if installing from scratch
```

### 0.2 Download the required checkpoints
```bash
bash scripts/setup/download_checkpoints.sh   # GEM-SMPL / GEM-SOMA / FoundationPose weights
```
**Known gotcha (GitHub issue #6):** the SMPL-X human-pose step (`recon_4dhoi` step 1) needs the **VIMO / HMR4D / ViTPose** sub-checkpoints inside `GEM-SMPL`, not just the base SMPL-X body model. If `download_checkpoints.sh` only pulls the SMPL-X body file, step 1 will crash at `ExtractorVIMO()` in `imports/GEM-SMPL/hmr4d/utils/preproc/vitfeat_extractor.py`. **Action for Claude Code:** after download, list `checkpoint/GEM-SMPL/` and confirm the VIMO, HMR2, HMR4D, YOLO, and ViTPose weight files are all present. If any are missing, check `imports/GEM-SMPL/` for its own download script and run it, and check issue #6 for the resolution before proceeding.

### 0.3 Reconcile my web-UI videos with what the pipeline expects
This is the critical fix. The normal `gen_2dhoi` command:
```bash
python -m grail.pipelines.gen_2dhoi --dataset ComAsset --category cordless_drill \
    --character kid --results_dir results --video_model_api kling-ai
```
does FOUR things: (1) assembles the Blender scene, (2) renders the init frame + **exports camera intrinsics/extrinsics and the settled object pose `Θ₁^O`**, (3) calls the Kling **API** to make the video, (4) writes everything into `results/` in the exact filename/metadata layout `recon_4dhoi` reads.

Because I used the **Kling web page**, I only have raw `.mp4` files. I am probably missing steps 2 and 4's metadata. **Claude Code, do this:**

1. Inspect `grail/pipelines/gen_2dhoi.py` and trace exactly what it writes to disk per sequence (filenames, camera param file, object pose file, any JSON/pkl sidecars, expected fps and resolution). Produce a checklist of every artifact `recon_4dhoi` expects to find.
2. Inspect `grail/pipelines/recon_4dhoi.py` (especially its input-loading / dataset-indexing code) to confirm exactly which files it reads and their expected paths.
3. Determine whether I can run `gen_2dhoi` in a **"render + export metadata only, skip video API call"** mode so it produces the Blender camera params + object pose for my scene *without* re-generating video. Look for a flag or an easy code path to short-circuit the API call and instead drop in my existing `.mp4`. If no such flag exists, tell me the smallest code change to add one.
4. Confirm my videos match Kling's expected format: the paper/pipeline uses **static camera, 24 fps, 1920×1080, 5 or 10 s** (121 frames for 5 s). If my web-UI videos differ (fps, resolution, or — fatal — a *moving* camera), flag it. A moving camera breaks the whole "known `C_E`" assumption and the reconstruction will fail. List which of my videos are safe.
5. Place my videos and their reconstructed metadata into the expected `results/<dataset>/<category>/...` structure with the correct naming convention so downstream stages find them automatically.

**Do not proceed to Stage 1.5/2 until every file in the `recon_4dhoi` input checklist exists for at least one test sequence.**

---

## Stage 1.5 — Optional: 3D asset + terrain generation

Only needed if I want objects/terrain beyond what I already have. Skip if my current object + scene are enough for a first end-to-end run.

- **Procedural terrain** (curbs/slopes/stairs), env `grail`:
  ```bash
  python -m grail.pipelines.gen_terrain --type stairs --num 50 --output_dir data/syn_stairs
  ```
- **AI-generated 3D objects** (Hunyuan3D), env `hunyuan`:
  ```bash
  conda run -n hunyuan python -m grail.pipelines.gen_3d_assets \
      -i configs/gen_3d/example_objects.yaml -o data/gen_example
  ```
- Or just use a **ComAsset** object (simplest path):
  ```bash
  bash scripts/setup/download_comasset.sh --category cordless_drill
  ```

**For my first run, recommend I stick to one simple ComAsset object** (e.g. `cordless_drill`) so I'm on the well-trodden path and can compare against the repo's expected behavior.

---

## Stage 2 — 4D HOI Reconstruction (`recon_4dhoi`) — THIS IS MY NEXT REAL STEP

Command (env `grail`):
```bash
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill --results_dir results
```
Swap `--dataset`/`--category` to match how I registered my own scene in Stage 0.3.

### What happens internally (verify each sub-step produces output):
1. **Human motion estimation** — GENMO gives per-frame SMPL-X body pose (camera space), transformed to world space via my known `C_E`. Body *shape* is held fixed at the pre-fitted character morphology (not re-estimated). WiLoR refines per-frame MANO hand pose; gaps filled by linear interpolation + Savitzky–Golay smoothing; merged onto the body via wrist IK. *(This is the step that crashes if VIMO weights are missing — see 0.2.)*
2. **Object pose tracking** — FoundationPose (RGB-only fine-tuned variant, depth channels zeroed) initialized from `Θ₁^O`, propagated across all frames. Requires known object geometry/texture/camera — all of which must be present from Stage 0.3.
3. **Failure filtering** — compares FoundationPose renders against SAM2 masks; discards sequences with mask-tracking error `e_M > τ = 0.2`. **Expect some of my Kling web-UI clips to be dropped here** if they have texture flicker or inconsistent object appearance across frames. This is normal — the paper discards a non-trivial fraction.
4. **Joint optimization** — refines human + object trajectories together with the 5-term loss (keypoint, projection, MoGe-2 metric-depth alignment, VLM-labeled contact, regularization/smoothness). This is the slow step (~8 of ~14 min/sequence on an A100).

### Runtime expectation (per 5 s / 121-frame sequence, single A100):
video-gen ~1 min · human est ~2 min · object track ~1 min · optim preprocess ~2 min · **joint optim ~8 min** · **total ~14 min**. Budget accordingly; run one sequence end-to-end before batching.

### Claude Code actions:
- Run on **one** sequence first. Confirm a `recon/<...>.pkl` is written.
- Load the output pkl and sanity-check it: SMPL-X params present for all frames, object 6-DoF present, no NaNs, trajectory lengths match video frame count.
- If step 1 crashes → go back to 0.2 (VIMO/HMR4D weights).
- If step 2/3 drops the sequence → inspect the SAM2 vs render mask overlap; likely a video-quality issue, note which clips survive.
- Only after one clean sequence, batch the rest.
- **Verification I can do:** the repo has visualization tooling (`docs/source/visualization.md`, `web_visualizer.md`). Use it to render the reconstructed 4D HOI overlaid on the source video and confirm the human + object actually track the video.

---

## Stage 3 — Retargeting (`retargeting`)

Retargets the reconstructed SMPL-X motion to the **Unitree G1** skeleton using GMR (General Motion Retargeting), producing the robot joint-space kinematic reference `q̃_t`. Because my character was pre-fitted to G1 proportions in Stage 1, morphology mismatch is reduced and hand-object / body-scene contacts are better preserved.

### Claude Code actions:
- Read `docs/source/retargeting.md` for the exact entrypoint and flags (likely `python -m grail.pipelines.retarget ...` or a `scripts/` wrapper — confirm the real name from the repo, don't assume).
- Run on the one sequence that passed Stage 2.
- Confirm a `robot/<...>.pkl` (kinematic reference, 29 body DOFs + hand DOFs) is produced.
- Sanity-check for retargeting artifacts the GMR paper warns about: foot sliding, self-penetration, infeasible joints. Visualize the retargeted G1 motion before moving on.

---

## Stage 4 — Task-General Tracking (`tracking`) — HEAVY COMPUTE, env `sonic`

Converts the kinematic reference into **physically executable robot actions** via RL in Isaac Lab on top of the frozen **SONIC** whole-body controller. Two specializations:
- **Object-aware latent adaptor** (`π_φ`) — for pick-up / manipulation. Trains a small MLP adaptor; SONIC encoder/quantizer/decoder stay **frozen**.
- **Scene-aware tracker** — for terrain/sitting. Fine-tunes SONIC **end-to-end** + a height-map CNN encoder.

### ⚠️ Compute reality check — tell me before running:
The paper trains on **64× NVIDIA L40 GPUs, 30,000 PPO iterations, 1024 envs/GPU (~30 h/run)**. I almost certainly cannot replicate that. **Claude Code, recommend the cheaper path:**
1. **Preferred:** use the **released task-general tracking policy checkpoints** (the repo README marks "Release task-general tracking policy checkpoints" as ✅ done, under `SONIC/models/`). Load those instead of training from scratch, and just *run* tracking/inference on my retargeted motion.
2. **If I insist on training:** scale down drastically — 1 GPU, far fewer envs, far fewer iterations — and warn me it won't match paper quality. The repo notes warm-starting from an existing policy converges in ~6,000 iterations (~1/5 of a full run) when adding in-family motions; use that, never train from scratch on a single GPU.

### Claude Code actions:
- Read `docs/source/tracking.md` for the real entrypoints/flags for both trackers.
- Confirm `conda activate sonic` and Isaac Sim/Lab installed (`install_env_scratch.sh --install-sonic`, ~30 GB).
- Confirm `checkpoint/SONIC/models/` exists.
- For my pick-up object: run the **object-aware** path. For terrain/sitting: run the **scene-aware** path.
- Output: post-RL `robot/*.pkl` + `objects/*.pkl` — the actual physics-validated trajectory the simulated G1 realizes. This is the "GRAIL-generated data."
- Visualize the simulated G1 executing the motion in Isaac Lab before declaring success.

---

## Stage 5 — Sim-to-Real (only if I have a real G1)

Distills the tracking policy into an **egocentric RGB visual policy** (head-camera in → SONIC latent tokens out) with visual domain randomization, deployed on a real Unitree G1 (paper: streamed to a desktop RTX 5090, OAK-D W camera, 10 Hz). **Skip unless I explicitly say I have hardware.** If I do, read the deployment docs and treat this as a separate project.

---

## Suggested order of operations for Claude Code (checklist)

1. [ ] Stage 0.1 — verify `grail` env imports; build if not.
2. [ ] Stage 0.2 — download checkpoints; **explicitly verify GEM-SMPL sub-weights (VIMO/HMR4D/ViTPose)** exist.
3. [ ] Stage 0.3 — trace `gen_2dhoi.py` + `recon_4dhoi.py` I/O; build the input checklist; reconcile my web-UI videos into the expected structure with camera params + `Θ₁^O`; flag any moving-camera / wrong-fps videos.
4. [ ] Stage 2 — run `recon_4dhoi` on ONE sequence; verify `recon/*.pkl`; visualize overlay.
5. [ ] Stage 2 — batch remaining sequences; note which survive the SAM2 failure filter.
6. [ ] Stage 3 — run retargeting; verify `robot/*.pkl`; visualize G1 kinematics.
7. [ ] Stage 4 — **prefer released SONIC checkpoints**; run tracking; verify post-RL trajectory in Isaac Lab. Ask me before any from-scratch training.
8. [ ] (Stage 5 — only if I have a real G1.)

## Things to surface to me proactively
- Any missing checkpoint or submodule.
- Any video that fails the static-camera / fps / resolution assumptions.
- The true entrypoint names + flags for `retargeting` and `tracking` (read the docs, don't guess).
- Whether I should just download the released GRAIL dataset/checkpoints for the stages I can't feasibly compute, and only run the stages I can.
- Realistic wall-clock + GPU estimates before any long run.

## Reference links
- Repo: https://github.com/NVlabs/GRAIL
- Docs: https://nvlabs.github.io/GRAIL/
- Per-stage markdown: `docs/source/{installation,quick_start,gen_3d_assets,gen_2dhoi,recon_4dhoi,retargeting,tracking,data_export,visualization,web_visualizer}.md`
- Released dataset (reference layout + checkpoints): https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL
- Known human-pose weight issue: https://github.com/NVlabs/GRAIL/issues/6
