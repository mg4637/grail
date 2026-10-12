# GRAIL learning guide

You are learning by doing. Each section says **what** we do, **why**, and **what could go wrong**.
Sections are appended as we reach each stage.

---

## 1. The big picture: what GRAIL does

GRAIL ("Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors") makes **robot training data without motion capture**.

```
 3D asset (stairs)        ┌─ physics sim: where does the object rest?
        │                 │
        ▼                 ▼
 [1] gen_terrain ──► [2] gen_2dhoi ──► rendered start frame (kid standing before stairs)
                                             │
                                  [video model: Kling / web UI]   ← we do this manually
                                             ▼
                                   video of kid climbing stairs
                                             │
                          [3] recon_4dhoi: video ──► 3D body motion (SMPL-X) + object pose
                                             │
                          [4] retarget (GMR): human skeleton ──► Unitree G1 robot joints
                                             │
                          [5] tracking training (SONIC, RL): a policy that makes the
                                 simulated G1 follow that motion on those stairs
                                             │
                          [6] post-training in orcs (mjlab): adapt/fine-tune the policy
```

Key idea: a video generation model already "knows" how people climb stairs. GRAIL lifts that 2D knowledge into 3D motion, converts it to robot motion, and uses RL so physics makes the motion feasible.

Vocabulary you will see everywhere:

| Term | Meaning |
|---|---|
| HOI | Human-Object Interaction |
| 2D HOI / 4D HOI | video of the interaction / 3D + time reconstruction of it |
| SMPL-X | a parametric 3D human body model (pose + shape numbers) |
| Retargeting | map a human motion onto a robot with different proportions |
| G1 | Unitree G1 humanoid robot |
| Tracking policy | RL policy rewarded for imitating a reference motion |
| SONIC | NVIDIA's pretrained general motion-tracking foundation policy |

---

## 2. Which machine runs which stage?

| Stage | Needs | Mac M2? | Plan |
|---|---|---|---|
| gen_terrain (stairs mesh) | numpy/trimesh-style CPU code | yes | Mac |
| gen_2dhoi steps 1 & 3 (Blender sim + render) | Blender (+ GPU optional, config has `gpu: true`) | probably, slowly | try Mac; fall back to Karolina |
| gen_2dhoi step 4 (video) | Kling API | n/a | **you generate manually** |
| recon_4dhoi (GEM, SAM2, MoGe, FoundationPose, nvdiffrast) | CUDA | **no** | Karolina |
| retarget (GMR, MuJoCo IK) | CPU | yes | Mac if deps install, else Karolina |
| tracking training (Isaac Sim / Isaac Lab, SONIC) | NVIDIA GPU | **no** | Karolina |
| orcs post-training (mjlab = MuJoCo Warp) | GPU for scale | no | Karolina |

The docs state timings on an L40S GPU: recon ~30-40 min per video, which is why it needs a cluster.

---

## 3. Git: what we did and why (Step 0)

GRAIL is a git repo whose `origin` remote is NVIDIA's GitHub.

* `git fetch origin` — downloads new commits into your local copy of `origin/main`. **It never edits your working files**, so it is always safe. Use it to *look* before you *change*.
* `git rev-list --left-right --count main...origin/main` — prints `ahead behind`. We saw `0 4`: zero local commits, 4 behind.
* `git pull --ff-only origin main` — "fast-forward only": move `main` forward along the same line of history. If you had your own commits that diverged, `--ff-only` **refuses** instead of silently creating a merge. That is why we use it by default.
* **Untracked files** (`??` in `git status`) are files git has never been told about. They survive pulls and are **not uploaded** unless you `git add` them. This matters for the later push: `kernel-metadata.json` is a Kaggle config, and we will not commit it unless you decide to.

### Preview of the push to `mg4637/grail` (not done yet)

Your clone points at NVlabs. To upload elsewhere we add a **second remote**, e.g. `git remote add mine https://github.com/mg4637/grail.git`, then `git push mine main`. Pushing publishes the content (even if deleted later it may be cached), so I will ask you first, show exactly what will be uploaded (`git status`, `git diff --stat`), and check the repo is private/empty as you intend. You will need GitHub authentication (a personal access token or SSH key); `gh` is not installed.

---

## 4. Stairs: what "3 steps" means in GRAIL

`grail/pipelines/gen_terrain.py` builds a staircase from boxes. Each stair is a box; boxes are stacked so the shape is a **pyramid**: you climb up the treads, reach a top, and the far side descends again. Ranges (metres):

* step count `3..8`, rise (height) `0.06..0.15`, tread (depth) `0.09..0.24`, width `1.2..1.8`

Randomness comes from `random.seed(7070 + i)`, so asset `stairs_i` is always the same. Replaying that seed gives exactly 3 steps for `stairs_007, 008, 027, 037, 039, 047`. So "a kid climbing 3 steps" = pick one of these. (`--num N` generates `stairs_000..N-1`.)

Next: set up a small Python environment on the Mac (needs your permission, section to come), generate the mesh, and look at it.

---

## 5. MuJoCo from zero (Lesson 1)

MuJoCo ("Multi-Joint dynamics with Contact") is a physics engine. GRAIL's retargeting (GMR) uses it for robot kinematics; `orcs`/`mjlab` use it to train robots. You need five ideas.

1. **Model (`MjModel`)** — the fixed description: bodies, joints, geoms (shapes), masses, timestep. Written in **MJCF**, an XML format. Never changes while simulating.
2. **Data (`MjData`)** — the changing state: time, positions `qpos`, velocities `qvel`, contacts. One model can have many data objects (this is how GPU training runs thousands of robots at once).
3. **Stepping** — `mujoco.mj_step(model, data)` advances time by `model.opt.timestep` (here 2 ms).
4. **Generalized coordinates** — a free body has `nq=7` position numbers (xyz + a 4-number quaternion for orientation) but `nv=6` velocity numbers (3 linear + 3 angular). `nq != nv` is normal and a classic beginner confusion.
5. **Geoms and contacts** — only geoms collide. After a step, `data.contact[:data.ncon]` lists touching geom pairs.

Gotcha: a box's `size` is its **half**-extent.

File: [lessons/01_mujoco_stairs_ball.py](lessons/01_mujoco_stairs_ball.py). Run:

```bash
cd GRAIL
docs_learning/.venv/bin/python docs_learning/lessons/01_mujoco_stairs_ball.py
```

It builds a 3-step staircase with the *real* rise heights of `stairs_007`, drops a ball at x=0.5 (above step 2), and prints that it rests at z = 0.335 + 0.05 = 0.385. Check this yourself: change `pos="0.5 0 1.0"` to `0.1` and predict where the ball lands before running.

**What is `mjlab`?** (taught fully when we reach the orcs stage) — a library from the MuJoCo team that re-implements Isaac Lab's "manager-based" RL API on top of **MuJoCo Warp**, a GPU version of MuJoCo. Same ideas as above, but `MjData` is replicated across thousands of parallel environments on a GPU, with managers for observations, rewards, terminations and commands.

## 6. Where we are

Done on the Mac: pull, venv, config fix, 3-step stairs verified, MuJoCo lesson. Next: the 2D-HOI rendering stage, which needs Blender.

---

## 7. Stage 2: the 2D-HOI render, and what broke (read this, it teaches a lot)

**What the stage does.** `grail.pipelines.gen_2dhoi` has 4 steps: (1) drop the object in a physics sim to find how it rests, (2) check its scale, (3) render the object + a character in an indoor scene with Blender, (4) feed the rendered image to a video model. We run steps 1-3 skipped/real as configured and **stop before step 4**. The image from step 3 is the **start frame** that conditions the video: the video model animates *this* picture, so the geometry in it (stairs, camera, kid position) is inherited by everything downstream — reconstruction later recovers exactly this scene. That is why the tread width matters: a bad stair in the frame becomes a bad stair in the robot's training terrain.

**Blender in 60 seconds.** Blender is a 3D renderer you can run with no window (`--background`) and drive with Python (`--python script.py -- args`; everything after `--` is for the script). GRAIL's `render_scene.py` runs *inside* Blender's own embedded Python, which is a different interpreter from your venv. That one fact caused bugs 1 and 3 below. Cycles is Blender's path-tracing engine; `samples` is rays per pixel (more = less noise, slower).

**Bugs, and the general lessons** (full table in implementation_history.md §2.4):
1. *Symlink to a macOS app binary fails*: apps locate their resources relative to their own path. Use a wrapper script that `exec`s the real path.
2. *Exit code 0 does not mean success*: Blender returns 0 on Python errors. Always verify outputs exist.
3. *Two Pythons*: packages installed in your venv are invisible to Blender's Python. We installed pure-Python packages into a separate folder (`--target`, built for 3.11) and exposed it with `PYTHONPATH` + `--python-use-system-env`. We used `--no-deps` so we would not shadow Blender's own numpy.
4. *Names must match*: config said `SynStairsV1`, folder was `syn_stairs`.
5. *Hard-coded hardware*: the code assumed CUDA. Mac Blender only knows METAL. Fixed with a try-each-backend loop. Pattern to remember: probe capabilities, don't assume them.

**Why the first stairs were bad, and how we made better ones.** `gen_terrain` draws each tread depth uniformly from 9-24 cm. Each stair is a box; a box's x-extent runs from `sum(treads_up[:i])` to `total - sum(treads_down[:i])`. So tread `i` is `treads_up[i]` deep and the top platform is `treads_up[-1] + treads_down[-1]` deep. Our `lessons/02_wide_stairs.py` fixes tread = 0.35 m, rise = 0.12 m. Try changing `TREAD` and re-running to see the geometry change, then check it with the trimesh snippet from section 4 (find upward-facing faces, list their heights).

**Check yourself:** the start frame lives at
`results/generation/asset_renders/SynStairsV1/stairs_wide3/kid_indoor1-syn-stairs_rand00001.png`. Open it. Can you spot the 3 tiers? Which direction will the kid walk?

## 8. HAND-OFF: you generate the video (what to do)

1. Open the start frame above in your video web UI (image-to-video).
2. Paste the prompt from `llm_answers.md` Entry 1.
3. Settings to mimic GRAIL's config: 10 s, 1 video, static camera, highest quality ("pro"). Keep the 16:9 aspect ratio of the image.
4. Download the mp4 and send it to me. I will put it at
   `results/generation/videos_kling/SynStairsV1/stairs_wide3/kid_indoor1-syn-stairs_rand00001.mp4`
   (the exact path recon expects: `{results_dir}/{video_dir}/{video_id}.mp4`).
5. If the kid walks off-frame or skips steps, regenerate; a bad video becomes bad robot data.

---

## 9. What happens to the video next

The video is a *2D* observation. `recon_4dhoi` turns it into 3D in 6 stages (docs: `recon_4dhoi.md`):
1. **GEM-SMPL** — a network predicts the kid's SMPL-X body pose per frame (poses are joint rotations + body shape).
2. **SAM2 + MoGe** — SAM2 tracks the kid's pixels (a mask per frame); MoGe predicts metric depth.
3. **FoundationPose** — tracks the object's 6-DoF pose. For terrain the object is static, so the `loco_smplx.yaml` config skips it (`is_static_obj: true`) and reuses the known stairs pose from the render step (that is what `first_frame_pose.pickle` and the camera pickle were for).
4. **HOI optimization** — adjusts the body so that feet land on the stairs at the right height and contacts are consistent (uses an LLM to say *which joints touch the object* in each interval; we will answer those ourselves).
5. **Filter** — rejects bad reconstructions (penetration, floating, jitter).
6. **Visualize** — renders overlays and an HTML viewer.

Why GPU: stages 1-4 run neural nets (CUDA only; `nvdiffrast`, FoundationPose C++ extensions). On an L40S the docs say ~30-40 min per video. This is the point where we leave the Mac.

Command that will run on the cluster:
```bash
python -m grail.pipelines.recon_4dhoi --dataset SynStairsV1 --category stairs_wide3 \
    --results_dir results --config configs/recon_4dhoi/loco_smplx.yaml
```
(`--dataset` must match the folder name under `videos_kling/`, i.e. `SynStairsV1`; the docs' `syn_stairs` example does not match our run.)

---

## 10. Karolina: how an HPC cluster works (lesson)

**Mental model.** A cluster is a shared pool of machines. You never run heavy work where you log in.
```
 your Mac --ssh--> LOGIN NODE (shared; edit files, copy data, submit jobs; NO heavy compute)
                        |  Slurm scheduler (queue + accounting against your PROJECT ID)
                        v
                  COMPUTE NODES (partition qgpu = GPU nodes; you get them only while your job runs)
```
Vocabulary: **partition** = a queue of machine types (`qgpu` = GPU nodes, `qcpu` = CPU nodes); **project ID** = who gets billed (`-A PROJECT-ID`, required on every job); **walltime** = max run time you request (`-t HH:MM:SS`; qgpu limit is 2 days per the IT4I docs); **interactive job** (`salloc`) = you get a shell on a compute node now; **batch job** (`sbatch script.sh`) = you submit a script and leave.

**Commands (verified in the IT4I docs, sources at the end):**

| Goal | Command |
|---|---|
| interactive GPU session | `salloc -A PROJECT-ID -p qgpu --gpus=1 -t 01:00:00` |
| submit a script | `sbatch script.sh` |
| my jobs | `squeue --me` |
| cancel | `scancel JOBID` (or `scancel --me` = all mine) |
| clear modules (fixes container problems) | `ml purge` or `ml -f purge` |
| container with GPU | `apptainer exec --nv image.sif nvidia-smi` |
| mount a folder into the container | `apptainer exec --bind /host/path:/container/path image.sif ...` |
| convert a Docker image | `apptainer pull name.sif docker://user/image:tag` |

Minimal batch script skeleton (from the docs, adapted to GPU; replace PROJECT-ID):
```bash
#!/usr/bin/bash
#SBATCH --job-name grail-test
#SBATCH --account PROJECT-ID
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --time 00:30:00
ml purge
apptainer exec --nv image.sif nvidia-smi
```

**Why Apptainer, not Docker.** Docker needs root, which a shared cluster cannot give you. Apptainer runs images as *you*. GRAIL's official environment is a Docker image (`docker.io/nvgrail/grail:1.1.1`, Linux AMD64), which Apptainer can convert with `apptainer pull ... docker://...`. `--nv` passes the host's NVIDIA driver into the container.

**Things I have NOT verified and will not assume** — your first session below answers them:
1. Where to put large data (home quota vs. a project/scratch area, and their sizes).
2. Whether compute nodes can reach the internet (GRAIL's setup scripts download checkpoints and Blender).
3. Which GPU model `qgpu` gives you (the docs mention 8 GPU nodes; GRAIL's timings were measured on an L40S).
4. The exact module name for Apptainer.

### Your first Karolina session (read-only; nothing here changes anything or uses compute)

Run these yourself in a terminal and paste the output to me. Replace `<login>` with your IT4I username.
```bash
ssh <login>@karolina.it4i.cz
# --- now on the login node ---
hostname
echo $HOME; pwd
df -h $HOME
ls -d /scratch /mnt/* 2>&1 | head
ml av Apptainer 2>&1 | head
squeue --me
exit
```
What each answers: `df -h $HOME` = how much space/quota your home has; `ls -d /scratch /mnt/*` = other storage areas that exist; `ml av Apptainer` = module name; `squeue --me` = whether you have jobs (should be empty).

**Security note:** I never see your keys or password; you type into your own terminal. Never paste a private key or password into this chat.

Sources: [Karolina intro](https://docs.it4i.cz/en/docs/clusters/karolina/introduction), [job submission](https://docs.it4i.cz/en/docs/general/run-jobs/job-sub-exec/job-submission-and-execution), [Apptainer](https://docs.it4i.cz/en/docs/software/tools/apptainer).

---

## 11. Git lessons from the upload

* **Public vs private matters before you push**: `git ls-remote <url>` works without credentials only for public repos. Anything you push to a public repo is world-readable and may be cached.
* **Unrelated histories**: `git merge-base A B` printing nothing means the two branches share no ancestor; a normal push of one onto the other is rejected (non-fast-forward). `--force` would erase the other side. Prefer a **new branch**.
* **Git LFS** stores big files outside git; commits hold small pointers. A push uploads *all* LFS objects reachable from the pushed history; one missing old object blocks the whole push. Don't "fix" it with `lfs.allowincompletepush` unless you accept dangling pointers.
* **Orphan branch** (`git checkout --orphan name`) starts a branch with no parent while keeping today's files staged: a snapshot without history. Trade-off: loses per-commit history on that branch; upstream remains the authority.
* **Check what you are committing**: `git diff --cached --name-only`, and compare the snapshot with upstream via `git diff --stat <upstream-sha> HEAD` (we got exactly 2 files).

## 12. Putting GRAIL on Karolina (plan; you run the commands, I never connect)

Facts used (verified in your output): scratch folder `/scratch/project/<project-id>/<you>/thesis` (created by you; see §13), project folder `/mnt/proj2/<project-id>` (group-writable), GPU = A100 40 GB, compute nodes have internet, Apptainer modules 1.1.5/1.2.5. Unknown: your storage quota in `/home` and the project folder.

**Layout** (all under your SCRATCH folder: fast, 20 TB per project; `/home` has only 25 GB):
```
/scratch/project/<project-id>/<you>/thesis/
    grail/            code (git clone of your snapshot branch)
    containers/       grail.sif (Apptainer image) + cache
    results/          copied from your Mac
```

**Step A: code (login node, uses no compute hours).**
```bash
export WORK=/scratch/project/<project-id>/$USER/thesis
mkdir -p $WORK/containers $WORK/tmp && cd $WORK
GIT_LFS_SKIP_SMUDGE=1 git clone -b grail-stairs-snapshot https://github.com/mg4637/grail.git grail
cd grail && git log --oneline -1       # expect ce62787
```
`GIT_LFS_SKIP_SMUDGE=1` skips downloading Git LFS files. **Correction (found later, §19): I said these were only demo GIFs/videos. They also include the G1 robot meshes and other assets that training needs, so those arrived as 130-byte stubs and had to be copied from the Mac.** If `git clone` complains about LFS not installed, tell me.

**Step B: container.**
```bash
ml purge && ml apptainer
export APPTAINER_CACHEDIR=$WORK/containers/cache APPTAINER_TMPDIR=$WORK/tmp
cd $WORK/containers
apptainer pull grail.sif docker://nvgrail/grail:1.1.1
```
The cache/tmp variables keep the multi-GB image build out of `/home`. I have not verified the image size; `df -h $WORK` afterwards tells us. (Tag name comes from GRAIL's README.)

**Step C: data from your Mac** (run on the Mac, from the GRAIL folder; replace `<login>`):
```bash
rsync -avh --progress results/ <login>@karolina.it4i.cz:/scratch/project/<project-id>/<login>/thesis/grail/results/
rsync -avh data/syn_stairs/stairs_wide3 <login>@karolina.it4i.cz:/scratch/project/<project-id>/<login>/thesis/grail/data/syn_stairs/
```
`rsync` copies only differences and can be re-run safely. `results/` is about tens of MB. (Careful: the trailing `/` on `results/` means "contents of".) `<login>` is your username (`$USER` on the cluster); the folder `.../<login>/thesis` already exists because you created it.

Please run Steps A and B and paste the output (especially the last lines and `df -h $WORK`). Do Step C after A.


## 13. Which Karolina storage to use (corrects my earlier plan)

I first suggested `/mnt/proj2/<project-id>/` from its group permissions alone. After you created `/scratch/project/<project-id>/<cluster-user>/thesis` and asked, I read the [IT4I storage page](https://docs.it4i.cz/en/docs/clusters/karolina/storage):

| Area | Use for | Limits (from the docs) |
|---|---|---|
| HOME `/home/<user>` | code, configs, small files | 25 GB/user, NFS, backed up |
| SCRATCH `/scratch/project/<project>/...` | **everything a running job reads/writes**: repo, container, checkpoints, results | 20 TB/project, Lustre, very fast; files **not accessed for 90 days are deleted** |
| PROJECT `/mnt/projN/<project>` | long-term storage of finished results | shared across IT4I clusters; quota not stated in what I read |
| TMP (local) | tiny temporary files | few GB, purged |

Rule: **work in SCRATCH, archive to PROJECT/Mac.** At the end of each stage we copy the keepers (reconstruction outputs, retargeted motions, trained checkpoints) out of scratch. `touch`-ing files to dodge the 90-day cleaner is poor practice; copy instead. The project allocation also ends on 2026-10-27.

---

## 14. Checkpoints and licensed body models (what you must do yourself)

**Why checkpoints exist.** A neural network is code plus *weights* (the learned numbers). The code is in git; the weights are big files (GBs) hosted on Hugging Face, so they are downloaded separately. Recon uses: GEM-SMPL (body pose from video, ~14 GB), FoundationPose weights (object tracking), and later SONIC (the robot tracking policy).

**Why SMPL/SMPL-X are different.** The SMPL family (the parametric human body) is licensed by the Max Planck Institute. You must create your own account, accept the license, and download the files; scripts and I cannot do this for you, and they must never be committed to GitHub or shared. GRAIL expects exactly:
```
imports/GEM-SMPL/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
imports/GEM-SMPL/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
```
The downloaded archives may use different file names; the destination names above are what GRAIL requires (check the files' names in the archive against these before renaming).

**Order of work (see message in chat for commands).** 1) you download the licensed models on your Mac; 2) copy data + models to Karolina with rsync; 3) on Karolina, init the submodules and run `download_checkpoints.sh` *inside the container* with `--skip-gem-soma --skip-hunyuan3d --skip-sonic`, with `HF_HOME` on scratch; 4) a short GPU test; 5) the real recon job.

---

## 15. Looking at the retargeted G1 motion in MuJoCo (hands-on)

**What the motion file is.** `data/motion_lib/stairs_wide3/robot/<name>.pkl` (written with `joblib`) holds, for each of 240 frames at 24 fps:
`root_trans_offset (x,y,z)` of the pelvis in metres, `root_rot` as a quaternion in **(w,x,y,z)** order, and `dof` = 29 joint angles in radians. That is exactly the G1's MuJoCo state: **qpos = [x y z | qw qx qy qz | 29 angles] = 36 numbers**. `nq` (36) is bigger than `nv` (35) because a quaternion has 4 numbers but a rotation only 3 velocities, as in the box lesson (§5).

**Kinematic playback, no physics.** We write each frame into `data.qpos` and call `mujoco.mj_forward(m, d)`. `mj_forward` computes where every body is (forward kinematics) but never advances time. So you see the *reference* motion. Training later asks an RL policy to reproduce it with real physics (gravity, contact, motors). If the reference is impossible (e.g. feet floating), the policy cannot copy it exactly.

**Two ways to open it**
1. **Interactive window (my script).** `docs_learning/.venv/bin/mjpython docs_learning/lessons/03_view_retargeted_motion.py`. On macOS the viewer must run under `mjpython`, not `python`, because the window must own the main thread. Useful options: `--speed 0.25` for slow motion. Mouse: left-drag rotates, right-drag pans, scroll zooms; F1 shows the viewer's built-in help.
2. **MuJoCo.app (simulate).** You can drag `imports/GMR/assets/unitree_g1/g1_mocap_29dof.xml` onto it to see the robot and move joints by hand with its sliders, but it has no way to play our pkl file, which is why the Python script exists.

**Offscreen video:** add `--video out.mp4` (use `python`, not `mjpython`). I made `docs_learning/frames/retargeted_g1.mp4` this way.

**Two checks you can do yourself.** (1) Watch the feet at the top of the staircase: do they sit on the platform, or hover a few cm above it? My forward-kinematics measurement says 3-6 cm of hover at the end (uncertainty ~2 cm); see results.md. (2) Pause at a mid-climb frame and see whether the swing foot clears the step edge.

**Bug lessons from this script (both mine):** a name defined twice in one MuJoCo model raises `repeated name`; and `import x.y` inside a function makes `x` a local variable of that function in Python, which breaks earlier uses of `x` in the same function.

---

## 16. Contact, hover, and why it matters for RL (lesson from the replay)

**What you saw, in numbers.** The retargeted G1 hovers a few centimetres above the steps. Script `lessons/04_contact_gap_check.py` measures it: for each frame, `gap = lowest point of each foot - height of floor/step directly under that point`. gap > 0 means hovering, ~0 touching, < 0 inside the stairs. Our result (results.md): median +3.5 cm, supporting foot +3 to +7 cm during the climb, +3.3 cm standing on top; the pelvis rises 43 cm for a 36.6 cm staircase.

**Why this matters.** The reference motion is what the policy is rewarded for matching. In simulation the stairs are solid: a foot cannot hover 5 cm above a step and still support the robot. A policy trained on a floating reference must either pay a tracking error constantly or learn to deviate from the reference. Small errors are often tolerated by RL, but we do not know yet how tolerant this setup is; training will show, and we should say so honestly instead of assuming.

**Where does the hover come from? (we measured it)**
* The retargeted robot hovers (median +3.5 cm). We then ran the same measurement on the *reconstructed human* (`lessons/05_human_vs_stairs.py`: build the G1-proportioned SMPL-X body from `hoi_data.pkl` and measure foot vertices against the stairs). Result: the human already hovers, slightly more (median +5.4 cm; +5 to +9 cm while climbing; +5 cm on top), while at frame 0 it is grounded within 1 cm (which also shows the measurement itself is sound). **So the hover is created by the reconstruction, not by retargeting.** It was invisible in the recon overlay video because a 5 cm vertical offset cannot be seen from one camera view.
* Why did the quality filter say "Valid"? Its checks (initial position, mask overlap, 2D keypoint error, contact, initial penetration, average velocity) never measure the distance between feet and the terrain per frame.
* Why does nothing pull the feet down? (reasoning from the code, not a proven cause) `human_foot_contact` only penalises the *velocity* of feet predicted to be in contact (it stops sliding; it has no idea where the ground is); the loss with the stairs is zero for a static object because there are no contact labels; `depth_pointcloud` gives depth cues but no "foot on step" constraint. Vertical position is therefore set by 2D keypoints, a strong prior to the initial pose estimate (weight 3000) and depth. A single camera cannot judge a few centimetres in height well.
* The retargeting stage also has no ground correction (`fix_height` is documented but not implemented), so it simply inherits the offset. My earlier idea that the joint scaling in `retarget.py` shrinks the legs is wrong for this run: in the G1 path `scale` is forced to 1.0.
* **Lesson:** to find where an error enters a pipeline, measure the same quantity at each stage and compare (here: reconstruction vs retargeting). Do not blame the last stage.

**Possible remedies** (none applied yet; each changes the data, so they need your decision):
1. *Kinematic contact fix:* per frame, shift the root vertically (smoothed) so the lower foot just touches the surface under it. Cheap, but it moves the pelvis, which may bend legs unrealistically if the feet were really mid-swing.
2. *Check the pipeline options* (`--height_offset`, `--downstairs_initial_height_correction`) and the body scale in the retargeting step.
3. *Train anyway* and let the RL tracking policy absorb the error, then judge from training curves and playback.

**General lesson.** "The pipeline finished and the filter said valid" is not the same as "the motion is physically right". Check the quantities that matter downstream (here, foot-to-surface distance), and compare against what you see.

---

## 17. Adding a new loss term to GRAIL's optimizer (what we did and why it is trustworthy)

**Problem (measured, §16):** nothing in the reconstruction ties the feet to the step surfaces, so the kid's feet hover 5-9 cm above the stairs.

**Idea:** add a loss term. In an optimization-based reconstruction the result is the minimiser of a weighted sum of terms: `keypoint_tracking` (match the 2D pose in the video), `depth_pointcloud` (match the scene depth), smoothness terms, priors, ... Each term pulls the body somewhere; the weights set who wins. A missing physical constraint (feet on the ground) is added as another term.

**Our term `terrain_support`** (code: `grail/optimization/terrain.py`, `loss_computer.py::_terrain_support_loss`):
1. Build a **height map** of the stairs once: rasterise the mesh triangles onto a 1 cm XY grid and keep the highest surface per cell (floor where there is no mesh). Done in the stairs' own frame, then every iteration the feet are transformed into that frame with the *current* stairs pose, because the optimizer also adjusts that pose slightly.
2. For each frame find the foot vertex that is lowest *relative to the terrain under it*: `gap = z_foot - height_under_foot`.
3. Loss = weight x mean(gap of the lower foot ^2) + weight x mean(max(0, -gap of the other foot)^2). The lower foot is assumed to be the supporting one; the other foot may lift but not sink into the stairs. Squared, so small errors are tolerated and big hovers are pulled hard.
4. Heights are looked up as constants; the gradient acts on the foot height.

**How we checked it before using GPU hours (a good habit):**
* Height map vs the analytic stairs formula on 3,731 random points: max error 0.00 cm.
* Real loss code on the real reconstruction (CPU, container): the gaps matched my independent measurement from §16 exactly.
* Gradient sign: positive over hovering frames, so optimisation pushes the body down.
* Sanity: an artificially grounded body gives loss ~0.03 instead of 3.1.
This does NOT prove the full optimisation will give a good motion; the terrain term competes with `keypoint_tracking` etc. That is what the next run will show.

**Two lessons from the failures along the way:** (1) the cluster container has *separate* Python environments (`grail` with pytorch3d, `sonic` with mink/GMR/Isaac); a script must only import code whose dependencies exist in the env it runs in (`retarget.py` imports `mink` at the top, so it cannot be imported from `grail`). (2) Never derive repo paths from `__file__` in scripts that get copied around.

---

## 18. From a reference motion to a policy: what "training" means here

**The goal.** We have a *reference*: a 10 s clip of G1 joint angles climbing three steps. A robot cannot just replay joint angles: with gravity, contact and motor limits it would fall. Training produces a **policy**, a neural network that outputs motor commands every control step so that the *physically simulated* G1 follows the reference as closely as it can.

**Reinforcement learning (PPO) in one paragraph.** Thousands of copies of the robot ("environments") run in parallel on the GPU. In each step the policy sees an observation (its own joint state and history, the next few reference frames, a height map of the terrain around the robot, the stairs' pose) and outputs an action. The simulator advances; a **reward** says how well the robot matches the reference (here: five tracked body points: torso, both wrists, both ankles, plus penalties such as foot acceleration); a **termination** ends an episode if the robot deviates too much (falls). One **iteration** = collect 24 steps from every environment, then update the network once with PPO. 6,000 iterations x 24 steps x N environments is the amount of experience.

**Warm start (finetuning).** GRAIL ships a policy already trained on stairs (`terrain_stairs_warmstart`). Starting from it and training on our single clip is much faster than learning to walk from scratch. `++resume=True ++checkpoint=<file>` loads the weights and keeps training; the same flags let us continue our own run later.

**Why the reference quality matters (§16).** The policy is rewarded for matching the reference. If the reference hovers 5 cm above the step, the reward asks the robot to do something the physics forbids. That is why we spent effort on the hover before training.

**Practical set-up facts** (from the GRAIL/SONIC docs and config): entry point `imports/SONIC/train_agent_trl.py` (Hydra), config `scene/terrain_tracking`, data folder with `robot/`, `objects/`, `object_usd/` (matching file stems), a checkpoint every 500 iterations, `num_envs` 4096 by default (we will use fewer; one clip does not need 4096 copies).

**Our staged plan:** download the pretrained checkpoints; a 3-iteration smoke test that proves Isaac Sim runs in our container and measures speed; then the ~6,000-iteration run; judge by the training curves and by replaying the rollout; extend with `--resume` if needed.


---

## 19. Git LFS stubs: the bug that looked like a graphics problem

**What Git LFS is.** Large binary files (meshes `.STL`, `.usd`, model weights `.pt`, `.pkl`, images) are not stored inside git. The repository holds a tiny text *pointer* (about 130 bytes: `version https://git-lfs.github.com/spec/v1`, `oid sha256:...`, `size ...`) and the real file lives on an LFS server. A normal `git clone` replaces pointers with real files ("smudge"). With `GIT_LFS_SKIP_SMUDGE=1` you keep the pointers.

**What happened to us.** I told you to clone on Karolina with `GIT_LFS_SKIP_SMUDGE=1` to avoid downloading demo GIFs. But `imports/SONIC` also keeps the G1 robot's 105 mesh files in LFS (`.gitattributes`: `*.STL`, `*.usd`, `*.pt`, ...). On the cluster they were stubs. When Isaac Sim tried to build the robot from its URDF, it read stubs instead of meshes, wrote no intermediate USD layer, and crashed with `Failed to open layer ... pelvis.tmp.usd` / `NULL UsdStage`. That message does not mention meshes or LFS at all.

**Why it looked like a graphics bug.** The log was full of Vulkan/GPU errors before the crash: real errors, but caused by the A100 having no ray-tracing cores, and harmless for this run. We spent several attempts on Vulkan (an empty ICD file inside the container, a driver-version error) before reading what the failing step actually needed.

**How we found it:** (1) read the converter source to see what it builds from (a URDF), (2) read the URDF to see what it references (STL meshes), (3) look at those files on the cluster: `ls -lL` showed 130-byte files and `head` showed the LFS pointer text. The same check on the Mac showed real 1.8 MB meshes.

**General lessons.**
* A crash message from deep inside a framework often names the *last* thing that failed, not the cause. Follow the inputs of the failing step.
* A file that is "there" can still be a stub. Check size and header (`ls -l`, `head -c 100`) of data files after cloning a repo that uses LFS. `git lfs ls-files` marks `*` for present content and `-` for pointers.
* Noisy errors that appear in every run (here Vulkan on an A100) are not evidence for the crash that comes after them. Test the alternatives instead of fixing the loudest message.
* Correct clone recipe for a repo like this: `git lfs install` then `git lfs pull` (or copy only the needed asset folders, as we did).

---

## 20. Looking back: was the new loss a good idea, and what changes for the chair/bookshelf task

### Was adding `terrain_support` to the reconstruction a good idea?
**For this staircase clip: reasonable, with clear limits.**

What was done right:
* The problem was *measured* (hover of 5-9 cm), and we found *where* it entered (reconstruction, not retargeting) before changing anything.
* The fix was isolated (new loss, new config, new output folder; the original pipeline and results are untouched) and tested on the CPU against an independent measurement before spending GPU hours.
* It worked on the quantity we cared about: median hover +5.4 -> +1.7 cm, frames with a foot within 5 cm of the surface 96/240 -> 240/240, no penetration.

What is weak or unproven:
* **It is a heuristic.** "The lower foot of every frame is the supporting foot and must touch the surface" is true for walking and stair climbing. It is false for jumps, kneeling, sitting, crawling, climbing with hands, or any frame where the person is supported by something other than the terrain mesh.
* **"Floor" is defined as the lowest vertex of the object mesh, in the object's frame.** That is only correct when the object is a static terrain standing on the floor. For an object that is *carried*, the "floor" would move with it: wrong.
* **It trades against the video:** keypoint error went from 0.48 to 0.66 (still valid), and because the object pose is also optimised, the optimiser moved the stairs by 5 mm to help. For static terrain the object pose should probably be frozen.
* **One weight (1000), one clip, no ablation.** We do not know yet whether the reduced hover actually makes the trained policy better. The honest test is to train on trial 1 and trial 2 with the same settings and compare tracking error and failure rate.
* **The robot motion of trial 2 was not checked before training** (my omission): run `lessons/04_contact_gap_check.py data/motion_lib/stairs_wide3_terrain` on the Mac.
* Alternatives we did not test: a cheap post-hoc vertical snap of the robot motion, or simply letting RL absorb a few cm.

For a thesis this is a defensible *contribution* only with that ablation; as a pipeline fix it is fine for static terrain.

### Other lessons from the session (things that cost the most time)
1. **Silent failures and untrustworthy exit codes**: Blender returned 0 after crashing; job logs printed `exit code: 0` after a traceback; GRAIL printed "OK" for nothing. Always check for the expected output files.
2. **The loudest error is often not the cause**: Vulkan errors on the A100 looked fatal; the real cause was LFS pointer stubs for the robot meshes (from my `GIT_LFS_SKIP_SMUDGE=1` instruction).
3. **Documentation and code disagree**: GRAIL docs give a wrong training-script path; `fix_height` is documented but not implemented; `--mesh_scale` help says 0.6 though the scale is baked in; a config pointed to a non-existent `_v1` file.
4. **Two Python environments in one container** (`grail` with pytorch3d, `sonic` with mink/Isaac) and **two terminals** (Mac vs cluster) caused several avoidable failures.
5. **"Valid" does not mean physically right**: the reconstruction filter never measures foot-to-terrain distance.
6. **Allocation**: the project's GPU hours are shared with your other jobs (24.51 -> 389.64 node-hours in 6 days), and the project ends 2026-10-27.

### What changes for the target task: carry a chair next to a bookshelf, climb on it, take an object from the top
This is no longer one "terrain" clip; it combines three interaction types GRAIL treats separately:

| phase | object role | GRAIL path today |
|---|---|---|
| carry the chair | *moving* object held by the hands | manipulation config (`manip_smplx.yaml`): FoundationPose tracking, LLM contact labels every 8 frames |
| climb on the chair | chair becomes *static* support (terrain) | loco/terrain config, `terrain_support`-like constraint needed |
| take the item from the shelf | second object, hand contact; bookshelf is static scenery | pickup config; a second tracked object |

Consequences:
* **Multiple objects** (chair, bookshelf, item): the reconstruction tracks one object per video. Expect to split into sub-clips or extend the pipeline.
* **Video length**: one 10 s clip will not contain all phases; GRAIL supports multi-segment prompts (`#`-separated, `num_video_segments`), each continuing from the last frame.
* **Many more manual LLM answers**: contact labelling for moving objects runs per 8-frame interval (about 30 calls per 10 s clip) plus interaction detection; our wrapper can serve them, but each needs a looked-at answer.
* **The terrain loss must be generalised before reuse**: world floor plane plus the object's surface (`max(floor, object surface)`), contact-gated support (use `foot_contact_probs` or low foot velocity, not "the lower foot"), and only while the chair is static.
* **Training**: the released SONIC configs are either terrain (`terrain_tracking`, static object) or manipulation (`hoi/pnp_*`); none covers carrying + climbing + reaching. That is research work, not configuration.
* **Robot feasibility**: check heights at G1 scale (about 0.76 of the human mesh): chair seat, shelf top, reach of the arms. Measure with the gap/reach scripts before generating videos.
* **Evaluation on Karolina**: GRAIL's rendered evaluation needs Isaac Sim graphics, which fail on the A100; plan to export rollouts and replay them in MuJoCo on the Mac.

Suggested order: (1) finish the stairs loop (measure trial-2 robot gap, evaluate run1, compare with a trial-1 run); (2) do each phase of the new task as its own small clip, reusing the measurement scripts at every stage; (3) only then combine.

---

## 21. Evaluating a policy properly (what the E0 -> E1 investigation taught)

**1. Evaluate the thing you want, in the way you want it used.** Training metrics were computed from random start phases (mean episode length about 245 steps). The question "does the robot climb the whole staircase starting from the beginning?" needed a different test: start at frame 0 and run the entire clip. Same policy, different question, different answer.

**2. Make failures observable before changing anything.** The robots reset at about 5 s. Instead of guessing which rule fired and loosening rules one by one, we added a *passive* logger (`docs_learning/cluster/term_logger.py`, a subclass of SONIC's recorder that appends the name of every termination rule that fires). It changes no behaviour. The first run with it showed that the finetuned policy has no failure terminations at all.

**3. One behaviour change per experiment.** E0 vs E1 differ only by the checkpoint. That is why we can say "finetuning removed the early failures" with some confidence (the E0 failure reason itself was never logged, so we do not claim to know it).

**4. Quantities to report for a tracking policy** (all from `08_rollout_report.py`):
* *reset/termination events*: which rule, when;
* *root distance to the reference over time*: how closely the robot follows the clip (here mean 6.5-6.9 cm);
* *pelvis height range*: did it actually climb (0.76-1.15 m, like the reference);
* *foot gap to the surface under physics*: a physical contact check that is independent of the reference (median -0.1 cm: the feet really are on the steps).
Look at the video as well; numbers and eyes catch different problems.

**5. Why this agreement is meaningful.** In simulation a foot cannot hover. So the policy's feet touching the stairs (gap about 0) is a physical fact, and it also validates our analytic stairs model against the simulator's stairs (they agree to about 5 mm).

**6. Caveats that remain:** one clip, one seed, 4 robots with the same start; the contribution of the terrain loss (trial 2 vs trial 1) is not isolated; the policy is finetuned and evaluated on the same clip (it may be overfitted to it, which is acceptable for a data-generation experiment but not for a claim of generalisation).

**7. SSH lesson (access).** An SSH connection that is already open keeps working even if the server stops accepting your key; new connections fail. When keys misbehave: (a) read `ssh -v` output (`Offering public key ... ` then `No more authentication methods` means the server rejected the key), (b) compare fingerprints (`ssh-keygen -lf`) on both sides, (c) fix access while a session is still open, (d) never paste the private key anywhere.

---

## 22. Contact points and forces in a MuJoCo replay: what you may and may not conclude

**Question that triggered this section:** "In the replay I see contacts between robot and floor, but none between robot and stairs, and pressing `f` shows no forces. Why?"

**1. Two simulators were involved.** The policy was trained and evaluated in **Isaac Sim (PhysX)**. What we recorded is only *poses* (joint angles, root position/orientation). We then replay those poses in **MuJoCo**, a *different* physics engine, using our own copy of the stairs (three boxes) that we built in the viewer script. So anything MuJoCo draws is MuJoCo's opinion about the recorded poses, not what PhysX computed.

**2. Why no stairs contacts: a collision filter we set ourselves.** In MuJoCo two geoms collide only if `(contype_A & conaffinity_B) != 0` or `(contype_B & conaffinity_A) != 0`. Our stair boxes had `contype=0, conaffinity=0`, so they could never collide with anything. We had done that on purpose to keep the replay "clean", but it also hid the contacts you wanted to see. The robot's feet are four small spheres per ankle (the other collision shapes are meshes and cylinders); the floor has `contype=1, conaffinity=0`.

**3. A subtle MuJoCo fact (we fell into it).** Changing `geom_contype` in memory after the model is compiled did nothing, because MuJoCo also stores per-*body* collision bits computed at compile time and uses them to skip whole body pairs early. To change collision filtering, change it where the model is *built* (in the spec/XML), not afterwards. `mujoco.mj_geomDistance` ignores the filter and gave the true geometry: the foot spheres overlapped the steps by about 4.7 mm, so contacts were geometrically there all along.

**4. Why `f` showed nothing useful.** The replay is **kinematic**: each frame we write `qpos` and call `mj_forward`. There is no time stepping, no velocity, no motors. For each contact MuJoCo then reports the force needed to push the overlapping shapes apart *at that instant*. That is not the force the legs carried: summed over the feet it was 5-150 N while the robot weighs 337 N. Treat such arrows as a debugging artefact, never as ground-reaction forces.

**5. Where the real forces are.** Isaac Sim's environment has contact sensors (`contact_forces` on all robot links, `feet_terrain_contact_sensor` between feet and stairs). The rewards use them (a foot with vertical force above 1 N counts as "in contact"). We had not recorded them. Our passive logger v2 now saves them (`forces.npz`), and the viewer can draw them (`--forces`): these arrows come from PhysX, not from MuJoCo.

**6. A sanity check you can always do on real forces:** when the robot stands still, the summed vertical force on the feet should equal its weight (mass x 9.81). Large deviations mean you are not looking at what you think (wrong sensor, wrong frame, robot accelerating, or a bug).

**7. Limits to remember:** arrows start at the ankle link (the contact position was not recorded); the sensors report net force on the foot link, which includes all its contacts (floor, stairs); and the MuJoCo robot geometry differs slightly from Isaac's collision shapes, so MuJoCo "contacts" and PhysX contacts need not appear on exactly the same frames.

---

## 23. Publishing work safely (what we did before pushing to a public repository)

**Threat model.** A public repository is readable by everyone, forever (forks and caches keep copies even after deletion). "Secret" is wider than passwords: cluster user names and project ids, e-mail addresses, IP addresses, key fingerprints, personal paths, licensed files (SMPL/SMPL-X models), model checkpoints and API keys all count.

**Procedure we used.**
1. *Separate private from public at the file level:* real code goes into the repository as it is; notes and scripts that contain real paths go through a sanitizer first.
2. *Whitelist, don't blacklist:* the generator copies only named files, so a new private file cannot slip in by accident.
3. *Replace identifiers with placeholders* (`<cluster-user>`, `<PROJECT-ID>`, `<THESIS_DIR>`), keeping the text readable.
4. *Scan the output* for leftover identifiers and for credential-shaped strings (private-key blocks, `ssh-ed25519 AAAA...`, tokens that start with `hf_`, `ghp_`, `sk-`, `AKIA`, assignments like `OPENAI_API_KEY=...`, IPv4 addresses, very large files).
5. *Test the scanner with planted fakes.* A scanner that has never found anything proves nothing.
6. *Scan the staged diff itself* (`git diff --cached`), because it is what will actually be published.
7. *Keep the sanitizer local:* it contains the strings it removes.
8. *Commit locally first, push only after looking at the file list.* Local commits can be amended freely; pushed ones cannot.
9. *Check the commit identity:* the author name and e-mail in a commit are public too (`git config user.name/user.email`).
10. *Never rely on `.gitignore` alone for secrets:* it only protects untracked files; anything already committed stays in history.

**Large files and licences.** Git LFS stores big binaries outside git (pointers in the repository); uploading LFS objects uses your hosting quota. SMPL/SMPL-X must not be redistributed (their licence forbids it); checkpoints and datasets belong on their own hosting.
