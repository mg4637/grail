# Implementation history

A chronological log of every action taken, why, and what changed.
Format: **what** / **command** / **risk** / **permission** / **effect**.

Machine: MacBook Pro M2 (arm64, macOS, Darwin 25.6), no NVIDIA GPU.
Date started: 2026-10-05.

---

## Step 0.1 — Inspect the workspace (read-only)

- **What:** listed `<THESIS_DIR>`, checked the GRAIL git state and installed tools.
- **Commands:** `ls -la`, `git remote -v`, `git status -sb`, `git log --oneline -3`, `which git python3 uv conda ffmpeg gh ssh`.
- **Risk:** none (read-only). **Permission:** not needed.
- **Findings:**
  - `GRAIL/` is its own git repo, `origin = https://github.com/NVlabs/GRAIL.git`, on `main` at `aa31d82`.
  - Untracked (never committed) items: `.idea/`, `GRAIL.pdf`, `kernel-metadata.json` (Kaggle config for account `<kaggle-user>`), `imports/SONIC/gear_sonic/models`.
  - Tools present: git, python 3.10.13 (pyenv), uv, ffmpeg, conda, ssh. Missing: `gh` (GitHub CLI).
  - Sibling folders: `orcs/` (post-training library), `mujoco_course/`, `vibe/`, `HOI-PAGE/`, `hoipage_project/`, `.ssh_cluster/` (cluster SSH material; contents deliberately not read).

## Step 0.2 — `git fetch` (network, no local change)

- **Command:** `git fetch origin`
- **Risk:** very low: only updates `origin/main` bookkeeping, never your files.
- **Permission:** implied by "pull the latest".
- **Effect:** discovered `main` was 4 commits behind. `git diff --stat` showed only `README.md` changes (CoRL 2026 citation/links).

## Step 0.3 — `git pull --ff-only origin main`

- **Permission:** explicitly asked and granted.
- **Why `--ff-only`:** it refuses to create a merge commit; if histories had diverged it would abort instead of altering anything.
- **Effect:** `aa31d82 → 8b9afa5`. Only `README.md` changed (13 insertions, 10 deletions). Untracked files untouched.

## Step 0.4 — Read the documentation (read-only)

Read `README.md`, `docs/source/{quick_start,gen_2dhoi,recon_4dhoi,retargeting}.md`, `configs/gen_2dhoi/terrain_stairs.yaml`, `grail/pipelines/gen_terrain.py`, and the `orcs` readme/CLAUDE.md.

### Findings that affect the plan

1. **Stairs are procedural.** `gen_terrain` builds a staircase from a seed (`7070 + index`). Step count is `random.randint(3, 8)`. I reproduced that calculation with the standard library: assets `stairs_007, 008, 027, 037, 039, 047` have exactly **3 steps** (the 50 already in `data/syn_stairs/`).
2. **Config naming mismatch (possible bug).** `configs/gen_2dhoi/terrain_stairs.yaml` points to `configs/objects/terrain_syn_stairs_v1.yaml`, but the repo only has `terrain_syn_stairs.yaml`. Not changed yet; needs your decision (see guide).
3. **Stage hardware needs** (from the docs): see the table in `guide.md` §2.
4. **OpenAI usage points:** prompt refinement (2D-HOI step 4) and contact-joint labelling (4D recon step 4). Both will be answered by Claude and logged in `llm_answers.md`.
5. **Kling usage point:** only 2D-HOI step 4 — we stop and hand you the starting frame.

## Step 0.5 — Created `docs_learning/`

- **Files created:** `guide.md`, `implementation_history.md`, `results.md`, `llm_answers.md`.
- **Location:** inside the GRAIL repo, at your choice. They are untracked; they only reach GitHub if we `git add` them.
- **Risk:** none (new files).

---
*(next entries are appended below as work proceeds)*

## Step 1.1 — Local Python venv (needs your permission: granted)

- **Command:** `uv venv --python 3.10 .venv` then `uv pip install --python .venv/bin/python numpy trimesh mujoco` (run inside `docs_learning/`).
- **Risk:** low. Everything is inside `docs_learning/.venv`; delete the folder to undo.
- **Installed:** numpy 2.2.6, trimesh 5.1.1, mujoco 3.14.0 (+ glfw, pyopengl, absl-py, etils, ...).
- **Extra change:** appended `docs_learning/.venv/` to `.git/info/exclude` (a *local-only* ignore list, never pushed) so the venv can't be committed by accident.

## Step 1.2 — Config fix in the repo (permission: granted, "edit in place")

- **File:** `configs/gen_2dhoi/terrain_stairs.yaml` line 5.
- **Change:** `object_config: configs/objects/terrain_syn_stairs_v1.yaml` -> `configs/objects/terrain_syn_stairs.yaml`.
- **Why:** the `_v1` file does not exist in the repo; the pipeline would fail to load the object list.
- **Consequence:** this is a tracked-file modification and will appear in `git diff` / be part of what is pushed to `mg4637/grail`.

## Step 1.3 — Verify stairs_007 really has 3 steps (read-only)

- Script (scratchpad, not in repo): loads `data/syn_stairs/stairs_NNN/model.obj` with trimesh and lists distinct heights of upward-facing faces.
- **Mistake made and fixed:** I first named the script `inspect.py`, which shadows Python's stdlib `inspect` and crashed `trimesh` with a circular-import error. Renamed to `stairs_info.py`. (Lesson: never name a script after a stdlib module.)
- Result: see results.md.

## Step 1.4 — MuJoCo lesson script

- **Created:** `docs_learning/lessons/01_mujoco_stairs_ball.py` (new file; runs headless; no window).
- Result: see results.md; explanation in guide.md section 5.

## Step 2.0 — Finding about Blender (read-only)

- `grail/pipelines/gen_2dhoi.py:38` hard-codes `imports/blender/blender` as the Blender executable.
- `scripts/setup/install_env_docker.sh` only downloads the **Linux x64** build of Blender 4.3.2.
- Consequence: on a Mac we would need the macOS arm64 Blender 4.3.2 and a link at that path. Decision pending with the user.

## Step 2.1 — Blender symlink (permission: granted)

- Existing install: `/Applications/Blender.app` = **Blender 4.4.3** (GRAIL targets 4.3.2 — minor-version mismatch, a known risk; will be reported if rendering breaks).
- **Commands:** `mkdir -p imports/blender && ln -s /Applications/Blender.app/Contents/MacOS/Blender imports/blender/blender`
- **Why:** `grail/pipelines/gen_2dhoi.py:38` hard-codes `imports/blender/blender`. A symlink makes GRAIL find your Mac Blender without editing code or downloading anything.
- **Check:** `imports/blender/blender --background --version` prints `Blender 4.4.3`.
- `imports/blender` is git-ignored (verified with `git status`), so it is not part of any push.

## Step 2.2 — Dependency check for gen_2dhoi (read-only)

- `docs_learning/.venv/bin/python -c "import grail.pipelines.gen_2dhoi"` fails with `No module named 'requests'`. Full `setup.cfg` dependency list includes torch/open3d etc., which gen_2dhoi does not need at import time.
- Decision pending: install a light subset into the lesson venv.

## Step 2.3 — Light dependencies installed in the lesson venv (permission: granted)

- `uv pip install requests openai pyyaml omegaconf pillow opencv-python-headless imageio ffmpeg-python scipy natsort`, then `tqdm` (import error showed it was needed; it is in GRAIL's `setup.cfg`).
- Result: `import grail.pipelines.gen_2dhoi` succeeds; `--help` lists all options including `--skip_step4`, `--object_config`, `--num_rand_scenes`.
- `openai` is installed only as a library; no key exists and no API call is made.

## Step 2.4 — Getting Blender to render on the Mac (bugs found, in order)

| # | Symptom | Root cause | Fix | Repo change? |
|---|---|---|---|---|
| 1 | `Internal error initializing Python ... No module named 'encodings'` | Blender (macOS) finds its bundled Python relative to the launched path; my symlink broke that | Replaced symlink with wrapper script `imports/blender/blender` that `exec`s the real binary | no (git-ignored dir) |
| 2 | Pipeline printed "OK" but nothing rendered | Blender exits 0 even when its Python script crashes; GRAIL trusts the exit code | none; **always check output files exist** | no |
| 3 | `No module named 'yaml'` inside Blender | Blender uses its own Python 3.11, not our venv | `uv pip install --target docs_learning/blender_site --python-version 3.11 --python-platform aarch64-apple-darwin --no-deps pyyaml opencv-python-headless<4.12 imageio trimesh pillow tqdm`; wrapper adds `--python-use-system-env` and `PYTHONPATH` | no (`blender_site/` excluded in `.git/info/exclude`) |
| 4 | `No mesh ... under data/SynStairsV1` | object config says dataset `SynStairsV1`, folder is `syn_stairs` | `ln -s syn_stairs data/SynStairsV1` | no (`data/` git-ignored) |
| 5 | `enum "CUDA" not found in ('NONE','METAL')` | `render_scene.py:134` has `--gpu` as `store_true, default=True` (cannot be disabled) and `setup_gpu_rendering()` assigns `"CUDA"` outside its try/except | Patched `grail/pipelines/blender/utils.py`: loop over `("CUDA","METAL")` with try/except, else CPU; dropped dead OpenCL branch | **yes** (tracked file, ~27 lines; permission granted after I explained Karolina vs Mac trade-offs) |

Wrapper content and the symlink-vs-wrapper lesson are in guide.md §7.

## Step 2.5 — Render of stairs_007 and user feedback

- First render (2 min, Mac) of `stairs_007` worked. **User feedback: treads too narrow** — stairs_007 has 9–24 cm treads (random range in `gen_terrain.py`), so the kid cannot place his feet comfortably.

## Step 2.6 — Custom wide 3-step staircase (user request)

- **New file:** `docs_learning/lessons/02_wide_stairs.py`, run with `PYTHONPATH=.` (the first run failed with `No module named 'grail'`: Python puts the *script's* folder on `sys.path`, not the working directory).
- Reuses `_box_mesh`, `_random_color`, `export_terrain` from `gen_terrain.py` (no edit to that file). Parameters: 3 steps, rise 0.12 m, tread 0.35 m up and down (top platform 0.70 m), width 1.5 m.
- **Output:** `data/syn_stairs/stairs_wide3/{model.obj,model.mtl,texture.jpg}` (git-ignored).
- **Verified with trimesh:** size 2.1 x 1.5 x 0.36 m; step tops 0.12, 0.24, 0.36 m.
- **Local object config:** `docs_learning/configs/objects_wide3.yaml` = copy of `terrain_syn_stairs.yaml` plus a `stairs_wide3` entry (repo config untouched).
- **Render command:**
  `docs_learning/.venv/bin/python -m grail.pipelines.gen_2dhoi --config configs/gen_2dhoi/terrain_stairs.yaml --object_config docs_learning/configs/objects_wide3.yaml --category stairs_wide3 --results_dir results --skip_step4 --num_rand_scenes 1` (37 s).
- **Output frame:** `results/generation/asset_renders/SynStairsV1/stairs_wide3/kid_indoor1-syn-stairs_rand00001.png`.
- Steps 1 (simulation) and 2 (scale) were skipped by config; the log says initial-state/scale files not found and defaults were used. The staircase is drawn at scale 1.0 as intended.
- Files left over from the earlier stairs_007 render remain in `results/` (git-ignored); harmless.

## Step 3.1 — Received the generated video (user-made, web UI)

- User placed the mp4 at `results/generation/videos_kling/SynStairsV1/kid_indoor1-syn-stairs_rand00001.mp4` — one folder too high (missing `stairs_wide3/`).
- **Action:** `mkdir` + `cp` (copy, original kept) to `results/generation/videos_kling/SynStairsV1/stairs_wide3/kid_indoor1-syn-stairs_rand00001.mp4`. Risk: none (git-ignored `results/`).
- **Inspection (read-only):** `ffprobe` for format, `ffmpeg` to tile 9 frames into `docs_learning/frames/contact_sheet.png`, which I viewed.
- **Read-only look at `grail/core/contact_label.py`:** all OpenAI calls go through `chat_with_image(...)` with JSON responses; a file-based "manual LLM" substitute is feasible. To be designed when recon step 4 is reached, and only if that step actually calls it for static terrain.

## Step 4.0 — Karolina preparation (no connection made by me)

- Asked what access exists. User: IT4I login works, project ID with `qgpu` allocation exists.
- User declined (a) listing the names in `.ssh_cluster/` and (b) a web search for storage docs; I respected both. I fetched only the IT4I Apptainer and job-submission pages (read-only, public).
- Wrote guide §10 using only verified commands; listed what remains unverified (storage layout, compute-node internet, GPU model, Apptainer module name).
- I do not connect to the cluster myself; the user runs the commands and pastes output.

## Step 4.1 — Karolina first-session output (pasted by user)

- Login node `login2.karolina`; `$HOME=/home/<cluster-user>` on shared NFS `/home` (28 TB total, 62% used — total filesystem, **not** the user's quota).
- Other mounts: `/scratch`, `/mnt/archive`, `/mnt/proj1..3`. Which one is writable for the user: unknown yet.
- Modules: `apptainer/1.1.5`, `apptainer/1.2.5 (D)`.
- `squeue --me`: job 5143438, `qgpu`, name `test`, node `acn20`, state CD (completed, 0:01) — an earlier test by the user, shows qgpu submission works.
- Open questions: quota, project directory, GPU model, compute-node internet.

## Step 4.2 — Karolina project facts (pasted by user)

- `id`: uid <cluster-user>, groups include `<project-id>`.
- `it4ifree`: project **<PROJECT-ID>**, resource Karolina GPU, **800 node-hours allocated, 24.51 used**, valid 2026-04-28 to **2026-10-27** (22 d left on 2026-10-05). => deadline-driven plan; do not burn GPU hours on stages that run fine on the Mac.
- `/mnt/proj1..3` and `/scratch` are root-owned, world-readable directories; project subfolders are named by project ID (e.g. `atr-26-1`). Likely path `/mnt/projN/<project-id>` — **inferred, not confirmed**.
- Part 2 (GPU session: GPU model, compute-node internet) not yet run by the user.

## Step 5.0 — Probe of the target GitHub repo (read-only on GitHub; local `.git` changed)

- `git ls-remote https://github.com/mg4637/grail.git`: repo exists and is **not empty**: `main` = `0ec544b` ("Initial commit"), plus branch `stage4-slow-finetune` (`fabafac`, commits about stage-3/4 retraining, "howto.md" audit).
- **Command run:** `git remote add mine https://github.com/mg4637/grail.git` then `git fetch mine` (local `.git` only: adds remote-tracking refs `mine/main`, `mine/stage4-slow-finetune`).
- **Finding:** `git merge-base main mine/main` is empty -> **unrelated histories**. A plain `git push mine main` would be rejected as non-fast-forward; `--force` would destroy the existing content. **No push has been made and none will be made without explicit permission.**
- Also read `scripts/setup/download_checkpoints.sh` (read-only): checkpoints come from Hugging Face `nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL` (GEM-SMPL, FoundationPose, SONIC) and `nvidia/GEM-X`, `nvidia/soma-x`; Hunyuan3D from GitHub/HF.

## Step 5.1 — Commit and first push attempts (user permission: new branch, patches only; docs withheld because the repo is public and the docs contain cluster identifiers)

- Repo `mg4637/grail` is **public** (anonymous `ls-remote` worked).
- `git checkout -b grail-stairs-pipeline`; staged only `configs/gen_2dhoi/terrain_stairs.yaml` and `grail/pipelines/blender/utils.py`; commit `a0fc0f6` (author = user's configured git identity; Co-Authored-By trailer for Claude).
- `git push mine grail-stairs-pipeline` **rejected twice** by the Git LFS pre-push hook: missing local LFS object `4567dc75…` (`assets/videos/deployment_stairs.gif`, an *old* version in history; the current version `9f5174c4…` is present). `git lfs fetch origin --include=...` did not obtain it (not available upstream).
- The hook uploaded ~2.0 GB of LFS objects to the user's GitHub LFS storage before failing (594 objects, 500 sent). **Possible LFS quota use on the user's account; to be checked by the user.**
- Refused to set `lfs.allowincompletepush true` (would publish a dangling LFS pointer). No branch was created on GitHub; `main` and `stage4-slow-finetune` untouched.

## Step 5.2 — Orphan snapshot pushed (permission: granted, "orphan branch, current files only")

- `git checkout --orphan grail-stairs-snapshot`; committed the tracked tree as one commit `ce62787` (no untracked files; `git diff --stat 8b9afa5 HEAD` = exactly the 2 patched files).
- `git push mine grail-stairs-snapshot` succeeded (no `--force`). Remote heads afterwards: `grail-stairs-snapshot` ce62787, `main` 0ec544b (unchanged), `stage4-slow-finetune` fabafac (unchanged).
- Excluded on purpose: `docs_learning/` (contains cluster identifiers; repo is public), `kernel-metadata.json`, `GRAIL.pdf`, `.idea/`, `imports/SONIC/gear_sonic/models`.
- GitHub warned about a 75 MB file `imports/SONIC/docs/source/_static/pico_setup/picoSetUp.mp4` (upstream content).
- Verified local branches (`git branch -vv`, no stashes): `main` = `8b9afa5` (clean upstream, tracks origin/main, does NOT contain the patches); `grail-stairs-pipeline` = `a0fc0f6` (patch commit on top of upstream, local only, never pushed); `grail-stairs-snapshot` = `ce62787` (pushed, currently checked out). The working tree on `grail-stairs-snapshot` contains the patches.


## Step 5.3 — Storage correction

- User created `/scratch/project/<project-id>/<cluster-user>/thesis` and asked whether to use `/mnt/proj2` instead.
- Read IT4I's Karolina storage page (public docs, WebFetch): HOME 25 GB/user; SCRATCH 20 TB/project, Lustre, 90-day auto-delete, mandatory for I/O-heavy jobs; PROJECT = long-term cross-cluster storage.
- **My earlier recommendation (`/mnt/proj2/...`) was wrong for working data**; it was inferred from directory permissions, not from documentation. Corrected guide §12 to `WORK=/scratch/project/<project-id>/$USER/thesis` and added §13.

## Step 5.4 — Karolina Step A/B run by the user (2026-10-11)

- Repo already present at `/scratch/project/<project-id>/<cluster-user>/thesis/grail` at `ce62787` (snapshot branch) -> clone fine; `git clone` refused because the folder existed.
- Shell shows `(base)` = a conda base env is active on the login node (not touched by us).
- `ml purge` left `XALT/3.0.2` loaded; the IT4I Apptainer page warns XALT can break container builds and suggests `ml -f purge`.
- `apptainer pull docker://nvgrail/grail:1.1.1` printed layer unpack + harmless xattr warnings (Lustre has no xattrs), then went quiet at 14:04-14:07. Judged normal (large layer unpacking); some blobs were already cached from an earlier attempt. Result pending: user will report `ls -lh grail.sif`, `df -h`.
- Time note: allocation ends 2026-10-27 (16 days left on 2026-10-11).

## Step 5.5 — Fix for lessons/02_wide_stairs.py (user hit `No module named 'grail'`)

- **Cause:** my own oversight. In Step 2.6 I needed `PYTHONPATH=.` to run the script but left the docstring saying to run it without. Python adds the *script's* folder, not the current directory, to `sys.path`.
- **Fix:** the script now inserts the repo root (`Path(__file__).resolve().parents[2]`) into `sys.path` and builds `OUT` from that root, so it works from any directory.
- **Verified:** ran from the GRAIL root and from `/tmp`; `data/syn_stairs/stairs_wide3/model.obj` md5 identical before and after (`3e6cd755…`), i.e. deterministic and the mesh used for the video is unchanged.

## Step 5.6 — Apptainer image verified (2026-10-11)

- User's wrong-shell rerun (no env vars, cwd in repo) was stopped with Ctrl-C by the user on my instruction; it had added ~2.2 GB to `~/.apptainer` (a cache, deleted by the user after a `pgrep` check showed no Apptainer process). `~` went 17 GB -> 15 GB. `~/.cache` (7.2 GB) is not ours and was left alone.
- Rerun with correct env: `FATAL: Image file already exists: "grail.sif"` => the FIRST pull had actually completed (image built 14:24, file written 14:31); it only looked frozen/crashed.
- Verified: `grail.sif` = 23 GB; `apptainer inspect` OK (Apptainer 1.2.5, amd64); `apptainer exec ... python3 --version` = Python 3.10.12. Warnings (no nv files, passwd/group) expected: no `--nv`, login node has no GPU.

## Step 5.7 — Requirements for recon read from the setup scripts (read-only)

- `scripts/setup/download_checkpoints.sh`: GEM-SMPL (~14 GB HF, `nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL`), GEM-SOMA (~6.4 GB), FoundationPose weights, Hunyuan3D, SONIC checkpoints. Flags: `--skip-gem-soma --skip-hunyuan3d --skip-sonic ...`.
- **Licensed SMPL / SMPL-X body models are NOT downloadable by script**; the user must register and download `SMPL_NEUTRAL.pkl` and `SMPLX_NEUTRAL.npz` and place them under `imports/GEM-SMPL/inputs/checkpoints/body_models/{smpl,smplx}/`.
- `docs/source/installation.md`: anonymous Hugging Face downloads are rate-limited; a read token via `huggingface-cli login` is the fallback; `HF_HOME` can relocate the cache (we will point it to scratch).

## Step 5.8 — Data transfer to Karolina (user ran the commands)

- First rsync attempts failed: `Permission denied (publickey)` because plain `user@karolina.it4i.cz` bypassed the user's `~/.ssh/config` alias `karolina` (key `id_ed25519_karolina`, passphrase-protected, `IdentitiesOnly yes`). Fix: use the `karolina:` alias. I had not opened `.ssh_cluster`; the user pasted the config block.
- Created on cluster: `thesis/grail/{results,data/syn_stairs}`, `thesis/staging/smplx`. **Staging** is used because `imports/GEM-SMPL` is an empty submodule dir; files placed inside would make `git submodule update --init` fail.
- Verified on cluster: `staging/smplx/SMPLX_NEUTRAL.npz` 104 MB; `grail/data/syn_stairs/stairs_wide3/{model.mtl,model.obj,texture.jpg}`; video `results/generation/videos_kling/SynStairsV1/stairs_wide3/kid_indoor1-syn-stairs_rand00001.mp4` 1.3 MB.
- Licence safety check: `models/` is git-ignored (0 tracked files); verified that the pushed branch does not contain MPI body models (matches under `data/g1_smplx`, `grail/models` are upstream GRAIL files).

## Step 5.9 — Which SMPL file to use (read-only investigation)

- `GRAIL/models/` contains: `smplx/` (SMPLX_NEUTRAL.npz 104 MB etc.), `SMPL_python_v.1.1.0/` (720 MB; `smpl/models/basicmodel_{f,m,neutral}_lbs_10_207_0_v1.1.0.pkl`) and `SMPL_NEUTRAL.npz` (43 MB, from the Julia download).
- `grep` of the code: `imports/GEM-SMPL/hmr4d/network/vimo/smpl.py:17` and `motiondiff/utils/config.py:12` load **`SMPL_NEUTRAL.pkl`** (pkl, not npz). `download_checkpoints.sh` and `installation.md` expect the same path.
- Decision: copy `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` as `SMPL_NEUTRAL.pkl` into `staging/smpl/` (rename by rsync destination; source untouched). The Julia `.npz` is not used. Rename convention from the `smplx` library; not yet verified by a run. Possible risk: v1.1.0 pickles may need `chumpy`.

## Step 5.10 — Karolina: submodules OK, container permission problem found (2026-10-11)

- Staging verified: `staging/smpl/SMPL_NEUTRAL.pkl` 236 MB, `staging/smplx/SMPLX_NEUTRAL.npz` 104 MB.
- `git submodule update --init imports/GEM-SMPL imports/FoundationPose imports/MoGe` (GIT_LFS_SKIP_SMUDGE=1): checked out at the pinned SHAs `77a0cc5f` (GEM-SMPL), `e3d597b8` (FoundationPose), `07444410` (MoGe). Small (5.2 MB / 8.1 MB / 133 KB).
- **Problem:** inside `grail.sif`, `/root` is `drwx------ root root`; `/root/miniconda3` (GRAIL's conda env, per README) is `Permission denied` for the non-root user Apptainer runs. System `python3` has no torch. => container cannot be used as the README's `docker run` (root) assumes. Candidate fixes to test: `apptainer exec --fakeroot`; or unpack to a sandbox as the user (files then user-owned) and run `--nv` from that. Neither verified yet.
- Plan: checkpoint download does not need the container (plain bash + `huggingface_hub`); use a venv in `$WORK` and `HF_HOME` on scratch.

## Step 5.11 — Container workaround tests (2026-10-11)

- Body models copied (cp) into `imports/GEM-SMPL/inputs/checkpoints/body_models/{smpl,smplx}/` on the cluster (236 MB / 104 MB); staging copies kept.
- Image contents: `/usr/local/cuda-12.8`, `/workspace/grail` (empty dir), `/opt/nvidia`; only Python env is `/root/miniconda3` (unreadable as user). The container inherits the host's conda env vars (user's miniforge at `/mnt/proj2/<project-id>/<cluster-user>/miniforge3`) — harmless.
- First fakeroot test used `| head -5` and was inconclusive (INFO lines filled the 5 lines). Rerun with `| tail -5`: **`FATAL: ... squashfuse not found`** => `apptainer exec --fakeroot grail.sif` is not usable on this cluster.
- Next: unpack to a sandbox folder as the user: `apptainer build --fix-perms --sandbox grail_sandbox grail.sif` (APPTAINER_TMPDIR on scratch). Whether `--fix-perms` + SIF source gives a readable `/root` is unverified; test = `apptainer exec grail_sandbox ls /root/miniconda3/envs`.

## Step 5.12 — Sandbox build from SIF failed; switching to docker source (2026-10-11)

- `apptainer build --fix-perms --sandbox grail_sandbox grail.sif` => `FATAL ... mount tmpfs ... broken pipe` (extraction mounts the SIF internally; same root cause family as the fakeroot `squashfuse not found` failure).
- Next try: build the sandbox from `docker://nvgrail/grail:1.1.1` (OCI layers are unpacked in userspace, which is how `apptainer pull` succeeded); reuses the layer cache in `$WORK/containers/cache` if `APPTAINER_CACHEDIR` is set.
- Open: checkpoint download (Part C) not yet confirmed started by the user.

## Step 5.13 — Sandbox build running; pip index problem; checkpoint download started (2026-10-11)

- `apptainer build --fix-perms --sandbox grail_sandbox docker://nvgrail/grail:1.1.1`: layers came from the cache (no re-download), unpack in progress (PID 276541; `tmp` 28 GB).
- `pip install huggingface_hub` in `$WORK/hfvenv` failed: pip tried **`pypi.ngc.nvidia.com`** (unresolvable on the cluster) — some pip config/env of the user's account sets an NVIDIA index (source not yet identified; not modified). Workaround: `pip install --isolated --index-url https://pypi.org/simple huggingface_hub` (`--isolated` ignores pip config files and `PIP_*` env). Installed huggingface_hub 2.2.0.
- `download_checkpoints.sh --skip-gem-soma --skip-hunyuan3d --skip-sonic` started in background with `HF_HOME=$WORK/hf_cache` (log `$WORK/download.log`). First files `[OK]` anonymously (HF warns about unauthenticated rate limits only).

## Step 5.14 — Checkpoint download finished (2026-10-11)

- `download.log`: 32 `[OK]`, 23 `[FAIL]`; `hf_cache` = 14 GB (matches the ~14 GB the install doc gives for GEM-SMPL).
- All 23 `[FAIL]` are in groups skipped on purpose (GEM-SOMA 8, Hunyuan3D RealESRGAN 1, SONIC 14): the script's final summary checks every file regardless of `--skip-*` flags. No FAIL for GEM-SMPL, FoundationPose or the body models in the pasted list. (Confirmation `ls` requested.)
- SONIC checkpoints still needed later for training.
- **My monitoring mistake:** I told the user to watch `$WORK/tmp` for the sandbox unpack; the real unpack dir is `$WORK/containers/build-temp-*/rootfs` (from the build log). Corrected.

## Step 6.1 — Found that recon uses the LLM for interaction start even for static stairs; answered it myself (2026-10-11)

- Read-only code reading: `hoi_optimizer._detect_contact_labels` **skips** VLM contact labelling for `is_static_obj` (comment: stairs). But `identify_interaction_start_end` (`grail/optimization/interaction.py`) runs first: with a static object no motion is found -> `detect_interaction(image, obj_name)` at 8 keyframes, scanning from the end, stopping at the first true.
- `loco_smplx.yaml` uses `body_model: g1_smplx`, whose `load_shape_params` uses shared G1 params, so the missing `data/RenderPeople_fitsmpl` is not needed (other body models return `None` there silently).
- Extracted keyframes 0,30,...,210 with ffmpeg into `docs_learning/frames/keyframes/`, viewed the tiled sheet, answered as the VLM -> `llm_answers.md` Entry 2 and `manual_llm/interaction_answers.json`.
- Open: how to feed the answers to the code without OpenAI (options: a `sitecustomize` shim outside the repo that fakes the OpenAI client; or a small tracked change in `openai_api.py`). Not implemented; will ask the user before touching repo code.
- Flagged risk: first question (frame 210) is already "true" => interaction start 210/240.

## Step 6.2 — Sandbox works; first GPU smoke test (2026-10-11)

- Sandbox build finished (`Build complete`, 55 GB, process gone). Tests: envs `grail`, `hunyuan`, `sonic` present; `grail` = Python 3.10.20, torch 2.7.1+cu128.
- Decided **not** to run `scripts/setup/install_env_docker.sh` as is: it runs `conda init bash` (would edit the user's real `~/.bashrc` because Apptainer shares `$HOME`) and pip-installs into a read-only image env (user's pip config also points at the unreachable NVIDIA index).
- GPU smoke test (`salloc -A <PROJECT-ID> -p qgpu --gpus=1 -t 00:15:00`, node acn06; `apptainer exec --nv --cleanenv --bind $WORK/grail:/workspace/grail ...`): torch sees A100-SXM4-40GB (cc 8.0, 42.4 GB), matmul OK; `import grail.pipelines.recon_4dhoi` OK; **nvdiffrast FAIL (CUDA error 209: prebuilt kernels not for sm_80)**; **FoundationPose mycpp FAIL (`mycpp.build` missing)**. Both are what the install script would rebuild.
- **Allocation observation:** `it4ifree` showed 389.64 / 800 node-hours used (was 24.51 on 2026-10-05). Our own use is minutes, so ~365 node-hours were used by other members of the shared project. ~410 left for all, valid to 2026-10-27. Our 15-min session did not change the displayed value (below 0.01 resolution) -> billing rounding still unknown.
- Plan: compile on the login node with `TORCH_CUDA_ARCH_LIST=8.0` (no GPU needed to compile), GPU only for a final check. Pending read-only diagnostics of the image (nvcc, how nvdiffrast is installed, cmake/ninja/pybind11).

## Step 6.3 — Native extensions built; GPU verification (2026-10-11)

- Diagnostics in the image: `nvdiffrast` 0.4.0 is a *precompiled* `_nvdiffrast_c...so` (explains CUDA 209 on sm_80); nvcc, cmake, ninja, pybind11 3.0.3 present; `mycpp` has sources but no `build/`.
- `build_ext.sh` (login node, CPU only, in background): cloned `NVlabs/nvdiffrast` to `$WORK/nvdiffrast_src`; `pip install --isolated --no-build-isolation --no-deps --target $WORK/pylibs` with `TORCH_CUDA_ARCH_LIST=8.0` -> new `_nvdiffrast_c...so` (3 MB) in `$WORK/pylibs` (image untouched; `PYTHONPATH` shadows the old copy). `mycpp` built with cmake/make into `imports/FoundationPose/mycpp/build` (printf format warning only).
- User's interactive `salloc ... exit` closed the SSH terminal (my block mixed login-shell and GPU-shell lines); replaced by a **batch job** (`sbatch smoke_job.sh`, log file) — terminal-proof.
- Result (job on acn06, 5 s): nvdiffrast loaded from `pylibs`, `nvdiffrast: OK`, `mycpp: OK True`, `import recon_4dhoi: OK`, torch sees A100.
- Pre-downloaded `Ruicheng/moge-2-vitl-normal` and `facebook/sam2-hiera-large` to `$HF_HOME` on scratch (hf_cache 17 GB) so the GPU job does not download. `sam2` imports; `moge` imports via an editable install at `/workspace/grail/imports/MoGe`, so the off-by-one path in `grail/adapters/depth.py` (`../../..`) is harmless in this setup. No repo change.

## Step 6.4 — Manual-LLM wrapper (no repo change)

- `docs_learning/manual_llm/run_recon_manual_llm.py`: replaces `grail.adapters.openai_api.chat_with_image` before importing `grail.core.contact_label` (import order makes the `from ... import` bind the replacement), then runs `grail.pipelines.recon_4dhoi` via `runpy`. Answers come from `interaction_answers.json` keyed by frame filename. Only the "physical contact with the object" question is answered; anything else raises and is written to `missing_requests.jsonl`.
- Local test (Mac venv): real `detect_interaction()` returned True/True/False/False for frames 210/90/60/0 as recorded; an unknown frame was refused with RuntimeError.

## Step 7.1 — First 4D reconstruction run on Karolina (2026-10-11)

- Wrote `recon_job.sh` (sbatch: `-A <PROJECT-ID> -p qgpu --gpus=1 -t 01:30:00`; `apptainer exec --nv --cleanenv --pwd /workspace/grail`, caches on scratch via `HF_HOME/TORCH_HOME/XDG_CACHE_HOME/TORCH_EXTENSIONS_DIR`, `PYTHONPATH=$WORK/pylibs:/workspace/grail`, `TORCH_CUDA_ARCH_LIST=8.0`, manual-LLM wrapper) and submitted job **5181637**. Started within ~1 min on acn06 (16:06:12).
- Correction of an earlier statement of mine: I had attributed the jump in allocation use (24.51 -> 389.64 node-hours) to "someone else"; `squeue --me` shows dozens of jobs under the user's own account (bowlK17, chairT_*, trolley*, mg_*), 8 running — most likely the cause. Not verified; retracted the claim.
- Result: `OK — 927s (15.4min)`, `exit code: 0`, `Videos: 1/1 valid | Frames: 240/240 valid`.
- Filter (Step 5), all `Valid`: initial human position error 0.0931 m (< 0.5), object-mask difference 0.0122 (< 0.3), mask aligned all frames, keypoint tracking 0.480 (< 2.0), contact 0.0 (< 5.0), min distance at initial frame 1.2327 (>= 0.02), average max velocity 0.0360 (>= 0.01), object motion static (range 0.0000 m).
- Log excerpt: `inter start idx: 210` (as predicted from my answers), `Applied offsets - human_offset [10.5, 8.8, 0.13072]`. Optimization stage 4 took 10 min 52 s (600 iterations of `init_opt`; keypoint_tracking ~1.44, depth_pointcloud 0.555 at the end).
- Tracing `inter_start_idx` (read-only): contact loss unaffected (no labels -> 0.0); `align_frame=inter_start_idx` in `human_model.transform_global_motion` anchors the global alignment at frame 210. Whether that distorts the world-frame motion is to be judged from the rendered outputs (pending: user copies `recon_result*.mp4` to the Mac).

## Step 7.2 — Reconstruction outputs copied to the Mac and inspected (2026-10-11)

- `rsync` from `karolina:.../grail/results/generation/4dhoi_recon_smplx_valid/` (18 entries, ~40 MB) and `recon_5181637.log` -> `docs_learning/recon_5181637.log`.
- Probed videos with `ffprobe`, tiled frames 0/60/110/150/190/230 with `ffmpeg` and viewed them. Visual result recorded in results.md. Judged consistent with the input video.

## Step 8.1 — Retargeting to G1 (GMR) on the Karolina login node, CPU only (2026-10-11)

- `git submodule update --init imports/GMR` (public, checked out at pinned `bb1bbe40`).
- Command (in the container, `sonic` env, no GPU, no `--nv`): `python -m grail.retargeting.retarget --data_dir results/generation/4dhoi_recon_smplx_valid --all --robot unitree_g1 --no_viewer --zero_out_wrist --output_dir data/motion_lib/stairs_wide3 | tee $WORK/retarget.log`. Followed GRAIL's documented *terrain* recipe (`docs/source/retargeting.md`): defaults for `--height_offset` (0.0) and `--mesh_scale` (1.0). The `0.6` in the `--mesh_scale` help is stale: `gen_terrain.py` states the scale is now baked into the generated stairs.
- My monitoring mistake: `| tee ... | tail -40` shows nothing until the end (tail buffers); told the user to `tail -f` the log from a second terminal.
- Result: retarget bar to 249/250 in 17 s; outputs `robot/*.pkl`, `objects/*.pkl`, `meta/*.pkl`, `object_usd/*.usd` + textures (`common/` empty).
- **Error seen:** `convert_mesh.py` (Isaac Lab `MeshConverter`) -> `RuntimeError: Accessed invalid null prim`. GRAIL documents this error in `retarget.py` and repairs it (`ensure_physics_schemas`). BUT the log also shows Isaac Sim started on a node with no GPU/driver (`Could not open libcuda.so`, `CUDA driver version is insufficient`, `libcuda.so.1: file too short`), so I suspected the error might instead be a side effect of running without a GPU. Did not assume; verified the output.
- Verification (`verify_usd.py`, `pxr`, read-only): defaultPrim `/model`; one mesh `/model/geometry/mesh` with 18 faces (3 boxes x 6 quads = matches the 36-triangle OBJ); `PhysicsRigidBodyAPI` + `PhysicsMassAPI` on `/model`; `PhysicsCollisionAPI` + `PhysicsMeshCollisionAPI` on the mesh. => repair step worked. Collision approximation / SDF settings not visible with this script (unverified).

## Step 8.2 — Retargeted motion copied to the Mac, inspected, replay script written (2026-10-11)

- `rsync` of `data/motion_lib/stairs_wide3/` from the cluster (robot pkl 59 KB, object_usd 6.5 KB + textures).
- Installed `joblib` into the lesson venv (tiny; needed to read the `joblib.dump(..., compress=True)` pkl files).
- Robot motion file structure: dict -> key `stairs_wide3_kid_indoor1-syn-stairs_rand00001` -> `root_trans_offset (240,3)`, `root_rot (240,4)` **wxyz** (verified in `retarget.py:903-911`, same as MuJoCo `qpos[3:7]`), `dof (240,29)` (29 hinge joints in MJCF order), `pose_aa (240,30,3)`, `smpl_joints (240,24,3)` all zeros, `fps 24`. `objects/*.pkl`: `root_pos (240,1,3)` first `[0,0,0.006]`, `root_quat` `[0.707,0,0,-0.707]` (-90 deg about z), scale 1; contact-point tables empty (static object). `meta/*.pkl`: `object_name`, `scene_scale 1.0`, `smpl_height 1.66`.
- Mistake: my first inspection script printed every empty `contact_points_*` entry (hundreds of lines). Should have limited the output.
- G1 model: `imports/GMR/assets/unitree_g1/g1_mocap_29dof.xml`: nq 36 (free joint + 29 hinges), nv 35. Its ground plane `floor` is at z=0.
- Geometry cross-check by hand: with the -90 deg turn the stairs run along world -y from y=0 to -2.1; top platform y in [-1.4,-0.7]; robot ends at (-0.355, -0.991) -> on the platform. Start at y=+1.21 (1.2 m before the first step).
- Foot check by forward kinematics (read-only, no physics). Method 1 (ankle z minus offset calibrated at frame 0) and method 2 (lowest point of the real foot geoms: mesh vertices + spheres): both show feet on the floor at start (within ~1-3 cm) and **feet ~3-6 cm above the top surface at the end** (lowest points 0.400 / 0.399 vs 0.366; uncertainty ~2 cm because the geometry already sits ~1.5 cm below z=0 at frame 0). My "planted foot" detector (ankle speed < 0.1 m/s) is too crude to judge each step; no conclusion drawn from it.
- Bugs in my own `lessons/03_view_retargeted_motion.py`, found by running it: (1) added a second geom named `floor` -> "repeated name"; the G1 XML already has one -> removed mine. (2) `import mujoco.viewer` inside `main()` made `mujoco` a local name -> `UnboundLocalError`; fixed with `from mujoco import viewer`.
- Rendered `docs_learning/frames/retargeted_g1.mp4` offscreen (MUJOCO_GL=glfw, 1280x720, 24 fps) and viewed 6 frames: standing -> walking -> foot onto first step -> climbing -> standing on top.

## Step 8.3 — User inspected the replay in MuJoCo: robot hovers (2026-10-11)

- **User report:** in the MuJoCo replay the robot floats during the whole process, makes no contact with the stairs, hovers as it approaches the stairs and while climbing.
- **My earlier statement was incomplete:** I had only compared foot height with the floor (z=0), found feet on the floor at the start, and compared the end with the top surface (3-6 cm hover). I had NOT compared each foot with the surface beneath it per frame. The user's observation was right; I should not have summarised the early check as "feet on the floor/stairs".
- New script `lessons/04_contact_gap_check.py` (kinematic, no physics): for each frame and each foot, `gap = (lowest point of the foot geometry) - (height of floor/step under that point)`. The stairs' surface height is rebuilt from the object pose (`root_pos` (0,0,0.006), `root_quat` -90 deg about z) and the known box dimensions (3 steps, rise 0.12, tread 0.35, width 1.5). Foot geometry = mesh vertices + spheres of `*_ankle_roll_link` and `*_toe_link`.
- **Result (see results.md):** a negative baseline of about -1.5 cm when standing on the floor (the foot geometry reaches ~1.5 cm below z=0, so true hover = gap + ~1.5 cm); during the climb the supporting foot is +3 to +7 cm above the surface under it; at the end +3.3 to +5.9 cm. In 150/240 frames neither foot is within 2 cm of a surface, in 74/240 not within 5 cm, in 0/240 not within 10 cm. Pelvis rises 0.76 -> 1.19 m (43 cm) while the stairs are 36.6 cm: about 6 cm more than the stairs.
- **Code reading (read-only):** `grail/retargeting/retarget.py::convert_gmr_data_to_motion_lib` documents `fix_height` ("make sure feet touch ground") but only does `root_trans_offset = root_pos.copy()`: **no ground-contact correction is implemented**. The only contact-aware logic is `_apply_downstairs_initial_height_correction` (first frame only, raises the robot if feet are below the terrain mesh; disabled by default; meant for downstairs).
- `hoi_data.pkl` content (limited inspection): `human_data` = SMPL-X params (`poses (240,165)`, `betas (1,10)`, `trans (240,3)`, hand poses, `foot_contact_probs (240,4)`, `predicted_body_height 1.624`, `mocap_frame_rate 30` although the video is 24 fps; retargeting uses `--fps 24`), `obj_data` (static pose, empty contact tables), `meta` (`inter_start_idx 210`, `inter_end_idx 240`), `eval_data`. No joint or vertex positions are stored, so I cannot compare the human feet with the stairs without running the SMPL-X model (needs torch + smplx; available in the container's `grail` env).
- **Untested idea (not a finding):** `retarget.py:537` scales human joints about `transl` (`(joints - transl) * scale + transl`); if that shrinks the legs but not the pelvis height, feet would float. Frame 0 shows feet near the floor, so this does not obviously explain the hover.
- **Open question:** is the hover already in the reconstructed human (reconstruction/body-model/scale) or created by the retargeting (GMR inverse kinematics has no ground constraint)? Decision on how to proceed pending with the user.

## Step 8.4 — Diagnosis plan chosen by the user: "Diagnose first" (2026-10-11)

- Read-only findings while preparing the diagnosis: in `retarget.py::load_hoi_sequence` the G1 path forces `scale = 1.0` (`if g1_smplx is not None: scale = 1.0`), so the joint-scaling formula at line 537 is the identity for our run. **My earlier "untested idea" (scaling about `transl` shrinks the legs) is refuted for this run** and has been removed from the candidate causes.
- Also noticed: the retargeting pads the sequence with `PAD_FRAMES = 10` ramp-up dummy frames at the start (interpolating from zero to the first frame's parameters). Not related to height; noted for timing.
- Wrote `lessons/05_human_vs_stairs.py` (CPU, `sonic` env, same path as the retargeting: `G1ProportionSMPLX.from_g1_smplx_dir` + `forward_smplx` on `hoi_data.pkl`'s SMPL-X parameters; foot vertices = `Foot` + `ToeBase` segments from `grail/models/smplx_vert_segmentation_contact.json`; stairs pose and floor from `obj_t`/`obj_R` in the same file). It measures the same per-frame foot-to-surface gap for the HUMAN mesh. `py_compile` passes; it cannot run on the Mac (needs torch + smplx), so the cluster run is its first real test.

## Step 8.5 — First cluster run of 05_human_vs_stairs.py failed on MY path bug (2026-10-11)

- `FileNotFoundError ... /scratch/project/<project-id>/results/generation/...`: the script derived the GRAIL root from `__file__` (`parents[2]`, valid only in `docs_learning/lessons/`). I had told the user to copy it to `.../thesis/`, so the root resolved two levels too high. Same class of bug as lesson 02 (Step 5.5): do not derive paths from `__file__` for scripts that get copied around.
- Everything else worked: container start, `torch`/`smplx`/GRAIL imports (`xrobotoolkit_sdk not found` is a harmless warning from GMR). Also the user's pasted terminal showed the earlier `verify_usd.py` run again (already recorded in Step 8.1).
- Fix: `ROOT = $GRAIL_ROOT or the current directory` (container runs with `--pwd /workspace/grail`) plus an assert that `ROOT/grail` exists. `py_compile` passes.

## Step 8.6 — Diagnosis result: the HUMAN reconstruction already hovers (2026-10-11)

- Second run of `lessons/05_human_vs_stairs.py` (login node, container `sonic` env, CPU, `tee` to `$WORK/human_vs_stairs.log`) succeeded. Stairs pose in the reconstruction frame: `t = (0, 0, 0.006)`, `R = Rz(-90 deg)` (identical to the robot frame, because the pipeline already centres the data on the object: `post_processing.center_at_object: true`).
- Human foot-to-surface gap (G1-proportioned SMPL-X, foot + toe vertices): frame 0 -0.2/-0.6 cm (grounded; confirms the method); frame 60 +3.9/+0.1; frame 100 +16.9/+7.0 (right foot over step 1); frame 120 +8.5/+8.5; frame 160 +11.0/+4.4; frame 180 +11.4/+7.7; frame 200 +5.5/+7.1; frame 239 +5.5/+5.1. Lower-foot gap over 240 frames: median +5.4 cm, mean +4.3, min -2.0, max +9.5. Neither foot within 2 cm: 166/240; within 5 cm: 144/240; within 10 cm: 0/240; a foot > 2 cm inside a surface: 5 frames. Pelvis z rises 0.753 -> 1.179 (0.426 m) for 0.360 m of stairs.
- **Conclusion (measured):** the hover is already in the reconstruction; retargeting did not create it (robot median +3.5 cm vs human +5.4 cm; the robot values are slightly smaller).
- **Why nothing corrects it (code reading, not a proven cause):** `loss_computer._human_foot_contact_loss` penalises the *velocity* of feet predicted to be in contact (anti-sliding), with no ground or step height; the object-contact loss is zero for static objects (no contact labels, `contact: 0.0`); `depth_pointcloud` gives depth cues but no foot-on-step constraint; vertical position is constrained by 2D keypoints, a strong prior to the initial estimate (`human_global_init_reg` weight 3000) and depth. `foot_contact_probs` (240x4) exist in `hoi_data.pkl` but are only used for the anti-sliding term.
- Filter thresholds did not catch it: the checks are initial position error (< 0.5 m), mask difference, keypoint tracking (< 2.0), contact (< 5.0, zero for static objects), initial-frame penetration (min distance >= 0.02) and average velocity. None measures per-frame foot-to-terrain distance.

## Step 9.1 — Chosen path: re-run reconstruction with a foot-to-terrain term (user decision, 2026-10-11)

**Design (reasoned from the code, before implementation):**
- Existing terms do not solve it: `ground` pulls the lowest vertex to a FIXED height (0.14 m, flat floor); `penetration` (SDF) only penalises feet inside the stairs; `human_foot_contact` only penalises foot velocity; object contact is zero for static objects.
- New loss `terrain_support`: height map of the static terrain mesh built once (rasterised triangles on a 1 cm XY grid, highest surface per cell, floor = lowest mesh vertex), in the object's own frame; each iteration the foot vertices (`L_Ankle`+`L_Toe`, `R_Ankle`+`R_Toe`) are transformed into that frame with the CURRENT optimised stairs pose (as in `_penetration_loss`), gap = foot vertex z - terrain height under it (min over vertices); loss = w * ( mean(lower-foot gap^2) + penetration_scale * mean(relu(-upper-foot gap)^2) ). The lower foot is treated as the supporting foot (assumption: when climbing stairs one foot is always on the ground); the other foot may lift but not penetrate. Heights are constants (no gradient through the lookup); the gradient acts on foot height.
- Initial weight 1000 (squared loss: initial value ~ 1000 * 4e-3 = 4, comparable to the other terms ~1); expected to need tuning. `human_global_init_reg` (3000) only constrains velocity magnitude, `keypoint_tracking` (2D) may resist vertical shifts: the video model may not draw feet at exactly the rendered stairs' height.

**Implementation (git branch `terrain-support-loss`, local; not pushed):**
- new `grail/optimization/terrain.py` (`TerrainHeightMap`, numpy build, torch lookup; no pytorch3d).
- `grail/optimization/loss_computer.py`: import, `_terrain_cache`, registry entry `"terrain_support"`, `terrain_foot_gaps()` and `_terrain_support_loss()`.
- new `configs/recon_4dhoi/loco_smplx_terrain.yaml` = `loco_smplx.yaml` + `terrain_support` (weight 1000, clearance 0, penetration_scale 1) in `init_opt`, and `output_dir: generation/4dhoi_recon_smplx_terrain` (validated output goes to `..._terrain_valid`, so the original results stay untouched). The original config is unchanged.

**Tests so far (Mac):** height map vs analytic stairs geometry: 3731 random points away from edges, max error 0.00 cm; spot checks floor/step1/step2/top/outside/down-side all exact. Correction of my own test comment: I expected edge errors up to 12 cm but saw 36 cm, which is the full-height side wall of the staircase (top 0.36 m vs floor 0), so the expectation was wrong, not the map; ambiguity is limited to a ~1 cm band at edges. `py_compile` OK for the 3 changed files. Next: numerical test of the real loss on the cluster (`lessons/06_test_terrain_loss.py`, grail env, CPU), then a GPU run of steps 4-6 only (`--skip_step1 --skip_step2 --skip_step3`).

## Step 9.2 — First cluster run of 06_test_terrain_loss.py failed (my design error) (2026-10-11)

- `ModuleNotFoundError: No module named 'mink'`: the test imported helpers from `grail.retargeting.retarget`, which imports `mink` (a GMR dependency) at module top; `mink` exists only in the `sonic` env, while the loss needs `pytorch3d` from the `grail` env. The two envs have different dependency sets, so one script cannot import code that belongs to the other.
- Fix: use `G1ProportionSMPLX._forward_smplx` (same computation as `retarget.forward_smplx`: same arguments, `return_full_pose=True`) and plain `pickle.load` (the file was written by the `grail` env itself; `_pickle_load_compat` only exists for numpy-version aliasing across envs).

## Step 9.3 — Terrain loss verified numerically on the cluster (CPU, grail env) (2026-10-11)

`lessons/06_test_terrain_loss.py` (second run, after fixing the `mink` import) — all three checks passed:
1. Real `LossComputer.terrain_foot_gaps` on the existing reconstruction: lower-foot gap median +5.4, mean +4.3, min -2.0, max +9.5 cm; frames 0/120/239 L,R = (-0.2,-0.6) / (8.5,8.5) / (5.5,5.1): **identical** to the independent measurement of lesson 05 (different code path: per-vertex numpy surface function vs the new height map + object-frame transform).
2. Loss at weight 1000 = 3.133; gradient of the loss w.r.t. vertex z summed over frames with hover > 2 cm = +88.725 (positive => the loss grows when the body moves up => the optimizer moves it down).
3. Shifting each frame down by its own lower-foot hover: median gap +0.00 cm, loss 0.0326 (was 3.133). The remaining 0.033 comes from the other foot dipping below the surface after this crude shift (penetration part), as designed.
- `lessons/05_human_vs_stairs.py` now takes an optional path argument so it can measure other `hoi_data.pkl` files (needed to evaluate the new run).

## Step 9.4 — Reconstruction with the terrain loss: run and measurement (2026-10-11)

- Job (`recon_terrain_job.sh`: `loco_smplx_terrain.yaml`, `--skip_step1 --skip_step2 --skip_step3`): `OK — 638s (10.6 min)`, `exit code: 0`, `Valid HOI result!`, 240/240 valid. Output in `results/generation/4dhoi_recon_smplx_terrain_valid/` (original results untouched). Log values at the end of `init_opt`: `terrain_support` 0.222 (weighted), `keypoint_tracking` 1.98 (was 1.44). Filter: initial position error 0.0967 m (was 0.0931), keypoint tracking 0.660 (was 0.480; limit 2.0), object-mask difference unchanged 0.0122, avg max velocity 0.0362.
- **My mistake, caught from the output:** the first measurement attempt printed numbers IDENTICAL to the original run: the cluster still had the OLD `05_human_vs_stairs.py` (no path argument; always reads the original file; the new `reading ...` line was missing). I also had the user run the `rsync` (Mac -> cluster) inside the cluster terminal (`Could not resolve hostname karolina`). Told the user not to interpret the identical numbers; repeated with the copy done from the Mac and checked for the `reading ...` line.
- **Measurement of the new reconstruction (human, same method as Step 8.6):**
  - lower-foot gap: median +1.7 cm (was +5.4), mean +1.2 (was +4.3), min -0.7 (was -2.0), max +2.8 (was +9.5);
  - neither foot within 2 cm: 50/240 (was 166); within 5 cm: 0/240 (was 144); a foot > 2 cm inside a surface: 0 (was 5);
  - pelvis z rise 0.390 m (was 0.426) for 0.36 m stairs; frame 0 unchanged (grounded: +0.2/-0.1 cm); frame 120: +3.0/+2.6 (was +8.5/+8.5); frame 160 right foot +0.8 (was +4.4).
  - the optimised stairs pose moved by 5 mm: `t_z` 0.006 -> 0.001 (the object pose is also an optimisation variable); object-mask difference unchanged.
- Remaining: ~+1.7 cm median hover (squared loss competing with keypoint tracking); a higher weight would trade more keypoint error for less hover. Not tried.
- `lessons/03` and `04` now accept a motion folder (`--motion_dir` / argv[1]).

## Step 10.0 — Questions from the user and the training plan (2026-10-11)

- User questions: (1) where is the new reconstruction video / how to download; (2) keep the first trial `.pkl` as an archive; (3) start a not-too-heavy training (~6000 iterations) on the retargeted motion, able to continue later.
- (1) `results/generation/4dhoi_recon_smplx_terrain_valid/SynStairsV1/stairs_wide3/kid_indoor1-syn-stairs_rand00001/result_vis/` on the cluster; `rsync` to the Mac into the same relative path (distinct from trial 1's `4dhoi_recon_smplx_valid/`, nothing is overwritten).
- (2) Trials are kept in separate folders (see the trials table in results.md). Caution recorded: SCRATCH deletes files not accessed for 90 days, so the cluster copies are not a permanent archive; copies on the Mac / project storage are.
- (3) Read-only study of `docs/source/tracking.md` and `imports/SONIC/gear_sonic/config/exp/manager/universal_token/scene/terrain_tracking.yaml`:
  - training entry point `imports/SONIC/train_agent_trl.py` (Hydra), config `+exp=manager/universal_token/scene/terrain_tracking`, algo `ppo_im_phc`, `num_envs` default 4096, `num_learning_iterations` default 30000, `callbacks.model_save.save_frequency: 500`, `flat_to_terrain_ratio: 0` (all envs terrain; `flat_placeholder.usd` only needed for flat envs).
  - data layout: dataset root with `robot/*.pkl` + `object_usd/*.usd` (1:1 stem match) + `objects/`; our retargeted folder has exactly that; Hydra overrides `++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot`, `...object_motion_file=${DATA_DIR}/objects`, `++manager_env.config.terrain_motion_dir=${DATA_DIR}`.
  - finetune pattern: `++resume=True ++checkpoint=models/terrain_stairs_warmstart/last.pt experiment_dir=<new dir>`; the checkpoint bundle comes from `download_checkpoints.sh` (SONIC group; not downloaded yet; we skipped it earlier). Continuing a run uses the same `++resume=True ++checkpoint=<run>/last.pt`.
  - observations/rewards (from the config): height map + object-state observations; rewards track 5 body points (torso, both wrists, both ankles); PPO.
- **Unverified risks:** (a) Isaac Sim inside the Apptainer sandbox as a non-root user (earlier retargeting on the login node showed read-only/`kit/data` errors and missing libcuda; needs a GPU node with `--nv`, writable Kit data/cache dirs, and redirected `$HOME` because Isaac writes multi-GB caches and the user's home quota is 25 GB); (b) iteration speed on one A100-40GB with few environments (unknown); (c) the robot-gap of the NEW retargeted motion is not yet measured (`04_contact_gap_check.py` on `stairs_wide3_terrain`).
- Plan: A) download SONIC checkpoints (login node); B) confirm the terrain retarget output and measure the robot gap; C) smoke test job (4 envs, 3 iterations, warm-start path) to measure speed and find environment problems; D) the real run (~6000 iterations, checkpoints every 500), continuing later with `++resume=True`.

## Step 10.1 — Smoke-test attempts for SONIC training (2026-10-11)

- Retarget of trial 2 done on the login node (17 s; `data/motion_lib/stairs_wide3_terrain/robot/stairs_wide3_kid_indoor1-syn-stairs_rand00001.pkl`). Warm-start bundle present (`models/terrain_stairs_warmstart/{last.pt 371 MB, config.yaml}`); SONIC download finished, `hf_cache` 28 GB (9 "missing" files = skipped groups).
- **Attempt 1 (job 5182070) failed in 1 s:** I had put `--writable` on the container. Apptainer: `By using --writable, Apptainer can't create /scratch destination automatically without overlay or underlay` (+ `nv files may not be bound with --writable`) -> `/scratch` bind failed. My error: should have tested on the login node first. Cost: ~1 s of GPU time.
- Diagnosis for a better fix: `--writable-tmpfs` is not allowed here (`Read-only file system`). Isaac Sim's retargeting log showed it can only not-write two places: `.../site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/user.config.json` and `.../isaacsim/kit/cache/DerivedDataCache`.
- Fix used: copy those two folders (168 KB and 630 MB) from the sandbox to `$WORK/kit_rw/{data,cache}` and bind them back over the originals (`--bind $WORK/kit_rw/data:$KITC/data --bind $WORK/kit_rw/cache:$KITC/cache`), plus `--home $WORK/fakehome` (Isaac writes caches into `$HOME`; the user's home quota is 25 GB) and the EULA environment variables (`ACCEPT_EULA=Y`, `OMNI_KIT_ACCEPT_EULA=YES`, `PRIVACY_CONSENT=Y`; accepted by the user's action of running the job).
- **Attempt 2 (job 5182073):** container started fine, A100 40 GB free, but `train_agent_trl.py: No such file` at `imports/SONIC/`. The file is `imports/SONIC/gear_sonic/train_agent_trl.py` (SONIC's own README/docs use that path). **GRAIL's `docs/source/tracking.md` has the wrong path** (`python train_agent_trl.py` after `cd imports/SONIC`; the same doc's eval command correctly uses `gear_sonic/eval_agent_trl.py`). I followed the doc without checking. Fix: `gear_sonic/train_agent_trl.py`.

## Step 10.2 — Smoke test attempt 3 (job 5182076): Isaac Sim starts, new blocker (2026-10-11)

- With `gear_sonic/train_agent_trl.py` + kit data/cache binds + `--home`: container OK on acn18 (A100 40 GB free), W&B offline, **`Warm-resuming (weights + optimizer) from models/terrain_stairs_warmstart/last.pt`**, Isaac Lab AppLauncher on `cuda:0`, **`[TerrainAutoDiscover] Found 1 terrain motions from .../stairs_wide3_terrain`**.
- Non-fatal-looking message: `[gpu.foundation.plugin] ... activeGpu index 0 is higher than the available GPUs ... No device could be created ... Graphics API: Vulkan` (Isaac Sim's rendering side cannot enumerate the GPU in the container; the script continued; headless physics may not need it; NOT yet confirmed harmless).
- Failure: `FileNotFoundError: No flat_placeholder.usd found at <terrain_motion_dir>/flat_placeholder.usd or .../object_usd/flat_placeholder.usd. Set flat_usd_path explicitly.` raised in `modular_tracking_env_cfg.py` (~line 711) even though `flat_to_terrain_ratio: 0`.
- Code reading: the placeholder is the USD for non-terrain ("flat") envs; with ratio R=0 the block size is 1 and `env_id % 1 == R` is true for every env, so every env gets a terrain USD and the flat template is only existence-checked. Workaround chosen (to be verified by the run): `++manager_env.config.flat_usd_path=$DATA/object_usd/<our stairs>.usd` (any valid USD).
- Observation (cause NOT investigated): the job log printed `exit code: 0` although the Python run ended with a traceback (`Error executing job ...`). The exit code cannot be trusted here; read the traceback / look for the expected outputs (same lesson as the Blender stage, Step 2.4).

## Step 10.3 — Smoke test attempts 4-5 and the real cause: Git LFS stubs on the cluster (2026-10-11)

- **Attempt 4 (job 5182077):** flat-placeholder workaround worked; run reached `constructing ManagerBasedRLEnv`; then the Vulkan errors (`No device could be created`) and `Failed to open layer .../configuration/pelvis.tmp.usd` + `[Fatal] ... NULL TfRefPtr<UsdStage>`.
- **Vulkan diagnostic (job 5182140, `vulkan_diag_job.sh`):** the GPU node HAS `/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json` (151 B) and the NVIDIA graphics libs; `--nv` bound the libs (`libGLX_nvidia`, `libnvidia-glcore`, `libnvidia-rtcore`, ...), but inside the container `/etc/vulkan/icd.d/nvidia_icd.json` was an empty 0-byte placeholder and `/usr/share/vulkan/icd.d` had no nvidia entry.
- **Attempt 5 (job 5182146):** bound the host ICD over the empty file (`--bind /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json:/etc/vulkan/icd.d/nvidia_icd.json --env VK_ICD_FILENAMES=...`). `[PerRankUSD] ... terrain_envs=4, unique_usd=1, terrain_motions=4` confirmed the clip is wired into all 4 envs. Vulkan error changed to `ERROR_INCOMPATIBLE_DRIVER ... Vulkan 1.1 is not supported`; same `pelvis.tmp.usd` failure. Conclusion: the Vulkan messages are NOT what stops the run (A100 has no RT cores; graphics errors are expected there; I could not fix them and did not need to).
- Kit logs in the redirected home (`fakehome/.nvidia-omniverse/logs/`) had nothing on the importer. Converter source (`urdf_converter.py`): the USD is built by `omni.kit.commands.execute("URDFParseAndImportFile"...)`; the robot URDF is `gear_sonic/data/assets/unitree_description/urdf/g1/main_nodex.urdf` (`robots/g1.py`, `ASSET_DIR = gear_sonic/data/assets`; `unitree_description` is a symlink to `robot_description`), referencing `package://unitree_description/meshes/g1/*.STL`.
- **Cause found:** `.gitattributes` marks `*.STL`, `*.usd`, `*.pt`, `*.pkl`, ... as Git LFS. The cluster repo was cloned with `GIT_LFS_SKIP_SMUDGE=1` (my instruction, Step 12 of the guide: I said it skipped "the large demo GIFs/videos, which we do not need"). **That was wrong**: it also left the G1 meshes as 130-byte LFS pointer stubs (`version https://git-lfs.github.com/spec/v1 / oid sha256:...`; cluster assets folder 652 KB vs 130 MB on the Mac). The importer cannot read a stub as a mesh -> no layer written -> NULL stage.
- Scope: 796 LFS files under `imports/SONIC/gear_sonic/` (`data/robots` 468, `data/assets` 105, `data/robot_model` 65, `trl/utils` 5, `data/human` 1; `gear_sonic/data` = 1.2 GB, of which `data/robots` 1.0 GB). Fix chosen: rsync the real files from the Mac (which has them): `data/assets` (130 MB), `data/robot_model` (51 MB), `data/human`, `trl/utils/smplx` (5 small `.pt`); `data/robots` left out until needed.
- Also learned: Isaac Sim downloads the URDF importer extension at run time from NVIDIA's CDN (needs internet on the compute node; stored under `fakehome/.local/share/ov`).

## Step 10.4 — Smoke test PASSED (job 5182164, 2026-10-11)

- After copying the real assets from the Mac (`gear_sonic/data/{assets,robot_model,human}`, `gear_sonic/trl/utils/smplx`, ~190 MB; cluster check: `pelvis_contour_link.STL` 1,805,184 B starting with `solid`, assets folder 130 MB), the 4-env / 3-iteration warm-start run completed: log ends with `[train_breadcrumb] main: after trainer.train`; summary box: total episodes 12, total timesteps 288 (= 3 iterations x 24 steps x 4 envs), iteration time 2.03 s (4 envs incl. start-up; not a speed estimate), total 8.48 s; checkpoints/logs in `$WORK/training/smoke`. Whole job 37 s. Vulkan errors still appear in the log and are harmless here (A100, headless physics).
- Working recipe (all required): `--nv --cleanenv --home $WORK/fakehome`; binds for `$WORK/kit_rw/{data,cache}` over `.../isaacsim/kit/{data,cache}`; host Vulkan ICD bound over `/etc/vulkan/icd.d/nvidia_icd.json` + `VK_ICD_FILENAMES` (kept, though not decisive); EULA env vars; `gear_sonic/train_agent_trl.py`; `++manager_env.config.flat_usd_path=<any valid USD>`; real (non-LFS) robot assets.
- My slips on the way (all in this step list): `--writable` (Step 10.1), the wrong training-script path copied from GRAIL's doc (10.1), `GIT_LFS_SKIP_SMUDGE` claim (10.3), and watching an old log instead of the new job's (user-side confusion caused by my `ls -t | head -1` recipe; now always use the job id).

## Step 10.5 — 6000-iteration training run submitted (2026-10-11)

- `sonic_train_job.sh` (variables `RUN`, `ITERS`, `ENVS`, `CKPT` with defaults run1 / 6000 / 1024 / `models/terrain_stairs_warmstart/last.pt`; `--time=05:00:00`; output `training/stairs_run1`, checkpoint every 500 iterations). Job **5182169** started on acn18 at 18:53; log confirmed `max_train_steps: 6000`, `terrain_envs=1024`, all envs on our clip. Iteration time / ETA / reward not yet seen.
- Continue later: `RUN=run2 ITERS=<n> CKPT=$WORK/training/stairs_run1/last.pt sbatch sonic_train_job.sh`.
- The job is independent of the SSH session (batch job); user told they can log out after seeing the ETA.

## Step 11 — Review of the session (written at the user's request; documentation only, nothing else changed)

**State of the code and data (not committed anywhere):**
- Local Mac branch `terrain-support-loss`: `grail/optimization/loss_computer.py` modified; `grail/optimization/terrain.py` and `configs/recon_4dhoi/loco_smplx_terrain.yaml` are new and UNTRACKED. Not committed, not pushed. The pushed GitHub branch `grail-stairs-snapshot` contains only the two early patches.
- Cluster repo (`$WORK/grail`): has the terrain-loss files (copied by rsync), the real `gear_sonic` assets copied over LFS stubs (git sees them as modified), and many untracked outputs. `data/robots` (1.0 GB of LFS files) still stubs.
- `docs_learning/` (incl. lesson scripts and manual-LLM wrapper) exists only on the Mac (by the user's choice: the GitHub repo is public).
- Recommendation for the user (not done): commit the branch locally so the loss code is versioned.

**Open checks that should have been done before training (my omission):** the robot foot-to-surface gap of trial 2 (`04_contact_gap_check.py data/motion_lib/stairs_wide3_terrain`) was never measured; training was started on it anyway. Also no visual check of the trial-2 reconstruction videos.

**Was the new `terrain_support` loss a good idea?** See guide §20. Short version: justified for THIS case (the cause was measured; the fix was targeted, isolated in a new config, unit-tested against an independent measurement, and reduced the hover from +5.4 to +1.7 cm median), but it is a heuristic that only holds for static terrain on a flat floor, it costs 2D fidelity (keypoint error 0.48 -> 0.66), it let the optimizer move the "static" stairs by 5 mm, only one weight was tried, and its benefit for the trained policy is unproven (no trial-1 vs trial-2 training comparison yet).

## Step 11.1 — User's two replay videos reviewed (2026-10-11, documentation only)

- Files in the GRAIL root: `1_hovering_feet.mp4` (17:10, 2,192,345 B = same size as my replay of the trial-1 robot motion) and `2_with_4dhoi_new_reward.mp4` (18:18): MuJoCo kinematic replays of the REFERENCE motions (trial 1 and trial 2), not training output. 1280x720, 24 fps, 240 frames each. No training video exists yet.
- Frames compared (6 each): same walk -> step -> climb -> stand sequence, no visible degradation; the 2-4 cm foot-height change is not visible at this camera distance. Only visible difference: shading patches on the stairs' blue base in video 2: viewer artefact (the optimiser lowered the stairs 6 mm -> 1 mm, so the base is nearly coplanar with the z=0 floor in my replay scene and z-fights). Not a data problem.
- User's question "did the other changes cause a bad result?": no visible evidence in the reference replays; possible small costs remain (keypoint error 0.48 -> 0.66, stairs moved 5 mm). Still outstanding: robot gap of trial 2 (`04_contact_gap_check.py data/motion_lib/stairs_wide3_terrain`) and the trial-1 vs trial-2 training comparison.

## Step 10.6 — First look at training run 1 (job 5182169, iterations 66-71, 2026-10-11)

- Speed (1024 envs, A100): ~2,000-2,600 steps/s; iteration 9-12 s (collection ~7-9 s = simulation, learning ~2-2.9 s); total time 488 s at iteration 71. The printed ETA (~39,000-40,800 s ~ 11 h) is total_time/iterations x remaining; recent iterations (~11 s) imply even longer. The job limit is 5 h = 18,000 s, so 6000 iterations will NOT fit: expect to be stopped by the time limit at roughly iteration 1,600 (checkpoints every 500 -> 500/1000/1500 would exist).
- Quality already at iteration ~70 (warm start from the pretrained stairs policy): 99.9% of episodes end by `time_out` (anchor_ori_full 0.1%, no anchor_pos/ee_body_pos/foot_pos terminations); mean episode length ~230-275 steps; `error_body_pos` ~0.022 m, `error_anchor_pos` ~0.036-0.043 m, `error_joint_pos` ~0.056 rad, `error_anchor_rot` ~0.065 rad; velocity errors higher (`error_joint_vel` ~0.6, `error_body_ang_vel` ~0.93). Reward terms fluctuate (mean rewards 28-34) without a trend yet.
- Interpretation (not verified): the pretrained policy already tracks our stairs clip; long training may add little. Episode length limit (~250 steps?) and whether episodes cover the whole 10 s clip not checked. No evaluation of the trial-1 vs trial-2 difference exists.

## Step 10.7 — Run 1 cancelled; run 2 started; MuJoCo view of the trained policy planned (2026-10-11)

- User cancelled run 1 (job 5182169) at iteration 89 after my suggestion to let it reach the 1000-iteration checkpoint. **My statement "nothing saved is lost" was wrong:** the first scheduled checkpoint is at iteration 500, so no `model_step_*.pt` existed; `last.pt` (371 MB, 19:00) was written ~7 min after start and its content is unknown. Cost: ~14 min of one GPU.
- Run 2 (job 5182187): `RUN=run2 ENVS=512 ITERS=1000 sbatch --time=02:30:00 sonic_train_job.sh`, checkpoint every 100 iterations via `++callbacks.model_save.save_frequency=${SAVE:-100}` (key from `terrain_tracking.yaml`; sed-edited into `sonic_train_job.sh`). Output `training/stairs_run2`. Iteration time at 512 envs not yet seen (my estimate 5-6 s, unverified). User again typed the literal `<JOBID>` placeholder; I should write real numbers.
- Goal: see the trained policy in MuJoCo. GRAIL's rendering path needs graphics (fails on the A100) so a rollout export is used instead.
- Findings (read-only): SONIC has `TrajectoryRecorderTerm` (`gear_sonic/envs/manager_env/mdp/recorders.py`): per env it records `dof_pos` (Isaac Lab joint order), `root_pos_w` (relative to env origin; taken from `motion_cmd.robot_body_pos_w[:,0]` when available), `root_quat_w` (wxyz), `fps = 1/(step_dt * render_frame_skip)` (skip default 2 -> 25 fps at 50 Hz), written to `NNNNNN.trajectory.pkl` + `scene_metadata.json` in `close_writers()`. `ManagerEnvWrapper.end_render_results()` closes a recorder named `trajectory`; `eval_agent_trl.py` calls it when `max_render_steps` is reached, before `os._exit(0)` (which would skip destructors). Cameras are only created if `render_results`/`enable_cameras` are set, so a trajectory-only run avoids the graphics problem (to be verified).
- Joint order: SONIC converts MuJoCo-order motions to Isaac order with `G1_MUJOCO_TO_ISAACLAB_DOF`; the inverse list `G1_ISAACLAB_TO_MUJOCO_DOF` (in `robots/g1.py`, also `trl/utils/order_converter.py`) is used by `lessons/03_view_retargeted_motion.py --trajectory`. Our retargeted GMR `dof` order is fed to SONIC as MuJoCo order, so the viewer's order matches. Test (Mac): a fake rollout built from the reference (reference -> Isaac order -> loader) round-trips to the identical joint angles (`True`). Not yet verified: that the recorder's order really equals SONIC's Isaac order (to be checked numerically against the reference on a real rollout).
- New: `lessons/03_view_retargeted_motion.py --trajectory <file-or-folder> [--env N]`; cluster job script `sonic_export_job.sh` (variables `CKPT`, `OUT`, `ENVS`, `STEPS`; `eval_agent_trl.py +checkpoint=... ++max_render_steps=... ++manager_env.recorders.trajectory._target_=...TrajectoryRecorderCfg ++manager_env.recorders.trajectory.save_path=...`). First test planned on `training/smoke/last.pt` (3 iterations; pipeline test only).

## Step 12.0 — Diagnosis protocol for the early termination (2026-10-12)

- **User request:** test and diagnose; change ONE thing at a time; document changes and results; fix the early termination; show the FULL clip in MuJoCo.
- State: run 2 finished all 1000 iterations (checkpoints every 100 iterations: `model_step_000100.pt` 19:27 ... `model_step_001000.pt` 20:33, plus `last.pt`; iteration speed improved later in the run: 4-12 min per 100 iterations).
- Observation (E0, recorded earlier): rollout of the PRETRAINED warm-start policy on our clip: joint order verified (0.108 rad vs 0.358 without the conversion), start pose matches the reference, but every environment is reset at frame 130-140 (5.2-5.6 s) with a 1.26 m teleport back to the start; pelvis height tops out at 0.87 m (reference 1.15 m); upright measure >= 0.93 (no fall). Termination REASON unknown.
- Reasoning from code/logs: `tracking_time_out` ends an episode when `time_steps + motion_start_time_steps + 1 >= clip length (steps)`; training stats (mean episode length ~245 steps, 99.9% time_out, random start phases) are consistent with a 500-step (10 s) clip, so the eval episodes ending at ~5.4 s from frame 0 are most likely NOT time-outs but failure terminations (rules in `base_adaptive_strict_ori_foot_xyz`: `anchor_pos`, `anchor_ori_full`, `ee_body_pos`, `foot_pos_xyz`). Unverified.
- **Passive instrumentation (changes no behaviour):** `docs_learning/cluster/term_logger.py`: `TrajectoryTermLoggerTerm`, a subclass of SONIC's `TrajectoryRecorderTerm` registered under the existing recorder key `trajectory` (`RecordersCfg` has fixed fields, so a new key would be rejected; this way `end_render_results()` still closes it). After each env step it appends `{"step": N, "fired": {rule: [env ids]}}` to `terminations.jsonl` using `env.termination_manager.active_terms` / `get_term(name)` (Isaac Lab API from memory; wrapped in try/except, logs `logger_error` if the API differs). `docs_learning/cluster/sonic_export_job.sh` (v2): adds `PYTHONPATH=$WORK/extra_py` and the logger, plus an `EXTRA` variable for exactly ONE experiment override per run. Both copied to the cluster with rsync (cluster script overwrites the older hand-typed `sonic_export_job.sh`). Syntax checks pass; not yet run.
- Experiment plan (each row = one change relative to the previous state): E1 = run-2 final checkpoint `model_step_001000.pt`, no other change (also logs the termination reason). E2.. = depends on E1 (e.g. disable/loosen the single rule that fired, in evaluation only). A termination rule is a training signal; relaxing it in evaluation only reveals what the policy does afterwards, it does not by itself make the policy better.

## Step 12.1 — Experiment E1 result (2026-10-12)

- Mistakes of this round (mine / process): (a) first E1 submission (job 5188067) ran the OLD export script (log line lacked `EXTRA=''`; no logger) because the Mac->cluster `rsync` of the new files failed with `Permission denied (publickey)` and no passphrase prompt (cause not yet known; `~/.ssh` not touched); (b) job 5188070 was submitted anyway although my own check (`grep -c term_logger` = 0) showed the script was still old; both cancelled (`CA`); (c) workaround: the two files were created directly in the cluster terminal with heredocs (`extra_py/term_logger.py`, `sonic_export_job.sh` v2); `grep -c term_logger` = 2 as expected.
- **E1 = job 5188083** (new script, `EXTRA=''`, checkpoint `training/stairs_run2/model_step_001000.pt`, 4 envs, 500 steps): log `Loaded 1 motions with a total length of 9.940s and 498 frames`, `Reached max_render_steps=500`, trajectories saved (250 frames each = 10 s at 25 fps), `terminations.jsonl`: **`{"step": 497, "fired": {"time_out": [0, 1, 2, 3]}}`** (the only line).
- Interpretation: with the run-2 checkpoint no failure termination occurs; all environments run the entire clip until its natural end (step 497 of 498). The early resets seen with the pretrained warm start (E0, frame 130-140) are therefore a property of that un-finetuned policy on our clip, removed by 1000 iterations of finetuning (run 2, single change between E0 and E1: the checkpoint). Cause of E0's termination was never logged (logger did not exist) and remains unknown.
- Not yet verified: how well run 2 tracks the clip and whether the feet touch the stairs under physics; viewing the full clip in MuJoCo (needs the rollout files on the Mac; the Mac SSH key problem blocks `rsync`). Prepared `docs_learning/lessons/08_rollout_report.py` for the numerical check.


## Step 12.2 — E1 verified numerically; SSH access repaired; full clip viewed (2026-10-12)

- SSH problem (new connections refused with `Permission denied (publickey)` and no passphrase prompt, while an already-open VS Code remote session still worked): `ssh -v` showed the Mac offered `~/.ssh/id_ed25519_karolina` (fingerprint matched its `.pub` and its private key, so no stale-pub mismatch); the Mac's `~/.ssh/config` block for `karolina` was correct; on the cluster `~/.ssh/authorized_keys` (modified Jul 1, correct permissions) listed only the cluster's internal key and an RSA key from login4, NOT the Mac key. Why it worked on Oct 11 was not established (hypotheses: key managed through another IT4I mechanism that lapsed; unverified). Fix applied by the user: appended the Mac's PUBLIC key line to the cluster's `authorized_keys` from the still-open VS Code session. Lesson: an already-established SSH connection survives key removal; do not close it before fixing access.
- E1 rollout copied to the Mac (`rollouts/e1_run2_1000`); `08_rollout_report.py` report recorded in results.md (mean root distance 6.5-6.9 cm, lower-foot gap median -0.1 cm / min -0.5 cm, 0 frames > 2 cm inside a surface). User viewed the full clip in MuJoCo: very good.
- Process note: the experiment protocol (one behaviour change per run, passive instrumentation first) found that the early termination was a property of the un-finetuned policy; no behaviour-changing experiment (E2...) was needed.

## Step 13 — User noticed: no contacts with the stairs and no forces in the MuJoCo replay (2026-10-12; chair task put on hold)

- **Observation (user):** in `03_view_retargeted_motion.py --trajectory rollouts/e1_run2_1000`, pressing `c` showed robot-floor contacts but none between robot and stairs; pressing `f` showed no forces.
- **Cause 1 (my choice, not explained to the user at the time):** `build_model()` created the stair boxes with `contype=0, conaffinity=0`, so they can never collide with anything. I did it to keep a pure replay free of contacts and never said it would hide stair contacts.
- **First test result (mistaken reading):** flipping `geom_contype/conaffinity` of the stairs in memory after compiling still gave 0 contacts at frames 150/200. Reason found with `mj_geomDistance` (which ignores collision bits): the foot collision spheres DO penetrate the steps by about 4.7 mm at frames 125/175/200/225 (and the floor by 0.3-1.3 mm at frame 0), so contacts should exist. MuJoCo pre-computes collision bits per BODY at compile time (`body_contype/body_conaffinity`) and skips body pairs with them; changing geom bits afterwards has no effect. Robot collision shapes: 36 geoms with contype=conaffinity=1 (feet = 4 spheres per ankle link, plus mesh/cylinder geoms for other links).
- **Fix (one change):** in `build_model()` the stair boxes now use `contype=1, conaffinity=1` at build time. Result: frame 0: 2 floor contacts; frames 100/125/150: 2 stair contacts; 175: 4; 200: 6; 225: 5. `c` now shows robot-stairs contacts.
- **Cause 2 (forces):** the replay is kinematic (`qpos` written directly, `mj_forward`, no actuators, no velocity, no time stepping). MuJoCo only reports the constraint force that resolves the ~5 mm overlap at that instant: summed normal force 24 N (floor frame 0), 5 N (frames 100-150), 105-153 N (frames 175-225) vs robot weight 337 N (mass 34.4 kg in the MJCF). These are not ground-reaction forces. Before the fix there were no stair contacts at all, hence nothing to draw; floor forces (15-24 N) were tiny arrows.
- **Real forces exist in the simulation:** `modular_tracking_env_cfg.py` defines `contact_forces` (ContactSensor on all robot links, `net_forces_w`, used by the feet rewards via `find_bodies(".*ankle_roll.*")`) and `feet_terrain_contact_sensor` (sensed body = the terrain Object, filter = both ankle links; `force_matrix_w` `[envs,1,2,3]` = force on the stairs from each foot). They were not recorded.
- **Passive instrumentation v2 (no behaviour change):** `docs_learning/cluster/term_logger.py` now also writes `forces.npz` (`foot_net_force_w (T,envs,2,3)`, `foot_on_terrain_w (T,envs,2,3)` if the sensor exists, `foot_body_names`, `robot_mass_kg` from `root_physx_view.get_masses()`, `fps`), recorded on exactly the steps where a trajectory frame is recorded. Errors are written as `logger_error` lines to `terminations.jsonl` instead of breaking the run. Sensor/API names are from reading the SONIC source; NOT yet run (next: experiment E1b = same checkpoint and settings as E1, only the extra recordings).
- **Viewer/report additions:** `03_view_retargeted_motion.py --trajectory <folder> --forces [--force_scale 0.0015]` draws one arrow per foot (green = left, orange = right) from the ankle link, length proportional to the recorded force (contact position was not recorded, so the arrow start is approximate). Test of the drawing code with a stand-in scene and made-up forces: 2 arrows at 170/160 N, length 0.255 m, 0 arrows at zero force. `lessons/09_forces_report.py <folder>` prints, per env, the summed vertical foot force in four phases vs the weight, the number of airborne frames, and the feet->stairs sensor statistics.
- **Mistake note:** the macOS `sed -i` call that was supposed to change the stairs flags failed silently the first time (needs `-i ''`); I redid the edit with Python and re-ran the measurement.

## Step 14 — Forces verified; status of the stairs pipeline (2026-10-12)

- E1b (job 5188210; v2 logger copied from the Mac with rsync after SSH was repaired) produced `forces.npz`; `09_forces_report.py` on the Mac: standing force 97.8-100.9 % (floor) and 99.6-100.8 % (top) of the PhysX weight 327 N; 0 airborne frames; feet->stairs sensor active in 157/250 frames, peak ~455 N (1.4 x weight) at frame 162. Physics check passed (details in results.md).
- User question: are `e1_run2_1000` / `e1b_forces` outputs after RL, and is the GRAIL stairs pipeline finished? Answer given: yes, they are rollouts of the trained (finetuned) policy; the pipeline ran end to end on one clip with the substitutions and caveats listed in results.md ("Status of the GRAIL stair-climbing pipeline"). User then asked to continue with: finish docs, sanitized GitHub commit, chair task.

## Step 15 — Sanitized GitHub commit prepared (not pushed) (2026-10-12)

- User instruction: document, commit to GitHub, "don't share anything secret".
- `mg4637/grail` is PUBLIC, so every commit is world-readable. Approach: real code goes in as-is (it has no private content); notes/lessons/cluster scripts go into a new folder `thesis_notes/` produced by a LOCAL script `docs_learning/make_public_notes.py` (not committed: it holds the real identifiers it removes). It whitelists files, replaces cluster user name, project id (both cases), Kaggle name, Mac user name and paths, emails, git user name, IP addresses, SSH fingerprints/key comment with placeholders, and then scans the result (leftover identifiers, credential patterns such as private-key blocks, SSH public key lines, HF/GitHub/OpenAI tokens, API-key and password assignments, IPv4 strings, files > 1.5 MB).
- The scanner was tested with planted fakes (private key header, ssh public key line, `hf_`, `ghp_`, `sk-`, API-key export, passphrase assignment): all caught; sanitizer test line with real identifiers: all replaced, no leftovers.
- Not committed on purpose: `docs_learning/` (originals), `.venv`, videos/images, logs, `rollouts/`, `kernel-metadata.json` (Kaggle config), `.idea`, `GRAIL.pdf`, `run_clips_easy_access.txt` (personal viewer shortcuts), `uv.lock` (3-line file created by `uv`, not ours), SMPL/SMPL-X models and all checkpoints (licensed / large; `models/` is git-ignored).
- `objects_wide3.yaml` excluded from the public notes (3000-line copy of upstream's `terrain_syn_stairs.yaml` plus one entry; described in the README instead).
- Cluster job scripts that existed only on the cluster were saved into `docs_learning/cluster/` (reconstructed from the exact commands run): `build_ext.sh`, `recon_job.sh`, `recon_terrain_job.sh`, `sonic_train_job.sh`, plus `sonic_export_job.sh` and `term_logger.py`.
- Local commit `796b7fb` on branch `terrain-support-loss`: 25 files (code: `loss_computer.py` +41 lines, new `terrain.py`, new `loco_smplx_terrain.yaml`; 22 files under `thesis_notes/`), author identity = the machine's configured git identity (the same one already visible on the earlier pushed branch). A final check of the staged diff for identifiers/secrets/IPs found nothing.
- Observations about the repo root: `run_clips_easy_access.txt` (two viewer commands) and `uv.lock` appeared as untracked files; neither committed.
- Chair-task groundwork (read-only): the renderer supports several fixed objects per scene (`static_objects` in the scene config: `name` = file in `data/Scene` (.obj/.glb), `pos`, `rot`, `scale`, optional `opacity`; the pickup scene uses a table at opacity 0.5); the dynamic object comes from an object config; support for non-table fixed objects in recon, retargeting and the SONIC training environment (one `object` and one `table` entity seen) is NOT yet verified.
