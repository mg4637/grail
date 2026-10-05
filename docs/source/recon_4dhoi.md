# 4D HOI Reconstruction

Recovers the full 4D human-object interaction (SMPL-X body pose + MANO hand
pose + 6-DoF object trajectory) from a generated or captured RGB video.

## Quickstart

```bash
# Manipulation/pickup — SMPL-X (default, expects object to move)
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill \
    --results_dir results

# Manipulation/pickup — SOMA body model
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill \
    --results_dir results --config configs/recon_4dhoi/manip_soma.yaml

# Terrain / sitting (static object — bypasses FoundationPose)
python -m grail.pipelines.recon_4dhoi --dataset syn_stairs --results_dir results \
    --config configs/recon_4dhoi/loco_smplx.yaml
```

Validated outputs land under
`results/generation/4dhoi_recon_smplx_valid/{dataset}/{category}/{video_id}/`:

- `hoi_data/hoi_data.pkl` — body params + object 6-DoF poses per frame
- `result_vis/recon_result.mp4` — overlaid reconstruction on the input
- `result_vis/recon_comparison.mp4` — side-by-side input vs. recon
- `result_vis/recon_result_top_view.mp4` — top-down view
- `result_vis/recon_result.html` — interactive ScenePic viewer
- `mesh_data/` — the canonical object mesh used in optimization

## Pipeline steps

```{list-table}
:widths: 5 25 70
:header-rows: 1

* - #
  - Stage
  - Notes
* - 1
  - Human pose
  - GEM-SMPL body + WiLoR hands, fused per-frame. ~45 s/video on an L40S.
* - 2
  - Preprocess
  - SAM2 mask tracking + MoGe monocular depth. ~36 s/video.
* - 3
  - Object pose
  - FoundationPose 6-DoF tracking from cached masks + RGB. ~40 s/video.
* - 4
  - HOI optimization
  - Multi-stage; uses OpenAI vision calls inside `grail/core/contact_label.py`
    to detect contact joints per interval. Defaults use `gpt-4o`.
    Heaviest stage (~9-10 min/video on L40S).
* - 5
  - Filter
  - Quality thresholds: human-position error, mask alignment, keypoint
    tracking, contact penalty, penetration, motion magnitude.
* - 6
  - Visualize
  - PyTorch3D top-down + side-by-side renders, ScenePic HTML.
```

## Required environment

`OPENAI_API_KEY` is used by the OpenAI API for contact-joint detection in step 4.
The default vision model is `gpt-4o`.

## Common variants

```bash
# Single video by ID
python -m grail.pipelines.recon_4dhoi --video_id ComAsset/cordless_drill/<video_name> \
    --results_dir results

# Skip already-finished videos
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill \
    --results_dir results --skip_done

# Step 4+ only (after rerun of contact detection)
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill \
    --results_dir results --skip_step1 --skip_step2 --skip_step3

# Static-object mode (no global object motion expected)
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill \
    --results_dir results --is_static_obj
```

## Configs

Configs are split by **task** (manipulation vs. locomotion/terrain) × **body model** (SMPL-X vs. SOMA):

```{list-table}
:widths: 35 65
:header-rows: 1

* - File
  - Purpose
* - `configs/recon_4dhoi/manip_smplx.yaml`
  - Manipulation / pickup, SMPL-X (G1) body. `filter_object_motion: dynamic_only` drops static-object recons at step 5; FoundationPose runs (object expected to move). Default for `grail.pipelines.recon_4dhoi`.
* - `configs/recon_4dhoi/manip_soma.yaml`
  - Manipulation / pickup, SOMA body. Same params as `manip_smplx.yaml` — only `body_model` + paths differ.
* - `configs/recon_4dhoi/loco_smplx.yaml`
  - Locomotion / terrain / sitting, SMPL-X (G1) body. `is_static_obj: true` bypasses FoundationPose (terrain doesn't move); `filter_object_motion: static_only` keeps only static-object recons.
* - `configs/recon_4dhoi/loco_soma.yaml`
  - Locomotion / terrain / sitting, SOMA body. Same params as `loco_smplx.yaml`.
```

SOMA variants share **all** optimization params with their SMPL-X counterparts — only `body_model` + `hmr_dir` + `output_dir` differ.

## Sharded fan-out

```bash
# Run one shard per worker in your scheduler.
python -m grail.pipelines.recon_4dhoi \
    --dataset ComAsset \
    --results_dir results \
    --job_chunk_idx <i> \
    --num_job_chunks <N>
```

A typical 8-chunk run covers ~24 videos in ~35 minutes wall-clock (parallel).
