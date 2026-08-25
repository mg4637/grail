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
| this session, 6000 iters | `grail_table_v3f` | **0.604** | **92.2** | `anchor_pos` |
| untruncated (video only) | `grail_table_v3f` | 1.000 *(artifact)* | 485.8 | — |
| in flight | `grail_table_v3` | — | — | — |

Training on `v3f`: mean reward 3.14 → 5.83, but `error_body_pos` moved only 89 → 87 mm. The
reward gain came from survival time, not tracking accuracy.

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

### Open at end of session

- Training on `grail_table_v3` running; eval armed to fire automatically on exit.
- **Decision rule, committed in advance:** if `v3` does not clear 0.644, the conclusion is
  that the Stage 3 fixes were necessary but the kid's climbing strategy does not transfer to
  a fingerless robot. Next move is a different source clip or contact-implicit trajectory
  optimisation — not a fourth geometric edit.
- The chair clip is a compound carry-then-climb that no single released policy covers.

### Ops notes

`docker exec -d` with the script doing its own `exec >> logfile 2>&1`; never wrap container
runs in `timeout` (SIGTERM deadlocks Isaac Sim); one Isaac job at a time on the single GPU;
poll with `[t]rain_agent_trl`. MuJoCo rendering on the host needs `MUJOCO_GL=glfw DISPLAY=:3`
(EGL is broken — PyOpenGL pinned at 3.1.0; osmesa not installed). Container-created files are
root-owned; `chmod -R a+rX` inside the container before reading them from the host.
