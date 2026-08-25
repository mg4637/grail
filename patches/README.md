# Vendored-submodule patches

`imports/GMR` is a submodule of a third-party repo (YanjieZe/GMR) that we cannot
push to, so fixes made inside it cannot be recorded as a submodule commit — the
parent repo would end up with a gitlink to a commit nobody else can fetch. They
are kept here as patches instead.

Apply with:

    cd imports/GMR && git apply ../../patches/gmr-stage3-fixes.patch

## gmr-stage3-fixes.patch

Two Stage 3 retargeting bugs, both of which corrupted the table-climb reference:

1. `assets/unitree_g1/g1_mocap_29dof.xml` — 15 joint ranges were truncated
   relative to the URDF. Hip pitch declared `range="-1.57 1.57"` against a real
   limit of `-2.5307 2.8798`; `mink.ConfigurationLimit` (gain 0.95, up to 11
   solves) pinned it at exactly -1.5700 for frames 72–95, which is the entire
   mount. The sibling `g1_mocap_29dof_with_hands.xml` already had the correct
   values. Hip pitch/roll/yaw, waist_yaw, shoulder pitch/roll/yaw and elbow are
   restored on both sides.

2. `general_motion_retargeting/motion_retarget.py` — `mink.solve_ik` was called
   positionally at four sites, but mink 1.2.0's signature is
   `(configuration, tasks, dt, solver, damping, safety_break, limits, ...)`, so
   `self.ik_limits` bound to **`safety_break`**: the `VelocityLimit` never
   reached the solver and `safety_break` became truthy. Now passed as `limits=`.
