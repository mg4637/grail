"""Passive diagnostics recorder (v2): a TrajectoryRecorder that ALSO logs
  (1) which termination rules fire                      -> <save_path>/terminations.jsonl
  (2) the real PhysX contact forces on the feet         -> <save_path>/forces.npz     (new in v2)

It subclasses SONIC's TrajectoryRecorderTerm and is registered under the existing recorder name `trajectory`, so
SONIC's own wiring (ManagerEnvWrapper.end_render_results() -> close_writers()) still saves the trajectories.
It changes no behaviour of the environment or the policy.

forces.npz (one entry per RECORDED frame, same cadence/length as the *.trajectory.pkl files, 25 fps):
  foot_net_force_w   (T, num_envs, 2, 3)  net contact force on [left_ankle_roll, right_ankle_roll] (world frame, N)
                                          from env.scene.sensors["contact_forces"].data.net_forces_w
  foot_on_terrain_w  (T, num_envs, 2, 3)  force exerted ON THE STAIRS by [left, right] foot (N), from the
                                          "feet_terrain_contact_sensor" (force_matrix_w); the reaction on the robot
                                          is the negative of this. Missing key if that sensor does not exist.
  foot_body_names    names of the two feet bodies in the sensor
  robot_mass_kg      total robot mass seen by PhysX (so we can compare the summed vertical force with the weight)
  fps                effective frames per second of the arrays

Use (Hydra overrides):
  ++manager_env.recorders.trajectory._target_=term_logger.TrajectoryTermLoggerCfg
  ++manager_env.recorders.trajectory.save_path=<folder>     with PYTHONPATH containing the folder of this file.
"""
import json
import os

import numpy as np
import torch
from isaaclab.utils import configclass

from gear_sonic.envs.manager_env.mdp.recorders import TrajectoryRecorderCfg, TrajectoryRecorderTerm


class TrajectoryTermLoggerTerm(TrajectoryRecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._term_path = os.path.join(self.save_dir, "terminations.jsonl")
        self._term_file = open(self._term_path, "w")
        self._policy_step = 0
        self._error_logged = set()
        self._f_net, self._f_terr = [], []
        self._feet_ids = None

    def _log_error(self, tag, exc):
        if tag not in self._error_logged:
            self._term_file.write(json.dumps({"logger_error": tag, "detail": repr(exc)}) + "\n")
            self._term_file.flush()
            self._error_logged.add(tag)

    def _record_forces(self):
        """Append the foot forces of this step (called only on steps where a trajectory frame was recorded)."""
        sensors = self.env.scene.sensors
        cs = sensors["contact_forces"]
        if self._feet_ids is None:
            ids, names = cs.find_bodies(".*ankle_roll.*")
            self._feet_ids, self._feet_names = list(ids), list(names)
        self._f_net.append(cs.data.net_forces_w[:, self._feet_ids, :].detach().cpu().numpy().copy())
        if "feet_terrain_contact_sensor" in sensors:
            fm = sensors["feet_terrain_contact_sensor"].data.force_matrix_w   # [N, 1, 2, 3]
            self._f_terr.append(fm[:, 0].detach().cpu().numpy().copy())

    def record_post_step(self):
        frame_before = self.frame_id            # the parent records a frame iff frame_before % render_frame_skip == 0
        out = super().record_post_step()
        if getattr(self, "_initialized", False) and frame_before % self.render_frame_skip == 0:
            try:
                self._record_forces()
            except Exception as exc:  # noqa: BLE001  (diagnostics must never break the rollout)
                self._log_error("forces", exc)
        try:
            tm = self.env.termination_manager
            fired = {}
            for name in tm.active_terms:
                idx = torch.nonzero(tm.get_term(name)).flatten().tolist()
                if idx:
                    fired[name] = idx
            if fired:
                self._term_file.write(json.dumps({"step": self._policy_step, "fired": fired}) + "\n")
                self._term_file.flush()
        except Exception as exc:  # noqa: BLE001
            self._log_error("terminations", exc)
        self._policy_step += 1
        return out

    def close_writers(self):
        was_closed = getattr(self, "_closed", False)
        super().close_writers()
        if not was_closed and self._f_net:
            try:
                data = {
                    "foot_net_force_w": np.stack(self._f_net),
                    "foot_body_names": np.array(self._feet_names),
                    "fps": np.array(1.0 / (self.env.step_dt * self.render_frame_skip)),
                }
                if self._f_terr:
                    data["foot_on_terrain_w"] = np.stack(self._f_terr)
                try:
                    data["robot_mass_kg"] = np.array(float(self.env.scene["robot"].root_physx_view.get_masses()[0].sum()))
                except Exception as exc:  # noqa: BLE001
                    self._log_error("mass", exc)
                np.savez(os.path.join(self.save_dir, "forces.npz"), **data)
            except Exception as exc:  # noqa: BLE001
                self._log_error("save_forces", exc)
        try:
            self._term_file.close()
        except Exception:  # noqa: BLE001, S110
            pass


@configclass
class TrajectoryTermLoggerCfg(TrajectoryRecorderCfg):
    class_type = TrajectoryTermLoggerTerm
