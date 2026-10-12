# thesis_notes

Learning notes, lesson scripts and cluster job scripts from reproducing the GRAIL stair-climbing pipeline (generated from private working notes; identifiers such as cluster user name, project id, emails and key fingerprints are replaced by placeholders like `<cluster-user>`, `<PROJECT-ID>`, `<THESIS_DIR>`).

* `guide.md` : the tutorial (what each stage does and why)
* `implementation_history.md` : chronological log of every action, mistake and fix
* `results.md` : measured outputs
* `llm_answers.md` : answers written by Claude in place of OpenAI API calls
* `lessons/` : small scripts used while learning (MuJoCo, contact checks, rollout reports)
* `cluster/` : Slurm job scripts and the termination/force logger used on the HPC cluster
* `manual_llm/` : wrapper that feeds recorded answers to GRAIL instead of calling OpenAI
* `configs/terrain_stairs_mac.yaml` : copy of GRAIL's `terrain_stairs.yaml` with `rendering.gpu: false`

Not included: `objects_wide3.yaml` (= upstream `configs/objects/terrain_syn_stairs.yaml` plus one entry `stairs_wide3: {obj_scale: [1.0, 1.0, 1.0], scene: indoor1-syn-stairs}`).

Paths in the scripts use the placeholders; replace them with your own before running.
