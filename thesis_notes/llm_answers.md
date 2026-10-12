# LLM answers (Claude standing in for the OpenAI API)

GRAIL calls OpenAI (`gpt-4o`) in two places. We have no key, so Claude answers
the same prompts by hand. Every prompt and answer is recorded here so the run is
reproducible and you can see exactly what the LLM contributed.

| Call site | Purpose | Status |
|---|---|---|
| 2D-HOI step 4: prompt refinement | Look at the rendered frame, write the video prompt | answered (Entry 1) |
| 4D-recon step 4: contact labelling (`grail/core/contact_label.py`) | Which body joints touch the object per time interval | pending |
| 2D-HOI step 2: scale evaluation | small / big / correct | skipped by default |


---

## Entry 1 — 2D-HOI step 4: prompt refinement (stairs_wide3)

**Role:** I am standing in for `gpt-4o` in `refine_prompt_with_chatgpt()` (`grail/pipelines/gen_2dhoi.py:234`).

**Input image:** `results/generation/asset_renders/SynStairsV1/stairs_wide3/kid_indoor1-syn-stairs_rand00001.png`

**Base prompt (variant 1 of `terrain_stairs.yaml`):**
> The person walks toward the staircase and carefully climbs up the stairs step by step. They stop partway up and remain standing there. The camera should remain static.

**System message (variant 1):** "Generate a video prompt for a person climbing a staircase. ... If the staircase is tall, climb partway and stop. If short, they can reach the top. NEVER walk beyond the visible stairs or disappear from the frame. Static camera, slow careful pace one step at a time, person stays visible, stops on the stairs. Output ONLY the prompt."

**My reasoning (not part of the output):** the staircase is 0.36 m tall, only three low steps, so it counts as *short* — the kid may reach the top. The top platform is 0.7 m deep, so he can stand on it. The kid is on the left, the stairs are to his right, so the walk goes left-to-right. I avoid mentioning walls, ceiling, or the door to avoid the video model drifting.

**My answer (this is what you paste into the video tool):**
> A boy stands indoors on the left side of the frame, facing a low three-step staircase to his right. Static camera. He walks slowly toward the stairs, then carefully climbs up one step at a time, placing his feet fully on each step. He reaches the top platform, stops, and stands still, remaining fully visible in the frame.

**Settings GRAIL would send to Kling** (from the config): model `kling-v3`, mode `pro`, duration 10 s, 1 video, start image = the frame above, static camera. Use the closest equivalents in your web UI.


---

## Entry 2 — 4D recon: interaction start detection, `detect_interaction()` (stairs_wide3)

**Role:** standing in for `gpt-4o` at `grail/core/contact_label.py:207` (called from `grail/optimization/interaction.py:70`).

**Why it is called:** the stairs never move, so the object-motion loop in `identify_interaction_start_end` finds no start; it then asks the LLM at 8 keyframes (`i = T-interval, ..., 0`, `interval = T // 8`), **scanning from the end backwards, stopping at the first "true"**. For our 240-frame, 24 fps video: frames 210, 180, 150, 120, 90, 60, 30, 0. (Assumes the optimizer sees all 240 extracted frames named `%06d.jpg`; extracted by `extract_frames_from_video`; to be confirmed in the recon logs.)

**Question GRAIL asks (system: "precise multi-modal vision classifier"):** "determine whether the person is in physical contact with the object. Object: `<name>`. Contact = touching or holding; near proximity does NOT count; any body part touching counts; be conservative when uncertain. Respond as JSON `{"in_contact": true/false, "confidence": ...}`."

**Images:** `docs_learning/frames/keyframes/` (and the tiled `keyframes_sheet.png`).

| frame | what I see | my answer |
|---|---|---|
| 000000 | kid standing at left, far from stairs, hands in front | false (high) |
| 000030 | kid walking, both feet on floor, still left of the stairs | false (high) |
| 000060 | kid close to the lowest step, feet on floor, not touching | false (medium: close, but no visible touch; the rule says near-proximity does not count) |
| 000090 | one foot raised onto the lowest step | **true** (high) |
| 000120 | foot on the first/second step, body leaning forward | **true** (high) |
| 000150 | foot on the middle/top step, climbing | **true** (high) |
| 000180 | standing on the top platform, looking down | **true** (high) |
| 000210 | standing on the top platform, facing the camera | **true** (high) |

Machine-readable copy: `docs_learning/manual_llm/interaction_answers.json`.

**Consequence to watch (not decided yet):** because the scan runs from the end and stops at the first "true", the first question (frame 210) already returns true, so GRAIL would set `inter_start_idx = 210`, i.e. an interaction start in the last 30 of 240 frames. That is what the code's logic gives with a faithful answer; I answered the question as asked and did not tune the answer. For a static object the code skips contact labelling, so it may matter little, but I have not verified how `inter_start_idx` is used further down. We will check the recon log for it before trusting the result.
