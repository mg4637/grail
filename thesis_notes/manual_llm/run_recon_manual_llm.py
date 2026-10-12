"""Run grail.pipelines.recon_4dhoi with Claude's pre-recorded answers instead of OpenAI.

How it works (no change to the GRAIL repo):
  1. import grail.adapters.openai_api and REPLACE its `chat_with_image` with a lookup
     into interaction_answers.json (keyed by the frame file name).
  2. only THEN import grail.core.contact_label, so its `from ... import chat_with_image`
     binds our replacement (import order is the whole trick).
  3. run recon_4dhoi exactly as `python -m grail.pipelines.recon_4dhoi <args>` would.

Safety rules: only the "is the person in physical contact with the object?" question is
answered; any other question, or any frame without a recorded answer, raises an error and
is written to missing_requests.jsonl. It never guesses.

Env vars: MANUAL_LLM_ANSWERS (path to the json), MANUAL_LLM_LOG (path to a call log).
"""
import json
import os
import runpy
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ANSWERS_PATH = os.environ.get("MANUAL_LLM_ANSWERS", os.path.join(_HERE, "interaction_answers.json"))
LOG_PATH = os.environ.get("MANUAL_LLM_LOG", os.path.join(_HERE, "calls.log"))
MISSING_PATH = os.path.join(os.path.dirname(LOG_PATH), "missing_requests.jsonl")

with open(ANSWERS_PATH) as f:
    ANSWERS = {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def _log(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def manual_chat_with_image(prompt_text, image_path, **kwargs):
    key = os.path.basename(image_path)
    is_contact_question = "physical contact with the object" in prompt_text and "in_contact" in prompt_text
    if not is_contact_question or key not in ANSWERS:
        _log(MISSING_PATH, {"time": time.time(), "image": image_path, "prompt": prompt_text[:2000]})
        raise RuntimeError(
            f"manual LLM has no answer for image={image_path!r} (contact question: {is_contact_question}). "
            f"See {MISSING_PATH}; answer it in llm_answers.md and add it to {ANSWERS_PATH}."
        )
    answer = json.dumps(ANSWERS[key])
    _log(LOG_PATH, {"time": time.time(), "image": image_path, "answer": answer})
    print(f"[manual LLM] {key} -> {answer}", flush=True)
    return answer


# 1) replace the OpenAI call BEFORE contact_label is imported
import grail.adapters.openai_api as _oa  # noqa: E402

_oa.chat_with_image = manual_chat_with_image
# make the "key present" guard irrelevant if anything else calls it
os.environ.setdefault("OPENAI_API_KEY", "manual-llm-not-a-real-key")

# 2) import contact_label now so it binds the replacement; verify
import grail.core.contact_label as _cl  # noqa: E402

assert _cl.chat_with_image is manual_chat_with_image, "patch did not take effect"
print(f"[manual LLM] active; {len(ANSWERS)} recorded answers from {ANSWERS_PATH}", flush=True)

# 3) run the real pipeline CLI
if __name__ == "__main__":
    sys.argv[0] = "grail.pipelines.recon_4dhoi"
    runpy.run_module("grail.pipelines.recon_4dhoi", run_name="__main__", alter_sys=True)
