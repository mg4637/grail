"""
Manual drop-in replacement for grail/adapters/openai_api.py.

Instead of calling an API, this prints the prompt (and, for vision calls,
the image path to open in a browser) and pauses for you to paste in the
response you got from Gemini / DeepSeek's web chat interface.

No network calls, no API key, no local LLM server required.
"""

import json
import os
import subprocess
import sys

DEFAULT_REASONING_MODEL = "manual"


def _open_image(image_path):
    """Best-effort: open the image in the default viewer so you can drag it into the web UI."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", image_path])
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", image_path])
    except Exception:
        pass


def _prompt_and_wait(header, body, is_json=False):
    print("\n" + "=" * 70)
    print(header)
    print("-" * 70)
    print(body)
    print("=" * 70)
    print("Paste the response below. End input with a line containing only: END")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    text = "\n".join(lines).strip()

    if is_json:
        # Strip markdown code fences if the model wrapped its JSON
        cleaned = text.replace("```json", "").replace("```", "").strip()
        try:
            json.loads(cleaned)  # validate only
            return cleaned
        except Exception:
            print("  Warning: pasted text is not valid JSON, using as-is")
            return cleaned
    return text


def chat_text(prompt_text, max_tokens=512, temperature=0.7, system_prompt=None, model=None):
    header = "COPY THIS INTO GEMINI / DEEPSEEK (text-only):"
    body = (prompt_text if not system_prompt else f"[System instruction]\n{system_prompt}\n\n[User]\n{prompt_text}")
    return _prompt_and_wait(header, body)


def chat_with_image(
    prompt_text,
    image_path,
    max_tokens=512,
    temperature=0.7,
    system_prompt=None,
    model=None,
    response_format=None,
):
    _open_image(image_path)
    header = f"OPEN/DRAG THIS IMAGE INTO GEMINI OR DEEPSEEK: {image_path}"
    body = (prompt_text if not system_prompt else f"[System instruction]\n{system_prompt}\n\n[User]\n{prompt_text}")
    is_json = bool(response_format and response_format.get("type") == "json_object")
    return _prompt_and_wait(header, body, is_json=is_json)


def generate_image(prompt, output_path, **kwargs):
    """
    No good manual equivalent for image generation via chat web UIs at scale.
    Recommend: skip AI image-gen entirely and supply your own reference photos
    (see gen_3d_from_photos.py). This stub raises so you don't silently no-op.
    """
    raise NotImplementedError(
        "generate_image() has no manual web-UI equivalent for batch use. "
        "Use gen_3d_from_photos.py with your own reference photos instead of "
        "gen_3d_assets.py's AI image generation step."
    )