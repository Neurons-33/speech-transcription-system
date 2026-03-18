import os
import json
import re


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path)


def save_text(text: str, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(data, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_next_run_id(base_dir: str = "outputs/audio") -> str:
    """
    Return next run id like:
    No_001
    No_002
    No_003

    Scan from outputs/audio because processed wav files are stored there.
    Example files:
    outputs/audio/No_001.wav
    outputs/audio/No_002.wav
    """

    ensure_dir(base_dir)

    pattern = re.compile(r"^No_(\d{3})\.wav$")
    numbers = []

    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isfile(full_path):
            match = pattern.match(name)
            if match:
                numbers.append(int(match.group(1)))

    next_number = 1 if not numbers else max(numbers) + 1
    return f"No_{next_number:03d}"