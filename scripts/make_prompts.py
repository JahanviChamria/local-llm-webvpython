"""Assemble prompt files from eval/descriptions.json and data/split.json.

Outputs:
  eval/prompts.jsonl     held-out sims: {"name", "prompt", "reference"}
  data/lora_train.jsonl  training sims as chat pairs for the LoRA arm

Descriptions were written by hand (Claude-assisted, spot-checked) against
each sim's source. Held-out descriptions double as the pass@k eval prompts.

Usage: python scripts/make_prompts.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEM = (
    "You write Web VPython 3.2 programs in Python syntax for glowscript.org. "
    "Reply with only the program code, no explanations and no markdown fences."
)


def main() -> None:
    descriptions = json.loads((ROOT / "eval" / "descriptions.json").read_text("utf-8"))
    split = json.loads((ROOT / "data" / "split.json").read_text("utf-8"))
    sims = json.loads((ROOT / "data" / "sim_train.json").read_text("utf-8"))
    heldout = json.loads((ROOT / "data" / "sim_eval.json").read_text("utf-8"))

    missing = [n for n in split["train"] + split["heldout"] if n not in descriptions]
    if missing:
        raise SystemExit(f"missing descriptions: {missing}")

    novel = json.loads((ROOT / "eval" / "novel_prompts.json").read_text("utf-8"))
    with open(ROOT / "eval" / "prompts.jsonl", "w", encoding="utf-8") as f:
        for name in split["heldout"]:
            f.write(json.dumps({
                "name": name,
                "prompt": descriptions[name],
                "reference": heldout[name],
            }) + "\n")
        for name, prompt in novel.items():
            # pass@k judges execution only, so novel prompts need no reference
            f.write(json.dumps({"name": name, "prompt": prompt, "reference": None}) + "\n")

    with open(ROOT / "data" / "lora_train.jsonl", "w", encoding="utf-8") as f:
        for name in split["train"]:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": descriptions[name]},
                {"role": "assistant", "content": sims[name]},
            ]}) + "\n")

    print(f"{len(split['heldout']) + len(novel)} eval prompts "
          f"({len(split['heldout'])} held-out + {len(novel)} novel), "
          f"{len(split['train'])} LoRA pairs")


if __name__ == "__main__":
    main()
