"""Build training data from corpus/raw for both arms.

Outputs in data/:
  meta.pkl                      char vocab (stoi/itos), shared by all bins
  pretrain_train.bin/_val.bin   general corpus + all public VPython demos
  finetune_train.bin/_val.bin   her sims (train split, oversampled) + Python-syntax demos
  split.json                    which of her sims are held out for eval
  sim_train.json / sim_eval.json  her sims by split, for make_prompts / LoRA pairs

Cleaning: GlowScript version headers stripped (they are not valid Python
and the demo re-adds one on copy). Char-level tokens, uint16 bins.

Usage: python scripts/build_dataset.py
"""

import json
import pickle
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "corpus" / "raw"
DATA = ROOT / "data"

HEADER_RE = re.compile(r"^(GlowScript|Web VPython)[^\n]*\n", re.IGNORECASE)
HELDOUT_FRACTION = 0.25
SIM_OVERSAMPLE = 8  # her 5 training sims are tiny; repeat them in the finetune mix
VAL_FRACTION = 0.05
SEED = 42


def clean(text: str) -> str:
    text = HEADER_RE.sub("", text)
    return text.strip() + "\n"


def read_dir(d: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    seen: set[str] = set()
    for f in sorted(d.glob("*.py")):
        text = clean(f.read_text(encoding="utf-8", errors="ignore"))
        if text not in seen:
            seen.add(text)
            out[f.name] = text
    return out


def write_split(name: str, text: str, stoi: dict[str, int]) -> None:
    # nanoGPT layout: data/{dataset}/train.bin, val.bin, meta.pkl
    d = DATA / name
    d.mkdir(exist_ok=True)
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    n_val = max(1, int(len(ids) * VAL_FRACTION))
    ids[: len(ids) - n_val].tofile(d / "train.bin")
    ids[len(ids) - n_val :].tofile(d / "val.bin")
    print(f"{name}: {len(ids) - n_val:,} train / {n_val:,} val tokens")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    sims = read_dir(RAW / "sims")
    public = read_dir(RAW / "vpython-public")
    general = read_dir(RAW / "general") | {
        f.name: clean(f.read_text(encoding="utf-8", errors="ignore"))
        for ext in (".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".r", ".pde",
                    ".sql", ".java", ".c", ".cpp", ".h", ".sv", ".v", ".sh")
        for f in sorted((RAW / "general").glob(f"*{ext}"))
    }

    # held-out sims for pass@k eval, never trained on
    names = sorted(sims)
    rng.shuffle(names)
    n_hold = max(2, round(len(names) * HELDOUT_FRACTION))
    heldout, train_sims = names[:n_hold], names[n_hold:]
    (DATA / "split.json").write_text(
        json.dumps({"heldout": heldout, "train": train_sims}, indent=2))
    (DATA / "sim_train.json").write_text(
        json.dumps({n: sims[n] for n in train_sims}, indent=2))
    (DATA / "sim_eval.json").write_text(
        json.dumps({n: sims[n] for n in heldout}, indent=2))
    print(f"sims: {len(train_sims)} train, {n_hold} held out: {heldout}")

    python_demos = {n: t for n, t in public.items()
                    if not n.startswith("Examples-JavaScript")}

    pretrain_text = "\n\n".join(
        list(general.values()) + list(public.values()))
    finetune_text = "\n\n".join(
        list(python_demos.values())
        + [sims[n] for n in train_sims] * SIM_OVERSAMPLE)

    chars = sorted(set(pretrain_text) | set(finetune_text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    print(f"vocab: {len(chars)} chars")

    write_split("pretrain", pretrain_text, stoi)
    write_split("finetune", finetune_text, stoi)
    for name in ("pretrain", "finetune"):
        with open(DATA / name / "meta.pkl", "wb") as f:
            pickle.dump({"vocab_size": len(chars), "stoi": stoi, "itos": itos}, f)


if __name__ == "__main__":
    main()
