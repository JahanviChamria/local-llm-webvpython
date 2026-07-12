"""pass@k evaluation for both arms.

For each prompt in eval/prompts.jsonl, draw n samples per arm, run each
under the VPython stub (subprocess, timeout), and report unbiased pass@1
and pass@5.

Arm A (nanoGPT): loads out/ckpt.pt directly, prompts with the description
as a comment header since a from-scratch char model has no chat format.
Arm B (LoRA): calls the local Ollama API (default model jahanvi-coder);
start it first with `ollama serve` after `ollama create`.

Usage:
  python eval/passk.py --arm a
  python eval/passk.py --arm b [--model jahanvi-coder]
  python eval/passk.py --arm a --arm b   # both, prints combined table
Results are appended to eval/results.json.
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_SAMPLES = 10
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 700
TIMEOUT_S = 20

SYSTEM = (
    "You write Web VPython 3.2 programs in Python syntax for glowscript.org. "
    "Reply with only the program code, no explanations and no markdown fences."
)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator from the Codex paper (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def runs_ok(code: str) -> bool:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8", dir=tempfile.gettempdir()
    ) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(
            [sys.executable, str(ROOT / "eval" / "run_sim.py"), path],
            capture_output=True, timeout=TIMEOUT_S,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def trim_incomplete(code: str) -> str:
    """Drop a trailing line cut off by the token budget, standard for pass@k."""
    if "\n" in code and not code.endswith("\n"):
        code = code[: code.rfind("\n") + 1]
    return code


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


class ArmA:
    name = "nanoGPT (from scratch)"

    def __init__(self) -> None:
        import pickle

        import torch

        sys.path.insert(0, str(ROOT / "nanogpt"))
        from model import GPT, GPTConfig

        ckpt = torch.load(ROOT / "out" / "ckpt.pt", map_location="cpu")
        conf = GPTConfig(**ckpt["model_args"])
        self.model = GPT(conf)
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
        self.model.load_state_dict(state)
        self.model.eval()
        meta = pickle.loads(
            (ROOT / "data" / ckpt["config"]["dataset"] / "meta.pkl").read_bytes())
        self.stoi, self.itos = meta["stoi"], meta["itos"]
        self.torch = torch

    def generate(self, prompt: str) -> str:
        header = "".join(f"# {line}\n" for line in prompt.splitlines())
        return self.generate_completion(header)

    def generate_completion(self, start: str) -> str:
        ids = [self.stoi[c] for c in start if c in self.stoi]
        x = self.torch.tensor(ids, dtype=self.torch.long)[None, ...]
        with self.torch.no_grad():
            y = self.model.generate(
                x, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=100)
        return "".join(self.itos[int(i)] for i in y[0][len(ids):].tolist())


class ArmB:
    name = "Qwen2.5-Coder-0.5B + LoRA"

    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "system": SYSTEM,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_NEW_TOKENS},
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return strip_fences(json.loads(r.read())["response"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", choices=["a", "b"], required=True)
    ap.add_argument("--model", default="jahanvi-coder")
    ap.add_argument("--n", type=int, default=N_SAMPLES)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in
               (ROOT / "eval" / "prompts.jsonl").read_text("utf-8").splitlines()]

    results = {}
    for arm_id in args.arm:
        arm = ArmA() if arm_id == "a" else ArmB(args.model)
        print(f"\n== {arm.name}, n={args.n} samples/prompt")
        per_prompt = []
        for p in prompts:
            c = 0
            for _ in range(args.n):
                if runs_ok(trim_incomplete(arm.generate(p["prompt"]))):
                    c += 1
            per_prompt.append({"name": p["name"], "n": args.n, "passed": c})
            print(f"  {p['name']}: {c}/{args.n} ran")
        p1 = sum(pass_at_k(r["n"], r["passed"], 1) for r in per_prompt) / len(per_prompt)
        p5 = sum(pass_at_k(r["n"], r["passed"], 5) for r in per_prompt) / len(per_prompt)
        results[arm_id] = {"arm": arm.name, "pass@1": round(p1, 3),
                           "pass@5": round(p5, 3), "prompts": per_prompt}
        print(f"  pass@1 = {p1:.3f}   pass@5 = {p5:.3f}")

    out = ROOT / "eval" / "results.json"
    existing = json.loads(out.read_text("utf-8")) if out.exists() else {}
    existing.update(results)
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
