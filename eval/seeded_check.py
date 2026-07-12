"""Code-seeded run rate for arm A.

The from-scratch char model is a completion model, not an instruction
follower, so alongside pass@k (where NL prompts are out-of-distribution
for it) this measures what it did learn: seed it with a code opening and
count how often the completion parses and executes under the stub.

Usage: python eval/seeded_check.py [--n 10]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from passk import ArmA, runs_ok, trim_incomplete  # noqa: E402

SEEDS = [
    "ball=sphere(pos=vector(0,5,0), color=color.red, make_trail=True)\n",
    "sun=sphere(pos=vector(0,0,0), color=color.orange, radius=0.75)\n",
    "scene.background=color.white\n",
    "g=9.8\ndt=0.01\n",
    "graph(width=400, height=250, background=color.white)\n",
    "table=box(pos=vector(0,-2,0), color=color.black, length=4)\n",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    arm = ArmA()
    total = passed = 0
    per_seed = []
    for seed in SEEDS:
        c = 0
        for _ in range(args.n):
            code = seed + trim_incomplete(arm.generate_completion(seed))
            if runs_ok(code):
                c += 1
        per_seed.append({"seed": seed.splitlines()[0][:40], "n": args.n, "passed": c})
        total += args.n
        passed += c
        print(f"  {seed.splitlines()[0][:40]:42} {c}/{args.n} ran")
    rate = passed / total
    print(f"\ncode-seeded run rate: {passed}/{total} = {rate:.2f}")

    out = ROOT / "eval" / "results.json"
    existing = json.loads(out.read_text("utf-8")) if out.exists() else {}
    existing["a_seeded"] = {"arm": arm.name, "metric": "code-seeded run rate",
                            "rate": round(rate, 3), "seeds": per_seed}
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
