"""Collect Jahanvi's complete code corpus into corpus/raw/general.

Sources:
  1. GitHub: every non-fork public repo of JahanviChamria (shallow clones).
  2. Local project folders not on GitHub (course work, etc.).

Keeps source files only, prunes junk dirs during the walk, extracts code
cells from notebooks, dedupes by content hash. Files that look like Web
VPython also land in corpus/raw/sims. For repos owned by someone else,
only files Jahanvi authored (per git log) are taken.

Usage: python scripts/collect_repos.py
"""

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "corpus" / "raw" / "general"
SIMS = ROOT / "corpus" / "raw" / "sims"
CLONES = ROOT / "corpus" / "clones"

GITHUB_USER = "JahanviChamria"

LOCAL_SOURCES = [
    Path(r"C:\Users\jahan\Documents\Claude\couple-games"),
    Path(r"C:\Users\jahan\Documents\Claude\links-hub"),
    Path(r"C:\Users\jahan\Documents\Claude\Scientific-Inquiry-With-AI-Website"),
    Path(r"C:\Users\jahan\Documents\ComputerNetworks"),
    Path(r"C:\Users\jahan\Documents\Processing"),
    Path(r"C:\Users\jahan\PycharmProjects"),
    # coursework folders (Downloads), provided 2026-07-11
    Path(r"C:\Users\jahan\Downloads\COSC202Lab"),
    Path(r"C:\Users\jahan\Downloads\Program 5 starter files-20251031"),
    Path(r"C:\Users\jahan\Downloads\Lab 6 starter files-20251031"),
    Path(r"C:\Users\jahan\Downloads\Lab 05"),
    Path(r"C:\Users\jahan\Downloads\P01_starter (1)"),
    Path(r"C:\Users\jahan\Downloads\L04_starter"),
    Path(r"C:\Users\jahan\Downloads\L03_starter"),
    Path(r"C:\Users\jahan\Downloads\P02_milestone2"),
    Path(r"C:\Users\jahan\Downloads\Lab11_starter"),
    Path(r"C:\Users\jahan\Downloads\L10_starter"),
    Path(r"C:\Users\jahan\Downloads\L09_starter"),
    Path(r"C:\Users\jahan\Downloads\P02_starter"),
    Path(r"C:\Users\jahan\Downloads\L07_starter"),
    Path(r"C:\Users\jahan\Downloads\L06_starter"),
    Path(r"C:\Users\jahan\Downloads\L01_starter"),
    Path(r"C:\Users\jahan\Downloads\L02_starter"),
]

# repos where she is a contributor, not the owner: keep only her files
COLLAB_AUTHOR_FILTER = {"Scientific-Inquiry-With-AI-Website": "Jahanvi"}

SKIP_REPOS = {"forage-lyft-starter-repo", "Dataset", "ExoplanetData", "NASADataset", "testing"}

PRUNE_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
    ".next", "out", "site-packages", ".obsidian", "coverage", ".pytest_cache",
    "vendor", "third_party",
}
SOURCE_EXT = {
    ".py", ".ipynb", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".r",
    ".pde", ".sql", ".java", ".c", ".cpp", ".h", ".sv", ".v", ".sh",
}
SKIP_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
MAX_BYTES = 200_000

VPYTHON_MARKERS = ("GlowScript ", "Web VPython", "from vpython import", "import vpython")


def notebook_code(raw: str) -> str:
    try:
        nb = json.loads(raw)
        cells = [
            "".join(c["source"])
            for c in nb.get("cells", [])
            if c.get("cell_type") == "code"
        ]
        return "\n\n".join(cells)
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""


def authored_files(repo: Path, author: str) -> set[str]:
    r = subprocess.run(
        ["git", "-C", str(repo), "log", f"--author={author}", "--name-only",
         "--pretty=format:"],
        capture_output=True, text=True,
    )
    return {line.strip() for line in r.stdout.splitlines() if line.strip()}


def iter_source_files(base: Path):
    stack = [base]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if e.name not in PRUNE_DIRS and not e.name.startswith("."):
                    stack.append(e)
            elif (
                e.suffix.lower() in SOURCE_EXT
                and e.name not in SKIP_NAMES
                and not e.name.endswith(".min.js")
            ):
                yield e


def github_repos() -> list[str]:
    url = f"https://api.github.com/users/{GITHUB_USER}/repos?per_page=100"
    req = urllib.request.Request(url, headers={"User-Agent": "corpus-scraper"})
    with urllib.request.urlopen(req, timeout=30) as r:
        repos = json.loads(r.read().decode("utf-8"))
    return [r["name"] for r in repos if not r["fork"] and r["name"] not in SKIP_REPOS]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SIMS.mkdir(parents=True, exist_ok=True)
    CLONES.mkdir(parents=True, exist_ok=True)

    bases: list[tuple[Path, str]] = []  # (path, origin label)

    for name in github_repos():
        dest = CLONES / name
        if not dest.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet",
                 f"https://github.com/{GITHUB_USER}/{name}.git", str(dest)],
                check=True,
            )
        bases.append((dest, f"github:{GITHUB_USER}/{name}"))

    for p in LOCAL_SOURCES:
        if p.exists():
            bases.append((p, f"local:{p.name}"))
        else:
            print(f"missing local source: {p}")

    manifest, seen = [], set()
    sim_count = total = 0
    for base, origin in bases:
        allow: set[str] | None = None
        if base.name in COLLAB_AUTHOR_FILTER:
            allow = authored_files(base, COLLAB_AUTHOR_FILTER[base.name])
        for f in iter_source_files(base):
            rel = f.relative_to(base).as_posix()
            if allow is not None and rel not in allow:
                continue
            try:
                raw = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if f.suffix == ".ipynb":
                raw = notebook_code(raw)
            if not raw.strip() or len(raw.encode()) > MAX_BYTES:
                continue
            sha = hashlib.sha256(raw.encode()).hexdigest()
            if sha in seen:
                continue
            seen.add(sha)
            safe = f"{base.name}__{rel.replace('/', '__')}"
            if f.suffix == ".ipynb":
                safe += ".py"
            is_sim = any(m in raw for m in VPYTHON_MARKERS)
            dest_dir = SIMS if is_sim else OUT
            (dest_dir / safe).write_text(raw, encoding="utf-8")
            manifest.append(
                {
                    "file": f"{'sims' if is_sim else 'general'}/{safe}",
                    "origin": f"{origin}/{rel}",
                    "sha256": sha,
                    "bytes": len(raw.encode()),
                }
            )
            total += len(raw.encode())
            sim_count += is_sim

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(manifest)} files, {total / 1024:.0f} KB total, {sim_count} routed to sims")


if __name__ == "__main__":
    main()
