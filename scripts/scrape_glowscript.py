"""Scrape a glowscript.org user's programs into corpus/raw/sims.

Usage: python scripts/scrape_glowscript.py [username] [out_subdir]
Writes one .py per program plus a manifest.json with provenance.
out_subdir defaults to "sims"; public demo sets go to e.g. "vpython-public".
"""

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://www.glowscript.org/api"
ROOT = Path(__file__).resolve().parent.parent


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "corpus-scraper"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    user = sys.argv[1] if len(sys.argv) > 1 else "jahanvi"
    out = ROOT / "corpus" / "raw" / (sys.argv[2] if len(sys.argv) > 2 else "sims")
    out.mkdir(parents=True, exist_ok=True)

    folders = get_json(f"{BASE}/user/{user}/folder")["folders"]
    folders = [f["name"] if isinstance(f, dict) else f for f in folders]
    print(f"folders: {folders}")

    manifest = []
    total_bytes = 0
    for folder in folders:
        listing = get_json(f"{BASE}/user/{user}/folder/{folder}/program")
        for prog in listing["programs"]:
            name = prog["name"]
            detail = get_json(f"{BASE}/user/{user}/folder/{folder}/program/{name}")
            source = detail["source"]
            fname = f"{folder}__{name}.py"
            path = out / fname
            path.write_text(source, encoding="utf-8")
            total_bytes += len(source.encode("utf-8"))
            manifest.append(
                {
                    "file": f"{out.name}/{fname}",
                    "origin": f"glowscript:{user}/{folder}/{name}",
                    "url": f"https://www.glowscript.org/#/user/{user}/folder/{folder}/program/{name}",
                    "datetime": detail.get("datetime"),
                    "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    "bytes": len(source.encode("utf-8")),
                }
            )
            print(f"  {fname}  {len(source)} chars")
            time.sleep(0.3)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)} programs, {total_bytes / 1024:.1f} KB -> {out}")


if __name__ == "__main__":
    main()
