"""Execute one candidate program under the VPython stub.

Usage: python eval/run_sim.py <path-to-code-file>
Exit 0 = pass (parses and runs, StopSim counts as running fine).
Exit 1 = fail. Run under a subprocess timeout; a hang is a fail.
"""

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vpython_stub import StopSim, build_globals

HEADER_RE = re.compile(r"^(GlowScript|Web VPython)[^\n]*\n", re.IGNORECASE)


def main() -> int:
    code = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
    code = HEADER_RE.sub("", code)
    try:
        ast.parse(code)
    except SyntaxError as e:
        print(f"FAIL syntax: {e}")
        return 1
    try:
        exec(compile(code, "<candidate>", "exec"), build_globals())
    except StopSim:
        pass
    except BaseException as e:  # noqa: BLE001 - any crash is a fail
        print(f"FAIL runtime: {type(e).__name__}: {e}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
