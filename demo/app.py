"""Local demo: type a physics prompt, get a Web VPython sim from either arm.

Usage: python demo/app.py  ->  http://localhost:8155
Arm A needs out/ckpt.pt (downloaded from the Colab run).
Arm B needs Ollama serving jahanvi-coder (ollama create -f demo/Modelfile).

Stdlib only. Generated code gets the Web VPython header prepended so it
pastes straight into the glowscript.org editor.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from passk import ArmA, ArmB  # noqa: E402

PORT = 8155
HEADER = "Web VPython 3.2\n"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>jahanvi-coder</title>
<style>
body { font-family: ui-monospace, Consolas, monospace; max-width: 760px;
       margin: 3rem auto; padding: 0 1rem; background: #111; color: #ddd; }
h1 { font-size: 1.1rem; font-weight: 600; }
textarea, pre { width: 100%; box-sizing: border-box; background: #1a1a1a;
       color: #ddd; border: 1px solid #333; padding: .75rem; font: inherit; }
textarea { height: 5.5rem; }
pre { min-height: 12rem; white-space: pre-wrap; overflow-x: auto; }
button, select { background: #222; color: #ddd; border: 1px solid #444;
       padding: .5rem 1rem; font: inherit; cursor: pointer; margin-right: .5rem; }
button:hover { background: #333; }
a { color: #8ab4f8; }
.row { margin: .75rem 0; }
#status { color: #888; }
</style></head><body>
<h1>jahanvi-coder — Web VPython from a prompt</h1>
<div class="row"><textarea id="prompt"
 placeholder="A red ball bounces on a blue floor under gravity..."></textarea></div>
<div class="row">
  <select id="arm">
    <option value="b">Qwen2.5-Coder-0.5B + LoRA (Ollama)</option>
    <option value="a">nanoGPT from scratch (10M, char-level)</option>
  </select>
  <button onclick="gen()">Generate</button>
  <button onclick="copyCode()">Copy</button>
  <a href="https://glowscript.org" target="_blank">open glowscript.org to paste &amp; run</a>
  <span id="status"></span>
</div>
<pre id="code"></pre>
<script>
async function gen() {
  const status = document.getElementById('status');
  status.textContent = 'generating...';
  document.getElementById('code').textContent = '';
  try {
    const r = await fetch('/generate', { method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ prompt: document.getElementById('prompt').value,
                             arm: document.getElementById('arm').value }) });
    const j = await r.json();
    document.getElementById('code').textContent = j.code || j.error;
    status.textContent = j.code ? '' : 'error';
  } catch (e) { status.textContent = 'error: ' + e; }
}
function copyCode() {
  navigator.clipboard.writeText(document.getElementById('code').textContent);
  document.getElementById('status').textContent = 'copied';
}
</script></body></html>"""

arms: dict[str, object] = {}


def get_arm(arm_id: str):
    if arm_id not in arms:
        arms[arm_id] = ArmA() if arm_id == "a" else ArmB("jahanvi-coder")
    return arms[arm_id]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE.encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            code = get_arm(body["arm"]).generate(body["prompt"])
            payload = {"code": HEADER + code.strip() + "\n"}
        except Exception as e:  # noqa: BLE001 - surface any failure to the page
            payload = {"error": f"{type(e).__name__}: {e}"}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"http://localhost:{PORT}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
