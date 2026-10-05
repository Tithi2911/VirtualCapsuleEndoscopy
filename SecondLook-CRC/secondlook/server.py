"""Minimal local web interface: upload a recording, get the review report.

Standard library only and bound to 127.0.0.1, so nothing leaves the machine
(on-premises mode). Analysis runs synchronously; a hospital deployment would
put this behind the trust's authentication and a job queue (see docs/DESIGN.md).
"""

from __future__ import annotations

import html
import json
import re
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import INTENDED_USE_NOTICE
from .characterise import Characteriser, NullCharacteriser
from .cli import run_analysis
from .diagnosis import load_classifier
from .diagnosis.classifier import SAFETY_STATEMENT
from .ingest import IMAGE_EXTS, VIDEO_EXTS
from .models import Modality

MAX_UPLOAD_BYTES = 4 * 1024**3

UPLOAD_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SecondLook Upload</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--card:#fff;--line:#deddd8;--accent:#0f6e8c;--warn:#b4321f}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--card:#212120;--line:#383835;--accent:#53b5d4;--warn:#ef7b67}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:640px;margin:0 auto;padding:32px 16px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px}
label{display:block;margin:12px 0 4px;font-weight:600}input,select{width:100%;padding:8px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px}
button{margin-top:16px;background:var(--accent);color:#fff;border:0;border-radius:6px;padding:10px 18px;font-size:1rem;cursor:pointer}
.notice{border-left:4px solid var(--warn);padding:8px 12px;margin-bottom:16px;background:var(--card)}#status{margin-top:12px}
.muted{opacity:.8;font-size:.9rem}
</style></head><body><main>
<h1>SecondLook &mdash; upload a recording</h1>
<div class="notice">__NOTICE__ Runs entirely on this computer.</div>
<p class="muted">__CLASSIFIER__</p>
<div class="card">
<label for="file">Video file or single image</label><input id="file" type="file" accept="video/*,image/*">
<label for="modality">Modality</label><select id="modality"><option value="colonoscopy">Traditional colonoscopy</option><option value="capsule">Capsule endoscopy</option></select>
<label for="report">Procedure report JSON (optional &mdash; enables missed-finding comparison)</label><input id="report" type="file" accept=".json">
<button onclick="go()">Analyse</button><div id="status"></div></div>
<p><a href="/runs">Previous analyses</a></p>
</main><script>
async function go(){
  const f=document.getElementById('file').files[0], st=document.getElementById('status');
  if(!f){st.textContent='Choose a file first.';return;}
  const rep=document.getElementById('report').files[0];
  st.textContent='Uploading and analysing... this can take a few minutes for long videos.';
  const h={'X-Filename':encodeURIComponent(f.name),'X-Modality':document.getElementById('modality').value};
  if(rep){h['X-Report']=encodeURIComponent(await rep.text());}
  const r=await fetch('/analyse',{method:'POST',headers:h,body:f});
  const j=await r.json();
  if(r.ok){location.href=j.report;}else{st.textContent='Error: '+j.error;}
}
</script></body></html>"""


def classifier_status(characteriser: Characteriser) -> str:
    if isinstance(characteriser, NullCharacteriser):
        return "No lesion classifier loaded: findings are detected but not characterised (start with --classifier MODEL.onnx)."
    info = getattr(characteriser, "info", None) or {}
    trained_for = f", trained for {' and '.join(info['modalities'])}" if info.get("modalities") else ""
    parts = [
        f"Lesion classifier: {characteriser.name} {characteriser.version} (benign / precancerous / cancerous"
        f"{trained_for}).",
        info.get("validation_status"),
        SAFETY_STATEMENT,
    ]
    return " ".join(p for p in parts if p)


def make_handler(data_dir: Path, classifier_path: Path | str | None = None):
    data_dir.mkdir(parents=True, exist_ok=True)
    # Loaded once, so a bad model fails at start-up and requests share one ONNX session.
    characteriser = load_classifier(classifier_path) if classifier_path else NullCharacteriser()
    status = classifier_status(characteriser)
    upload_page = (
        UPLOAD_PAGE.replace("__NOTICE__", html.escape(INTENDED_USE_NOTICE))
        .replace("__CLASSIFIER__", html.escape(status))
        .encode()
    )

    class Handler(BaseHTTPRequestHandler):
        status_text = status

        def _send(self, status, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, obj):
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(HTTPStatus.OK, upload_page, "text/html; charset=utf-8")
            if self.path == "/runs":
                runs = sorted((p for p in data_dir.iterdir() if (p / "report.html").exists()), reverse=True)
                items = "".join(f'<li><a href="/runs/{p.name}/report.html">{html.escape(p.name)}</a></li>' for p in runs)
                return self._send(HTTPStatus.OK, f"<!doctype html><meta charset=utf-8><title>Runs</title><ul>{items}</ul>".encode(), "text/html")
            m = re.fullmatch(r"/runs/([A-Za-z0-9_-]+)/report\.html", self.path)
            if m and (data_dir / m.group(1) / "report.html").exists():
                return self._send(HTTPStatus.OK, (data_dir / m.group(1) / "report.html").read_bytes(), "text/html; charset=utf-8")
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):
            if self.path != "/analyse":
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            from urllib.parse import unquote

            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_UPLOAD_BYTES:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "empty or too large upload"})
            name = Path(unquote(self.headers.get("X-Filename", "upload"))).name
            ext = Path(name).suffix.lower()
            if ext not in VIDEO_EXTS | IMAGE_EXTS:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": f"unsupported file type {ext}"})
            modality = self.headers.get("X-Modality", Modality.COLONOSCOPY.value)
            if modality not in {m.value for m in Modality}:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "unknown modality"})

            run_id = uuid.uuid4().hex[:12]
            run_dir = data_dir / run_id
            run_dir.mkdir()
            upload = run_dir / f"input{ext}"
            remaining = length
            with open(upload, "wb") as f:
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1 << 20))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            report_path = None
            if self.headers.get("X-Report"):
                report_path = run_dir / "procedure_report.json"
                report_path.write_text(unquote(self.headers["X-Report"]))
            try:
                run_analysis(upload, run_dir, modality, report_path, audit_log=data_dir / "audit.jsonl",
                             characteriser=characteriser)
            except Exception as e:  # report the failure to the browser rather than dropping the connection
                return self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(e)})
            self._json(HTTPStatus.OK, {"report": f"/runs/{run_id}/report.html"})

    return Handler


def serve(port: int, data_dir: Path, classifier_path: Path | str | None = None) -> None:
    handler = make_handler(Path(data_dir), classifier_path)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"SecondLook running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    print(handler.status_text)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
