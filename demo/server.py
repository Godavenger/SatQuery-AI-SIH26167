# SatQuery AI — lead demo server (optional)
# Zero-dependency: serves demo/ + mock POST /api/query with same contract as finale FastAPI.
# Run: python server.py  -> open http://localhost:8000
import json, http.server, functools, pathlib

ROOT = pathlib.Path(__file__).parent

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)
    def do_POST(self):
        if self.path == "/api/query":
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            q = body.get("query", "").lower()
            if any(k in q for k in ("check if", "is there", "are there")): task = "vqa"
            elif "highlight" in q or "water body" in q: task = "ground"
            elif "increased" in q or "decreased" in q or "unchanged" in q: task = "change-qa"
            elif "changed" in q or "between" in q: task = "change"
            elif "together" in q or "fuse" in q: task = "fuse"
            elif "describe" in q or "land-cover" in q: task = "describe"
            else: task = "vqa"
            out = {"task": task, "model": "demo-stub",
                   "answer": f"Stubbed '{task}' answer for: {body.get('query','')}",
                   "confidence": 0.8, "trace": f"task={task} | model=demo-stub | params=tile512"}
            b = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers()
            self.wfile.write(b)
        else:
            self.send_error(404)

print("SatQuery lead demo at http://localhost:8000  (Ctrl+C to stop)")
http.server.ThreadingHTTPServer(("127.0.0.1", 8000), H).serve_forever()
