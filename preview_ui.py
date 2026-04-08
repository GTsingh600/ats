"""Quick UI preview — fills template placeholders and serves on http://localhost:9000"""

import base64, json, webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

ROOT = Path(__file__).parent / "server"
HTML_PATH = ROOT / "ui_console.html"
IMG_PATH  = ROOT / "ui_aircraft.png"

DUMMY_TASKS = [
    {"task_id": "delhi_monsoon_recovery_easy", "title": "Delhi Monsoon Recovery",
     "difficulty": "Easy", "summary": "10 flights | 2 runways | weather disruption",
     "random_baseline": 0.21},
    {"task_id": "mumbai_bank_balance_medium", "title": "Mumbai Hub Bank Balance",
     "difficulty": "Medium", "summary": "13 flights | 2 runways | airline equity",
     "random_baseline": 0.18},
    {"task_id": "bengaluru_irrops_hard", "title": "Bengaluru IRROPS Recovery",
     "difficulty": "Hard", "summary": "17 flights | 2 runways | emergency priority",
     "random_baseline": 0.12},
    {"task_id": "hyderabad_cargo_crunch_medium_hard", "title": "Hyderabad Cargo Crunch",
     "difficulty": "Hard", "summary": "7 flights | 1 runway | capacity constraint",
     "random_baseline": 0.15},
]

TASK_STRIPS = "\n".join(
    f'<article class="strip" data-task-id="{t["task_id"]}" data-state="ready">'
    f'<div class="strip-top"><span class="strip-title">{t["title"]}</span>'
    f'<span class="strip-difficulty {t["difficulty"].lower()}">{t["difficulty"]}</span></div>'
    f'<div class="strip-summary">{t["summary"]}</div>'
    f'<div class="strip-meta"><span>Baseline {t["random_baseline"]:.2f}</span>'
    '<span class="strip-score">Awaiting run</span></div>'
    "</article>"
    for t in DUMMY_TASKS
)

MODEL_OPTIONS = "\n".join(
    f'<option value="{m}">{m}</option>'
    for m in [
        "heuristic-baseline",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]
)

plane_b64 = "data:image/png;base64," + base64.b64encode(IMG_PATH.read_bytes()).decode()

RAW = (
    HTML_PATH.read_text(encoding="utf-8")
    .replace("__MODEL_OPTIONS__", MODEL_OPTIONS)
    .replace("__PLANE_IMAGE_SRC__", plane_b64)
    .replace("__TASK_DATA__", json.dumps(DUMMY_TASKS))
    .replace("__TASK_STRIPS__", TASK_STRIPS)
)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(RAW.encode())
    def log_message(self, *_):
        pass

PORT = 9000
print(f"Preview -> http://localhost:{PORT}   (Ctrl+C to stop)")
webbrowser.open(f"http://localhost:{PORT}")
HTTPServer(("", PORT), Handler).serve_forever()
