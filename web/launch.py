#!/usr/bin/env python3
"""
launch.py — starts the SDM backend and frontend in one command.

Usage:
    python3 launch.py              # default ports (backend 8000, frontend 5500)
    python3 launch.py --backend-port 8001 --frontend-port 5501
    python3 launch.py --no-browser # skip auto-opening the browser

Both processes are killed cleanly when you press Ctrl-C.
"""
import argparse
import http.server
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
VENV_PYTHON  = SCRIPT_DIR.parent / ".venv" / "bin" / "python3"
FRONTEND_DIR = SCRIPT_DIR / "frontend"

# Fall back to the system python if the venv is somewhere else
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Launch SDM backend + frontend")
parser.add_argument("--backend-port",  type=int, default=8090)
parser.add_argument("--frontend-port", type=int, default=5500)
parser.add_argument("--no-browser",    action="store_true",
                    help="Don't open the browser automatically")
args = parser.parse_args()

BACKEND_PORT  = args.backend_port
FRONTEND_PORT = args.frontend_port


# ── Helpers ────────────────────────────────────────────────────────────────────
def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_for_port(port: int, timeout: float = 60.0) -> bool:
    """Block until something is listening on port, or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_free(port):
            return True
        time.sleep(0.3)
    return False


class SilentHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Static file server with suppressed access logs and disabled browser caching."""
    def log_message(self, *_):
        pass

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()


# ── Pre-flight checks ──────────────────────────────────────────────────────────
print("=" * 55)
print("  SDM Explorer — launcher")
print("=" * 55)

if not port_free(BACKEND_PORT):
    print(f"⚠️  Port {BACKEND_PORT} already in use — backend may already be running.")
if not port_free(FRONTEND_PORT):
    print(f"⚠️  Port {FRONTEND_PORT} already in use — try --frontend-port <other>")

if not FRONTEND_DIR.exists():
    print(f"❌  Frontend directory not found: {FRONTEND_DIR}")
    sys.exit(1)

print(f"  Python   : {PYTHON}")
print(f"  Backend  : http://localhost:{BACKEND_PORT}")
print(f"  Frontend : http://localhost:{FRONTEND_PORT}")
print()


# ── 1. Backend (uvicorn + FastAPI) ─────────────────────────────────────────────
print("▶  Starting backend …", flush=True)
backend_proc = subprocess.Popen(
    [
        PYTHON, "-m", "uvicorn", "server:app",
        "--host", "0.0.0.0",
        "--port", str(BACKEND_PORT),
        "--workers", "1",
        "--log-level", "info",
        "--reload",
    ],
    cwd=str(SCRIPT_DIR),
    # Forward stdout/stderr so model loading messages are visible
    stdout=sys.stdout,
    stderr=sys.stderr,
)


# ── 2. Frontend (stdlib static server in a thread) ────────────────────────────
def run_frontend():
    os.chdir(str(FRONTEND_DIR))
    handler = SilentHTTPHandler
    with http.server.ThreadingHTTPServer(("0.0.0.0", FRONTEND_PORT), handler) as httpd:
        httpd.serve_forever()

frontend_thread = threading.Thread(target=run_frontend, daemon=True)
frontend_thread.start()
print(f"▶  Frontend serving {FRONTEND_DIR.name}/ on port {FRONTEND_PORT}", flush=True)


# ── 3. Wait for backend to be ready, then open browser ────────────────────────
def open_browser_when_ready():
    if wait_for_port(BACKEND_PORT, timeout=120):
        url = f"http://localhost:{FRONTEND_PORT}/map.html"
        print(f"\n✅  Backend ready — opening {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
    else:
        print(f"\n⚠️  Backend did not respond on port {BACKEND_PORT} within 120s.", flush=True)

threading.Thread(target=open_browser_when_ready, daemon=True).start()


# ── 4. Keep running until Ctrl-C ──────────────────────────────────────────────
print("\nPress Ctrl-C to stop both servers.\n", flush=True)

def _shutdown(sig, frame):
    print("\n\nShutting down …", flush=True)
    backend_proc.terminate()
    try:
        backend_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        backend_proc.kill()
    print("✅  Done.", flush=True)
    sys.exit(0)

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# Wait for backend process to exit on its own (e.g. crash)
ret = backend_proc.wait()
if ret != 0:
    print(f"\n❌  Backend exited with code {ret}.", flush=True)
sys.exit(ret)
