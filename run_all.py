"""Launch all 3 Gradipin demo apps concurrently."""

import multiprocessing
import os
import signal
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

from dotenv import load_dotenv

env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        pass


def _run_health_server():
    """Tiny HTTP server on the PORT env var so Render health checks pass."""
    port = int(os.environ.get("PORT", "8000"))
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    print(f"  Health-check server listening on :{port}")
    server.serve_forever()


def run_sentiment():
    from demos.sentiment import main
    main(port=7860)


def run_image_classify():
    from demos.image_classify import main
    main(port=7861)


def run_speech_to_text():
    from demos.speech_to_text import main
    main(port=7862)


DEMOS = [
    ("Sentiment Analysis", run_sentiment),
    ("Image Classification", run_image_classify),
    ("Speech to Text", run_speech_to_text),
]


def main():
    print("=" * 60)
    print("  Gradipin Demos — Launching 3 apps...")
    print("=" * 60)
    print()

    health_thread = Thread(target=_run_health_server, daemon=True)
    health_thread.start()

    processes: list[multiprocessing.Process] = []
    for name, target in DEMOS:
        p = multiprocessing.Process(target=target, name=name, daemon=True)
        p.start()
        processes.append(p)
        print(f"  Started: {name} (pid={p.pid})")

    print()
    print("  All demos launching. Check output above for Gradipin URLs.")
    print("  Press Ctrl+C to stop all demos.")
    print("=" * 60)

    def shutdown(signum, frame):
        print("\n\nShutting down all demos...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
