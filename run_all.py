"""Launch all 3 Gradipin demo apps concurrently."""

import multiprocessing
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


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
