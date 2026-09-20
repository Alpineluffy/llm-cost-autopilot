#!/usr/bin/env python3
"""Capture dashboard screenshots as portfolio artifacts.

Starts nothing itself — point it at a running dashboard:

    streamlit run dashboard/app.py --server.headless true &
    python scripts/capture_dashboard.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "reports" / "screenshots"

# Streamlit renders charts client-side, so the page needs time to settle
# before anything is worth capturing.
SETTLE_MS = 9000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8599")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed:  pip install playwright")
        print("then:                          playwright install chromium")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Streamlit scrolls inside its own container, so `full_page=True` captures
    # no more than the viewport. Use a tall viewport for the whole-page shot
    # instead. The theme comes from Streamlit's server config, not from the
    # browser, so it is not something this script can vary per shot.
    shots = [
        ("dashboard-hero", args.width, args.height),
        ("dashboard-full", args.width, 3200),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, width, height in shots:
            page = browser.new_page(
                viewport={"width": width, "height": height},
                device_scale_factor=2,  # retina, so the image survives zooming
            )
            try:
                page.goto(args.url, wait_until="networkidle", timeout=60_000)
            except Exception as e:
                print(f"Could not load {args.url}: {e}")
                print("Is the dashboard running?")
                browser.close()
                return 1

            page.wait_for_timeout(SETTLE_MS)

            path = OUT_DIR / f"{name}.png"
            page.screenshot(path=str(path))
            print(f"  {path.relative_to(PROJECT_ROOT)}  ({width}x{height} @2x)")
            page.close()
        browser.close()

    print(f"\nSaved to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
