from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

PIPELINE = [
    "commodity_strategy.py",
    "paper_trading/paper_runner.py",
    "paper_trading/paper_display.py",
    "export_for_portfolio_manager.py",
]


def run_step(script_path: str) -> None:
    full_path = PROJECT_ROOT / script_path

    if not full_path.exists():
        raise FileNotFoundError(f"Missing script: {full_path}")

    print("\n" + "=" * 80)
    print(f"Running: {script_path}")
    print("=" * 80)

    start = time.time()

    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=PROJECT_ROOT,
        text=True,
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"\nFAILED: {script_path}\n"
            f"Exit code: {result.returncode}\n"
            f"Stopped after {elapsed:.1f} seconds."
        )

    print(f"\nCompleted: {script_path} in {elapsed:.1f} seconds.")


def main() -> None:
    print("\nStarting commodity sleeve run for PortfolioManager...")
    print(f"Commodity root: {PROJECT_ROOT}")
    print(f"Python executable: {sys.executable}")

    start = time.time()

    for script in PIPELINE:
        run_step(script)

    elapsed = time.time() - start

    print("\n" + "=" * 80)
    print("COMMODITY SLEEVE EXPORT PIPELINE COMPLETE")
    print(f"Total time: {elapsed:.1f} seconds")
    print("=" * 80)
    print("\nPortfolioManager export folder:")
    print(PROJECT_ROOT / "exports" / "portfolio_manager")


if __name__ == "__main__":
    main()