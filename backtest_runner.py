# run_full_backtest.py

import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

PIPELINE = [
    "Commodity_System/data.py",

    "Commodity_System/scoring/momentum.py",
    "Commodity_System/scoring/relative_strength.py",
    "Commodity_System/scoring/trend.py",
    "Commodity_System/scoring/trend_persistence.py",
    "Commodity_System/scoring/volatility.py",
    "Commodity_System/scoring/risk.py",

    "Commodity_System/commodity_strategy.py",
    "Commodity_System/research/backtester.py",
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
    print("\nStarting full commodity backtest pipeline...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python executable: {sys.executable}")

    start = time.time()

    for script in PIPELINE:
        run_step(script)

    elapsed = time.time() - start

    print("\n" + "=" * 80)
    print("FULL PIPELINE COMPLETE")
    print(f"Total time: {elapsed:.1f} seconds")
    print("=" * 80)

    print("\nCheck outputs here:")
    print(PROJECT_ROOT / "Commodity_System/results/backtest/performance_summary.csv")
    print(PROJECT_ROOT / "Commodity_System/results/backtest/asset_contribution.csv")
    print(PROJECT_ROOT / "Commodity_System/results/backtest/charts")


if __name__ == "__main__":
    main()