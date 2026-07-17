
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

PIPELINE = [
    "data.py",
    "macro_data.py",

    "scoring/commodity_models/Gold/gold_data.py",
    "scoring/commodity_models/Gold/gold_features.py",
    "scoring/commodity_models/Silver/silver_features.py",
    "scoring/commodity_models/Copper/copper_data.py",
    "scoring/commodity_models/Copper/copper_features.py",
    "scoring/commodity_models/Oil/oil_data.py",
    "scoring/commodity_models/Oil/oil_features.py",
    #"scoring/commodity_models/Gas/gas_data.py",
    #"scoring/commodity_models/Gas/gas_features.py",
    "scoring/commodity_models/Agriculture/agriculture_data.py",
    "scoring/commodity_models/Agriculture/agriculture_features.py",

    "scoring/momentum.py",
    "scoring/relative_strength.py",
    "scoring/trend.py",
    "scoring/trend_persistence.py",
    "scoring/volatility.py",
    "scoring/risk.py",
    "scoring/macro_features.py",

    "commodity_strategy.py",
    "research/Backtesting/backtester.py",
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