from __future__ import annotations
import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
PAPER_TRADING_DIR = THIS_FILE.parent
COMMODITY_ROOT = THIS_FILE.parents[1]

for path in [COMMODITY_ROOT, PAPER_TRADING_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ============================================================
# IMPORTS
# ============================================================

import paper_config as pc


# ============================================================
# PATHS
# ============================================================

PAPER_STATE_PATH = getattr(
    pc,
    "PAPER_STATE_PATH",
    PAPER_TRADING_DIR / "state" / "paper_state.csv",
)

PAPER_REPORTS_DIR = getattr(
    pc,
    "PAPER_REPORTS_DIR",
    PAPER_TRADING_DIR / "reports",
)

DASHBOARD_HTML_PATH = PAPER_REPORTS_DIR / "paper_dashboard.html"

# Root-level project scripts
DATA_SCRIPT = COMMODITY_ROOT / "data.py"
FULL_PIPELINE_SCRIPT = COMMODITY_ROOT / "backtest_runner.py"
PAPER_RUNNER_SCRIPT = PAPER_TRADING_DIR / "paper_runner.py"
PAPER_DISPLAY_SCRIPT = PAPER_TRADING_DIR / "paper_display.py"


# ============================================================
# STATE HELPERS
# ============================================================

def load_state() -> dict[str, str]:
    if not PAPER_STATE_PATH.exists():
        return {}

    state = pd.read_csv(PAPER_STATE_PATH)

    if state.empty:
        return {}

    if not {"key", "value"}.issubset(state.columns):
        raise ValueError(
            f"Invalid state file: {PAPER_STATE_PATH}. "
            "Expected columns: key,value"
        )

    return {
        str(row["key"]): str(row["value"])
        for _, row in state.iterrows()
    }


def current_period() -> str:
    return pd.Timestamp.today().strftime("%Y-%m")


def choose_mode(
    state: dict[str, str],
    force_mark: bool = False,
    force_rebalance: bool = False,
) -> str:
    if force_mark and force_rebalance:
        raise ValueError("Cannot use --force-mark and --force-rebalance together.")

    if force_mark:
        return "mark"

    if force_rebalance:
        return "rebalance"

    last_rebalance_period = state.get("last_rebalance_period", "")

    # First ever run: build full signal set and open the paper book.
    if not last_rebalance_period:
        return "rebalance"

    # New calendar month: run the full model refresh.
    # paper_runner.py still prevents duplicate rebalances using paper_state.csv.
    if current_period() > last_rebalance_period:
        return "rebalance"

    return "mark"


# ============================================================
# EXECUTION HELPERS
# ============================================================

def run_script(script_path: Path, label: str, dry_run: bool = False) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script for {label}: {script_path}")

    print("\n" + "=" * 80)
    print(f"RUNNING: {label}")
    print(f"SCRIPT:  {script_path}")
    print("=" * 80)

    if dry_run:
        print("[DRY RUN] Skipping execution.")
        return

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=COMMODITY_ROOT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"\nFAILED: {label}\n"
            f"Script: {script_path}\n"
            f"Exit code: {result.returncode}"
        )


def open_dashboard(no_open: bool = False, dry_run: bool = False) -> None:
    if no_open:
        print("\nDashboard opening skipped because --no-open was used.")
        return

    if dry_run:
        print("\n[DRY RUN] Would open dashboard:")
        print(DASHBOARD_HTML_PATH)
        return

    if not DASHBOARD_HTML_PATH.exists():
        raise FileNotFoundError(
            f"Dashboard HTML was not created: {DASHBOARD_HTML_PATH}"
        )

    webbrowser.open(DASHBOARD_HTML_PATH.resolve().as_uri())

    print("\nOpened dashboard:")
    print(DASHBOARD_HTML_PATH)


# ============================================================
# MAIN ROUTINE
# ============================================================

def run_mark_update(dry_run: bool = False) -> None:
    """
    Normal checkup mode.

    This updates commodity prices only, then marks the paper book to market,
    then rebuilds the dashboard.

    It should not rebalance unless paper_runner detects something unusual.
    """
    run_script(DATA_SCRIPT, "price update / data.py", dry_run=dry_run)
    run_script(PAPER_RUNNER_SCRIPT, "paper account update", dry_run=dry_run)
    run_script(PAPER_DISPLAY_SCRIPT, "paper display refresh", dry_run=dry_run)


def run_rebalance_update(dry_run: bool = False) -> None:
    """
    Monthly rebalance mode.

    This runs the full research/strategy pipeline, then lets paper_runner
    execute the latest target weights if a new signal month is available.
    """
    run_script(FULL_PIPELINE_SCRIPT, "full monthly pipeline / backtest_runner.py", dry_run=dry_run)
    run_script(PAPER_RUNNER_SCRIPT, "paper account update", dry_run=dry_run)
    run_script(PAPER_DISPLAY_SCRIPT, "paper display refresh", dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command paper trading check/update for the commodity system."
    )

    parser.add_argument(
        "--force-mark",
        action="store_true",
        help="Force mark-to-market mode even if a new month is detected.",
    )

    parser.add_argument(
        "--force-rebalance",
        action="store_true",
        help="Force full monthly pipeline mode.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing scripts.",
    )

    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the dashboard in the browser.",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("COMMODITY PAPER CHECK")
    print("=" * 80)
    print(f"Project root:          {COMMODITY_ROOT}")
    print(f"Paper trading folder:  {PAPER_TRADING_DIR}")
    print(f"Python executable:     {sys.executable}")

    state = load_state()

    last_rebalance_period = state.get("last_rebalance_period", "NONE")
    last_signal_date = state.get("last_signal_date", "NONE")
    last_run_date = state.get("last_run_date", "NONE")

    mode = choose_mode(
        state=state,
        force_mark=args.force_mark,
        force_rebalance=args.force_rebalance,
    )

    print("")
    print("STATE")
    print("-" * 80)
    print(f"Current calendar month: {current_period()}")
    print(f"Last rebalance period:  {last_rebalance_period}")
    print(f"Last signal date:       {last_signal_date}")
    print(f"Last run date:          {last_run_date}")
    print("")
    print(f"MODE SELECTED:          {mode.upper()}")

    if mode == "rebalance":
        print("")
        print("Planned action:")
        print("1. Run full monthly pipeline")
        print("2. Update paper account")
        print("3. Rebuild display")
        print("4. Open dashboard")
        run_rebalance_update(dry_run=args.dry_run)

    elif mode == "mark":
        print("")
        print("Planned action:")
        print("1. Update prices")
        print("2. Mark paper account to market")
        print("3. Rebuild display")
        print("4. Open dashboard")
        run_mark_update(dry_run=args.dry_run)

    else:
        raise ValueError(f"Unknown mode selected: {mode}")

    open_dashboard(
        no_open=args.no_open,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 80)
    print("CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()