#!/usr/bin/env python3
import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_step(step_name, script_path):
    print(f"\n{'=' * 60}")
    print(f"  STEP: {step_name}")
    print(f"{'=' * 60}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {step_name} failed with exit code {result.returncode}")
        sys.exit(1)
    print(f"[OK] {step_name} completed successfully.\n")


def main():
    print("=" * 60)
    print("  CUSTOMER CHURN PREDICTION PIPELINE")
    print("=" * 60)

    run_step("1. Generate Synthetic Dataset", os.path.join(PROJECT_ROOT, "src", "generate_data.py"))

    run_step("2. Train & Evaluate Models", os.path.join(PROJECT_ROOT, "src", "modeling.py"))

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Dataset: {PROJECT_ROOT}/data/telco_churn.csv")
    print(f"  Models:  {PROJECT_ROOT}/models/")
    print(f"  Reports: {PROJECT_ROOT}/reports/")
    print(f"\n  To launch the dashboard, run:")
    print(f"    streamlit run {PROJECT_ROOT}/app/streamlit_app.py")
    print()


if __name__ == "__main__":
    main()
