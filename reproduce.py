from pathlib import Path
import argparse, subprocess, sys
ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description="Reproduce the manuscript experiments and illustrations")
p.add_argument("--experiments", action="store_true")
p.add_argument("--figures", action="store_true")
a = p.parse_args()
if not (a.experiments or a.figures):
    p.error("choose --experiments and/or --figures")
if a.experiments:
    for script in ["experiment_noise_covariance.py", "verify_algebra.py", "experiment_sobi.py"]:
        subprocess.run([sys.executable, str(ROOT/script)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT/"verify_recorded_results.py"), "--generated"], cwd=ROOT, check=True)
if a.figures:
    for i in range(1, 9):
        subprocess.run([sys.executable, str(ROOT/f"plot_figure_{i}.py")], cwd=ROOT, check=True)
