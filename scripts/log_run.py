#!/usr/bin/env python3
"""Append a result row to results/all_runs.csv.

Single source of truth for the results schema: the CSV header. Any column may be
passed as --key value; unknown keys are rejected so typos do not silently create
columns. Missing columns are left blank.

Examples
--------
# log a planning eval result
python scripts/log_run.py --variant official_lewm --benchmark pusht --phase 1 \
    --success 0.62 --return 0.71 --plan_latency_s 0.9 --plan_samples 300 \
    --train_epochs 100 --seed 42 --ckpt_path pusht/lewm

# log offline latent metrics (typically written by eval_latent_metrics.py)
python scripts/log_run.py --variant gated_spherical --benchmark pusht --phase 3 \
    --roll_err_1 0.02 --roll_err_20 0.21 --fut_r1 0.55 --eff_rank 84 --clumping 0.06
"""
import argparse
import csv
import datetime as dt
import sys
import uuid
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "results" / "all_runs.csv"


def header():
    with CSV.open() as f:
        return next(csv.reader(f))


def main(argv=None):
    cols = header()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for c in cols:
        p.add_argument(f"--{c}", default="")
    args = vars(p.parse_args(argv))

    if not args.get("run_id"):
        args["run_id"] = uuid.uuid4().hex[:8]
    if not args.get("timestamp"):
        args["timestamp"] = dt.datetime.now().isoformat(timespec="seconds")

    row = [args.get(c, "") for c in cols]
    with CSV.open("a", newline="") as f:
        csv.writer(f).writerow(row)
    print(f"logged run {args['run_id']}: variant={args.get('variant')} "
          f"benchmark={args.get('benchmark')} -> {CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
