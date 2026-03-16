# main/run_comparison.py
import argparse
import os
import sys
import shlex
import time
import subprocess
import re
import csv
from datetime import datetime

# ---------------- Script Mapping ---------------- #
SCRIPT_MAP = {
    "cmu": "fixed_arrival_rate_cmu.py",
    "cmuq": "fixed_arrival_rate_cmuq.py",
    "max_pressure": "fixed_arrival_rate_max_pressure.py",
}

# ---------------- Metric Parsing Regex ---------------- #
METRIC_PATTERNS = {
    "queue_len_mean": re.compile(r"queue length mean \(overall\):\s*([\-0-9.eE]+)"),
    "queue_len_std":  re.compile(r"queue length std\s*\(overall\):\s*([\-0-9.eE]+)"),
    "queue_len_se":   re.compile(r"queue length se\s*\(overall\):\s*([\-0-9.eE]+)"),
    "cost_mean":      re.compile(r"test cost mean:\s*([\-0-9.eE]+)"),
    "cost_std":       re.compile(r"test cost std\s*:\s*([\-0-9.eE]+)"),
    "cost_se":        re.compile(r"test cost se\s*:\s*([\-0-9.eE]+)"),
}

def parse_args():
    p = argparse.ArgumentParser(
        description="Run one experiment with <script_key> <env_stub>."
    )
    # Required arguments
    p.add_argument("script_key", type=str, help="Script key, e.g. cmu")
    p.add_argument("env_stub", type=str, help="Env stub, e.g. n_model_10")

    # Optional arguments
    p.add_argument("-m", "--model", type=str, default="ppg_singlebatch.yaml", help="Model yaml filename")
    p.add_argument("-n", "--experiment_name", type=str, default=None, help="Experiment name override")

    # Directories
    p.add_argument("--scripts_dir", type=str, default="configs/scripts")
    p.add_argument("--envs_dir", type=str, default="configs/env")
    p.add_argument("--logs_dir", type=str, default="results/cla")  # Only store CSV
    return p.parse_args()

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def append_csv_row(csv_path, row_dict):
    header = [
        "timestamp", "experiment_name",
        "Delta_T",
        "queue_len_mean", "queue_len_std", "queue_len_se",
        "cost_mean", "cost_std", "cost_se",
    ]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, "") for k in header})

def main():
    args = parse_args()

    # Script
    script_key = args.script_key.strip()
    if script_key not in SCRIPT_MAP:
        raise SystemExit(f"Unknown script key: {script_key}")
    script_file = SCRIPT_MAP[script_key]
    script_path = os.path.join(args.scripts_dir, script_file)
    if not os.path.isfile(script_path):
        raise SystemExit(f"Script not found: {script_path}")

    # Environment
    env_stub = args.env_stub.strip()
    env_file = f"{env_stub}.yaml"
    env_path = os.path.join(args.envs_dir, env_file)
    if not os.path.isfile(env_path):
        raise SystemExit(f"Env file not found: {env_path}")

    # Experiment name
    experiment_name = args.experiment_name or f"{env_stub}_{script_key}"

    # CSV path
    ensure_dirs(args.logs_dir)
    csvfile = os.path.join(args.logs_dir, "exp.csv")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Run command —— Use current interpreter + unbuffered
    py = shlex.quote(sys.executable)       # Ensure it is the Python in your current conda environment
    cmd_script = shlex.quote(script_path)
    command = f"{py} -u {cmd_script} -m={args.model} -e={env_file} -experiment_name={experiment_name}"
    print(f"[INFO] Running: {command}")

    metrics = {k: None for k in METRIC_PATTERNS.keys()}
    start_time = time.time()

    # Pass environment variables to ensure unbuffered and UTF-8
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )
    try:
        for line in iter(proc.stdout.readline, ''):
            print(line, end='')
            for k, pat in METRIC_PATTERNS.items():
                m = pat.search(line)
                if m:
                    try:
                        metrics[k] = float(m.group(1))
                    except ValueError:
                        pass
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()

    elapsed = time.time() - start_time
    print(f"[DONE] {command} finished in {elapsed:.2f} seconds (rc={proc.returncode})")

    # If the child process fails, do not write to CSV, exit directly (avoid dirty data)
    if proc.returncode != 0:
        print("[ERROR] Child process failed. Skip logging to CSV. "
              "Likely wrong python env before this fix; ensure torch is importable.")
        sys.exit(proc.returncode)

    # Write CSV (only when the child process succeeds)
    append_csv_row(csvfile, {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "Delta_T": f"{elapsed:.2f}",
        **metrics,
    })
    print(f"[LOGGED] csv row -> {csvfile}")

if __name__ == "__main__":
    main()