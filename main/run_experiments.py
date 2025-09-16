import argparse
import os
import yaml
import time
import subprocess

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-exp_dir', type=str)


args = parser.parse_args()


exp_configs = []

for file in os.listdir(f'configs/experiments/{args.exp_dir}'):
    if file.endswith('.yaml'):
        with open(f'configs/experiments/{args.exp_dir}/{file}', 'r') as f:
            exp_configs.append(yaml.safe_load(f))

processes = []
for exp_config in exp_configs:
    script_file = exp_config['script']
    model_file = exp_config['model']
    env_file = exp_config['env']
    experiment_name = exp_config['experiment_name']

    command = f"python configs/scripts/{script_file} -m={model_file} -e={env_file} -experiment_name={experiment_name}"
    print(f"[INFO] Running: {command}")
    start_time = time.time()

    process = subprocess.Popen(command, shell=True)
    processes.append((process, command, start_time))

for process, command, start_time in processes:
    process.wait()
    elapsed = time.time() - start_time
    print(f"[DONE] {command} finished in {elapsed:.2f} seconds")

