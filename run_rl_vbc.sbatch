#!/bin/bash
#SBATCH -J qgym_gpu_rl
#SBATCH -o slurm-%j.out
#SBATCH --time=07:00:00        # 运行时间
#SBATCH --ntasks=1             # 一个任务
#SBATCH --cpus-per-task=16      # 每个任务 6 核
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx8000:1


set -euo pipefail
set -x
module purge

# ===== 使用 conda 前缀环境的 python，不做激活 =====
PY=/scratch/jf5524/conda/pt-cu121/bin/python
which "$PY"
"$PY" -V

# （可选）有些集群需要这两行来找动态库
export CONDA_PREFIX=/scratch/jf5524/conda/pt-cu121
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ========= 路径与运行目录 =========
PROJ=/scratch/jf5524/QGymGPU
mkdir -p /scratch/jf5524/QGymGPU/logs
RUNDIR=${SCRATCH:-/scratch/jf5524}/QGymGPU/run-${SLURM_JOB_ID/.*}
mkdir -p "$RUNDIR"
cd "$RUNDIR"

echo "Host: $(hostname)"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
#export PYTHONPATH="$PROJ/RL/PPO:$PROJ:${PYTHONPATH:-}"

echo "PYTHONPATH=$PYTHONPATH"

# 可选：PyTorch/CUDA 自检
"$PY" - <<'PY'
try:
    import torch
    print("torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available(), "device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("current device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))
except Exception as e:
    print("PYTORCH_CHECK_FAILED:", e)
PY

# 在 RUNDIR 内运行
#srun --cpu-bind=cores "$PY" -m main.run_experiments -exp_dir reentrant_5
# 保留前面的 RUNDIR 创建等步骤
srun --cpu-bind=cores --chdir="$PROJ" "$PY" -m RL.PPO.train vanilla_bc.yaml n_model_mm_5
srun --cpu-bind=cores --chdir="$PROJ" "$PY" -m RL.PPO.train vanilla_bc.yaml n_model_mm_10
srun --cpu-bind=cores --chdir="$PROJ" "$PY" -m RL.PPO.train vanilla_bc.yaml n_model_mm_50
srun --cpu-bind=cores --chdir="$PROJ" "$PY" -m RL.PPO.train vanilla_bc.yaml n_model_mm_100
srun --cpu-bind=cores --chdir="$PROJ" "$PY" -m RL.PPO.train vanilla_bc.yaml n_model_mm_200

# 或： srun --cpu-bind=cores "$PY" "$PROJ/main/run_experiments.py" -exp_dir reentrant_5
