#!/bin/bash
#SBATCH -J qgym_gpu_td_n_model
#SBATCH -o slurm-%j.out
#SBATCH --time=09:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --chdir=/scratch/jf5524/QGymGPU   # 保底：作业启动时的 CWD 就在项目根

set -euo pipefail
set -x
module purge

# ===== Python 解释器（不激活环境，直接用前缀）=====
PY=/scratch/jf5524/conda/pt-cu121/bin/python
which "$PY"
"$PY" -V

# （可选）部分站点需要设置这些
export CONDA_PREFIX=/scratch/jf5524/conda/pt-cu121
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# ========= 路径 =========
PROJ=/scratch/jf5524/QGymGPU
LOGDIR=${PROJ}/logs
mkdir -p "$LOGDIR"

# 独立运行目录（用于中间文件/缓存），但注意：跑实验时会切回 $PROJ
RUNDIR=${SCRATCH:-/scratch/jf5524}/QGymGPU/run-${SLURM_JOB_ID/.*}
mkdir -p "$RUNDIR"
cd "$RUNDIR"

echo "Host: $(hostname)"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
echo "PYTHONPATH=$PYTHONPATH"

# ====== PyTorch/CUDA 自检 ======
"$PY" - <<'PY'
try:
    import torch
    print("torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available(), "device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("current device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))
except Exception as e:
    print("PYTORCH_CHECK_FAILED:", e)
PY

# ====== 小工具函数 ======
warn(){ echo "[WARN] $*" >&2; }
info(){ echo "[INFO] $*"; }

run() {
  local script_key="$1"
  local env_stub="$2"
  info "START $(date +'%F %T')  CWD=$(pwd)  run: ${script_key} ${env_stub}"
  # 关键：切回项目根再跑，确保相对路径可用
  ( cd "$PROJ" && \
    "$PY" -m main.run_comparison "${script_key}" "${env_stub}" \
      # 如需显式传绝对路径，也可以取消注释这三行：
      # --scripts_dir "${PROJ}/configs/scripts" \
      # --envs_dir    "${PROJ}/configs/env" \
      # --logs_dir    "${PROJ}/results/cla"
  ) || warn "experiment failed: ${script_key} ${env_stub}"
  info "END   $(date +'%F %T')  run: ${script_key} ${env_stub}"
}

# ====== 串行跑多个实验（不需要 srun）======
run cmu          n_model_mm_10
run cmuq         n_model_mm_10
run max_pressure n_model_mm_10

run cmu          n_model_mm_50
run cmuq         n_model_mm_50
run max_pressure n_model_mm_50

run cmu          n_model_mm_100
run cmuq         n_model_mm_100
run max_pressure n_model_mm_100

run cmu          n_model_mm_200
run cmuq         n_model_mm_200
run max_pressure n_model_mm_200

run cmu          en_model_mm_10
run cmuq         en_model_mm_10
run max_pressure en_model_mm_10

run cmu          en_model_mm_50
run cmuq         en_model_mm_50
run max_pressure en_model_mm_50

run cmu          en_model_mm_100
run cmuq         en_model_mm_100
run max_pressure en_model_mm_100

run cmu          en_model_mm_200
run cmuq         en_model_mm_200
run max_pressure en_model_mm_200

run cmu          reentrant_2
run cmuq         reentrant_2
run max_pressure reentrant_2

# 如果以后要并行跑（同一作业里开多个进程），改成：
# srun --exclusive -N1 -n1 bash -lc 'cd "$PROJ" && '"$PY"' -m main.run_comparison cmu  n_model_mm_50' &
# srun --exclusive -N1 -n1 bash -lc 'cd "$PROJ" && '"$PY"' -m main.run_comparison cmuq n_model_mm_50' &
# wait
