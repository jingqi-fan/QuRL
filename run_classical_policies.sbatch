#!/bin/bash
#SBATCH -J QuRL_gpu
#SBATCH -o slurm-%j.out
#SBATCH --time=9:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --chdir=/scratch/jf5524/QuRL   # Ensure the job starts from the project root

set -euo pipefail
set -x
module purge

# ===== Python interpreter (use the environment prefix directly, no activation) =====
PY=/scratch/jf5524/conda/py310/bin/python
which "$PY"
"$PY" -V

# (Optional) Some clusters require these environment variables
export CONDA_PREFIX=/scratch/jf5524/conda/py310
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# ========= Paths =========
PROJ=/scratch/jf5524/QuRL
LOGDIR=${PROJ}/logs
mkdir -p "$LOGDIR"

# Separate run directory (for temporary files / caches)
# Note: experiments will switch back to $PROJ when running
RUNDIR=${SCRATCH:-/scratch/jf5524}/QuRL/run-${SLURM_JOB_ID/.*}
mkdir -p "$RUNDIR"
cd "$RUNDIR"

echo "Host: $(hostname)"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
echo "PYTHONPATH=$PYTHONPATH"

# ====== PyTorch / CUDA self-check ======
"$PY" - <<'PY'
try:
    import torch
    print("torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available(), "device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("current device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))
except Exception as e:
    print("PYTORCH_CHECK_FAILED:", e)
PY

# ====== Helper functions ======
warn(){ echo "[WARN] $*" >&2; }
info(){ echo "[INFO] $*"; }

run() {
  local script_key="$1"
  local env_stub="$2"
  info "START $(date +'%F %T')  CWD=$(pwd)  run: ${script_key} ${env_stub}"
  # Switch back to the project root before running to ensure relative paths work
  ( cd "$PROJ" && \
    "$PY" -m main.run_comparison "${script_key}" "${env_stub}" \
      # If absolute paths are required, uncomment the following:
      # --scripts_dir "${PROJ}/configs/scripts" \
      # --envs_dir    "${PROJ}/configs/env" \
      # --logs_dir    "${PROJ}/results/cla"
  ) || warn "experiment failed: ${script_key} ${env_stub}"
  info "END   $(date +'%F %T')  run: ${script_key} ${env_stub}"
}

## ====== Run multiple experiments sequentially (no srun needed) ======
run cmu          n_model_mm_100
run cmu          n_model_mm_200
run cmu          n_model_mm_100
run cmu          n_model_mm_200

run cmu          reentrant_2
run cmuq         reentrant_2
run max_pressure reentrant_2