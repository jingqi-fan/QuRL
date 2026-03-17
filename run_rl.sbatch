#!/bin/bash
#SBATCH -J QuRL_gpu_rl_n_model
#SBATCH -o slurm-%j.out
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --chdir=/scratch/jf5524/QuRL   # Ensure the job starts from the project root

set -euo pipefail
set -x
module purge

# ===== Python interpreter (use the environment prefix directly, no activation) =====
PY=/scratch/jf5524/conda/py310/bin/python
which "$PY"
"$PY" -V

# (Optional) Some clusters require these variables
export CONDA_PREFIX=/scratch/jf5524/conda/py310
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# ========= Paths =========
PROJ=/scratch/jf5524/QuRL
LOGDIR=${PROJ}/logs
mkdir -p "$LOGDIR"

# Independent run directory (for temporary files / cache)
# Note: experiments will switch back to $PROJ when running
RUNDIR=${SCRATCH:-/scratch/jf5524}/QuRL/run-${SLURM_JOB_ID/.*}
mkdir -p "$RUNDIR"
cd "$RUNDIR"

echo "Host: $(hostname)"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
echo "PYTHONPATH=$PYTHONPATH"

# ====== PyTorch / CUDA check ======
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
  local env_stub="$1"   # e.g. vanilla_bc.yaml
  local tag="$2"        # e.g. n_model_mm_200
  info "START $(date +'%F %T')  CWD=$(pwd)  run: ${env_stub} ${tag}"
  # Switch back to project root to ensure relative paths work
  ( cd "$PROJ" && \
    "$PY" -m RL.PPO.train "${env_stub}" "${tag}" \
      # If absolute paths are required, add them if supported by RL.algorithms.train:
      # --envs_dir    "${PROJ}/configs/env" \
      # --logs_dir    "${PROJ}/results/cla"
  ) || warn "experiment failed: ${env_stub} ${tag}"
  info "END   $(date +'%F %T')  run: ${env_stub} ${tag}"
}

run_multi_env() {
  local env_stub="$1"   # e.g. vanilla_bc.yaml
  local tag="$2"        # e.g. n_model_mm_200
  info "START $(date +'%F %T')  CWD=$(pwd)  run: ${env_stub} ${tag}"
  # Switch back to project root to ensure relative paths work
  ( cd "$PROJ" && \
    "$PY" -m RL.PPO_multi_env.train "${env_stub}" "${tag}" \
  ) || warn "experiment failed: ${env_stub} ${tag}"
  info "END   $(date +'%F %T')  run: ${env_stub} ${tag}"
}


# ====== Run multiple experiments sequentially (no srun needed) ======

run WC.yaml          en_model_mm_10
run WC.yaml          en_model_mm_50
run WC.yaml          en_model_mm_100
run WC.yaml          en_model_mm_200