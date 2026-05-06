#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/datastor1/jiahuikchen/X_IL}
VENV_ACTIVATE=${VENV_ACTIVATE:-${ROOT_DIR}/.venv/bin/activate}
LOG_DIR=${LOG_DIR:-${ROOT_DIR}/logs/tmux}
SESSION_PREFIX=${SESSION_PREFIX:-libero_failure_only_bc}

# Keep the same seed sweep as script/bc/dec/transformer.sh, but run seeds in
# independent processes so they can occupy separate GPUs.
SEEDS=(${SEEDS:-0 1 2})
GPU_IDS=(${GPU_IDS:-0 1 2})

mkdir -p "${LOG_DIR}"

if [[ ${#GPU_IDS[@]} -lt ${#SEEDS[@]} ]]; then
  echo "Need at least as many GPU_IDS as SEEDS" >&2
  exit 1
fi

for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  gpu="${GPU_IDS[$idx]}"
  session="${SESSION_PREFIX}_failure_only_s${seed}"
  log="${LOG_DIR}/${session}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 1
  fi

  tmux new-session -d -s "${session}" \
    "bash -lc 'cd ${ROOT_DIR} && source ${VENV_ACTIVATE} && export HYDRA_FULL_ERROR=1 && export WANDB_PROJECT=libero_failure_data && export CUDA_VISIBLE_DEVICES=${gpu} && python run.py --config-name=libero_failure_config --multirun agents=bc_agent agent_name=bc_transformer_failure_only group=bc_decoder_only_failure_only run_group=bc_decoder_only_failure_only agents/model=bc/bc_dec_transformer trainset.include_successful=False trainset.include_failed=True valset.include_successful=False valset.include_failed=True seed=${seed} 2>&1 | tee ${log}'"

  echo "started ${session} on GPU ${gpu}"
  echo "log ${log}"
done
