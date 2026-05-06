#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/datastor1/jiahuikchen/X_IL}
VENV_ACTIVATE=${VENV_ACTIVATE:-${ROOT_DIR}/.venv/bin/activate}
LOG_DIR=${LOG_DIR:-${ROOT_DIR}/logs/tmux}
SEEDS=(${SEEDS:-0 1 2})
GPU_IDS=(${GPU_IDS:-0 1 2})
DISCOUNT=${DISCOUNT:-0.5}

DISCOUNT_TAG=${DISCOUNT_TAG:-$(python3 - "${DISCOUNT}" <<'PY'
import sys

discount = float(sys.argv[1])
print(f"d{int(round(discount * 100)):03d}")
PY
)}

SESSION_PREFIX=${SESSION_PREFIX:-libero_failure_awr_${DISCOUNT_TAG}}
AGENT_NAME=${AGENT_NAME:-awr_transformer_failure_inclusive_${DISCOUNT_TAG}}
GROUP_NAME=${GROUP_NAME:-awr_decoder_only_failure_inclusive_${DISCOUNT_TAG}}
RUN_GROUP=${RUN_GROUP:-${GROUP_NAME}}

mkdir -p "${LOG_DIR}"

if [[ ${#GPU_IDS[@]} -lt ${#SEEDS[@]} ]]; then
  echo "Need at least as many GPU_IDS as SEEDS" >&2
  exit 1
fi

for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  gpu="${GPU_IDS[$idx]}"
  session="${SESSION_PREFIX}_s${seed}"
  log="${LOG_DIR}/${session}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 1
  fi

  tmux new-session -d -s "${session}" \
    "bash -lc 'cd ${ROOT_DIR} && source ${VENV_ACTIVATE} && export HYDRA_FULL_ERROR=1 && export WANDB_PROJECT=libero_failure_data && export CUDA_VISIBLE_DEVICES=${gpu} && python run.py --config-name=libero_failure_config --multirun agents=awr_agent trainers=awr_trainer agent_name=${AGENT_NAME} group=${GROUP_NAME} run_group=${RUN_GROUP} agents/model=bc/bc_dec_transformer trainset.use_returns=True valset.use_returns=True trainset.discount=${DISCOUNT} valset.discount=${DISCOUNT} seed=${seed} 2>&1 | tee ${log}'"

  echo "started ${session} on GPU ${gpu}"
  echo "log ${log}"
done
