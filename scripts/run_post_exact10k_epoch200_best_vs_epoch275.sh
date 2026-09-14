#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="/home/Mingfan/wjn/CVTSP"
PYTHON_BIN="/home/Mingfan/.conda/envs/macsim/bin/python"
TRAIN_DIR="${PROJECT_ROOT}/outputs/train__exact10k_targetreward_macsimlr_from_scratch_e200_w96_gpu7_20260904"
EPOCH275_CKPT="${PROJECT_ROOT}/outputs/train__cvtsp_random_n100_nearest_quarter/checkpoint-275.pt"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/post_exact10k_epoch200_best_vs_epoch275_20260909"

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${OUTPUT_ROOT}/watcher.log") 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

training_complete() {
    [[ -f "${TRAIN_DIR}/train_summary.json" ]] || return 1
    [[ -f "${TRAIN_DIR}/best.pt" ]] || return 1
    [[ -f "${TRAIN_DIR}/last.pt" ]] || return 1
    [[ -f "${TRAIN_DIR}/checkpoint-200.pt" ]] || return 1
    "${PYTHON_BIN}" - "${TRAIN_DIR}/last.pt" "${TRAIN_DIR}/checkpoint-200.pt" <<'PY'
import sys
import torch

last = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
final = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
raise SystemExit(0 if int(last.get("epoch", -1)) == 200 and int(final.get("epoch", -1)) == 200 else 1)
PY
}

log "Waiting for the exact10k 200-epoch training run to finish."
while ! training_complete; do
    if ! pgrep -f '[t]rain_joint_self_improvement.py.*train__exact10k_targetreward_macsimlr_from_scratch_e200_w96_gpu7_20260904' >/dev/null; then
        log "ERROR: training ended before a valid Epoch 200 checkpoint was written."
        exit 2
    fi
    sleep 60
done

BEST_CKPT="${TRAIN_DIR}/best.pt"
"${PYTHON_BIN}" - "${EPOCH275_CKPT}" "${BEST_CKPT}" <<'PY'
import sys
import torch

for path in sys.argv[1:]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    print(
        f"checkpoint={path} epoch={int(checkpoint.get('epoch', -1))} "
        f"best_metric={float(checkpoint.get('best_metric', float('nan'))):.6f}",
        flush=True,
    )
PY

run_real529() {
    local label="$1"
    local checkpoint="$2"
    local gpu="$3"
    local output_dir="${OUTPUT_ROOT}/real529_${label}"
    mkdir -p "${output_dir}"
    log "Starting Real529 ${label} on physical GPU ${gpu}."
    (
        cd "${PROJECT_ROOT}" || exit 1
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u test_n100.py \
            --mode real529 \
            --split all \
            --checkpoint-path "${checkpoint}" \
            --output-dir "${output_dir}" \
            --batch-size 64 \
            --decode-type greedy \
            --augmentation \
            --aug-factor 8 \
            --gurobi-threads 1 \
            --seed 1234 \
            > "${output_dir}/live.log" 2>&1
    )
    local status=$?
    log "Finished Real529 ${label}, exit=${status}."
    return "${status}"
}

run_synthetic5000() {
    local output_dir="${OUTPUT_ROOT}/synthetic5000_epoch275_vs_best"
    mkdir -p "${output_dir}"
    log "Starting paired synthetic5000 comparison on physical GPU 6."
    (
        cd "${PROJECT_ROOT}" || exit 1
        CUDA_VISIBLE_DEVICES=6 "${PYTHON_BIN}" -u benchmark_synthetic_checkpoints.py \
            --baseline "${EPOCH275_CKPT}" \
            --baseline-label epoch275 \
            --model "best=${BEST_CKPT}" \
            --output-dir "${output_dir}" \
            --sizes 20 40 60 80 100 \
            --instances-per-size 1000 \
            --batch-size 64 \
            --workers 16 \
            --device cuda:0 \
            --seed 20260827 \
            > "${output_dir}/live.log" 2>&1
    )
    local status=$?
    log "Finished paired synthetic5000 comparison, exit=${status}."
    return "${status}"
}

# Physical GPUs 0-3 are intentionally excluded.
run_real529 epoch275 "${EPOCH275_CKPT}" 4 &
REAL275_PID=$!
run_real529 best "${BEST_CKPT}" 5 &
REALBEST_PID=$!
run_synthetic5000 &
SYNTH_PID=$!

REAL275_STATUS=0
REALBEST_STATUS=0
SYNTH_STATUS=0
wait "${REAL275_PID}" || REAL275_STATUS=$?
wait "${REALBEST_PID}" || REALBEST_STATUS=$?
wait "${SYNTH_PID}" || SYNTH_STATUS=$?

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${REAL275_STATUS}" "${REALBEST_STATUS}" "${SYNTH_STATUS}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

payload = {
    "exit_status": {
        "real529_epoch275": int(sys.argv[2]),
        "real529_best": int(sys.argv[3]),
        "synthetic5000": int(sys.argv[4]),
    },
    "real529_epoch275": read_json(root / "real529_epoch275" / "all_summary.json"),
    "real529_best": read_json(root / "real529_best" / "all_summary.json"),
    "synthetic5000": None,
}
synth = read_json(root / "synthetic5000_epoch275_vs_best" / "results.json")
if synth is not None:
    payload["synthetic5000"] = synth.get("overall")
(root / "combined_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2), flush=True)
PY

log "All post-training comparisons finished: real275=${REAL275_STATUS}, realbest=${REALBEST_STATUS}, synthetic=${SYNTH_STATUS}."
exit $(( REAL275_STATUS != 0 || REALBEST_STATUS != 0 || SYNTH_STATUS != 0 ))
