#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="/home/Mingfan/wjn/CVTSP"
TRAIN_PID="866930"
TRAIN_DIR="${PROJECT_ROOT}/outputs/train__multibound_block1_nm1pct_from_scratch_e200_w96_gpu6_20260831"
BASELINE_CKPT="${PROJECT_ROOT}/outputs/train__cvtsp_random_n100_nearest_quarter/checkpoint-275.pt"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/post_training_eval_epoch200_20260901"

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
    python3 - "${TRAIN_DIR}/train_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
epoch = int(float(summary.get("last_metrics", {}).get("epoch", -1)))
raise SystemExit(0 if epoch == 200 else 1)
PY
}

log "Waiting for the 200-epoch training run to finish."
while ! training_complete; do
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
        log "ERROR: training process ${TRAIN_PID} ended before a valid epoch-200 summary was written."
        exit 2
    fi
    sleep 60
done

BEST_CKPT="${TRAIN_DIR}/best.pt"
LAST_CKPT="${TRAIN_DIR}/last.pt"

python3 - "${BEST_CKPT}" "${LAST_CKPT}" <<'PY'
import sys
import torch

for path in sys.argv[1:]:
    checkpoint = torch.load(path, map_location="cpu")
    print(f"checkpoint={path} epoch={int(checkpoint.get('epoch', -1))}", flush=True)
if int(torch.load(sys.argv[2], map_location="cpu").get("epoch", -1)) != 200:
    raise SystemExit("last.pt is not epoch 200")
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
        CUDA_VISIBLE_DEVICES="${gpu}" python3 -u test_n100.py \
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
    local label="$1"
    local checkpoint="$2"
    local gpu="$3"
    local output_dir="${OUTPUT_ROOT}/synthetic5000_${label}"
    mkdir -p "${output_dir}"
    log "Starting synthetic5000 ${label} versus epoch275 on physical GPU ${gpu}."
    (
        cd "${PROJECT_ROOT}" || exit 1
        CUDA_VISIBLE_DEVICES="${gpu}" python3 -u benchmark_synthetic_checkpoints.py \
            --baseline "${BASELINE_CKPT}" \
            --model "${label}=${checkpoint}" \
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
    log "Finished synthetic5000 ${label}, exit=${status}."
    return "${status}"
}

run_lane() {
    local label="$1"
    local checkpoint="$2"
    local gpu="$3"
    local status=0
    run_real529 "${label}" "${checkpoint}" "${gpu}" || status=$?
    run_synthetic5000 "${label}" "${checkpoint}" "${gpu}" || status=$?
    return "${status}"
}

# Training releases GPU 6 before this point. Two independent lanes run in
# parallel on GPUs 4 and 6; GPUs 0-3 are never used.
run_lane best "${BEST_CKPT}" 4 &
BEST_LANE_PID=$!
run_lane last "${LAST_CKPT}" 6 &
LAST_LANE_PID=$!

BEST_STATUS=0
LAST_STATUS=0
wait "${BEST_LANE_PID}" || BEST_STATUS=$?
wait "${LAST_LANE_PID}" || LAST_STATUS=$?

python3 - "${OUTPUT_ROOT}" "${BEST_STATUS}" "${LAST_STATUS}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "lane_exit_status": {"best": int(sys.argv[2]), "last": int(sys.argv[3])},
    "experiments": {},
}
for label in ("best", "last"):
    real_path = root / f"real529_{label}" / "all_summary.json"
    synth_path = root / f"synthetic5000_{label}" / "results.json"
    payload["experiments"][f"real529_{label}"] = (
        json.load(open(real_path, encoding="utf-8")) if real_path.exists() else None
    )
    payload["experiments"][f"synthetic5000_{label}"] = (
        json.load(open(synth_path, encoding="utf-8")).get("overall")
        if synth_path.exists()
        else None
    )
(root / "combined_summary.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
print(json.dumps(payload, indent=2), flush=True)
PY

log "All four post-training experiments finished. best_lane=${BEST_STATUS}, last_lane=${LAST_STATUS}."
exit $(( BEST_STATUS != 0 || LAST_STATUS != 0 ))
