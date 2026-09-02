#!/bin/bash
# Sequentially evaluate skill_id from START_ID to END_ID.
#
# Usage:
#   bash benchmark/scripts/run_all_skills.sh [start_skill_id] [end_skill_id] [options]
#
# Options:
#   --task <name>          task name
#   --port <port>          websocket port (default: 8001)
#   --host <host>          websocket host (default: 127.0.0.1)
#   --log_dir <path>       log output directory
#   --manifest <path>      path to state_manifest.json
#   --eval_config <path>   path to skill_eval_configs/*.yaml (per-episode in multi-trajectory scenarios)
#   --prompt <text>        override the instruction of all skills (passed through to skill_eval.py --prompt)
#   --n_episodes <N>       evaluate each skill N times (default 1; multi-run statistics are controlled by the outer num_runs)
#
# Special: END_ID = -1 reads the maximum skill_id from the manifest automatically
#
# Examples:
#   bash benchmark/scripts/run_all_skills.sh 0 16
#   bash benchmark/scripts/run_all_skills.sh 5 10 --task setting_the_fire --port 8002
#   bash benchmark/scripts/run_all_skills.sh 0 -1 --manifest /path/to/state_manifest.json --eval_config /path/to/episode_X.yaml

set -u

# ── Path resolution (based on the script's own location, independent of cwd) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${BENCH_DIR}")"   # parent of benchmark/ (package import root)

# ── Positional args: start/end skill_id ────────────────────────────────────
START_ID=${1:-0}
END_ID=${2:-16}

shift 2 2>/dev/null || true

# ── Defaults ────────────────────────────────────────────────────────────────
TASK="task name"
WS_HOST="127.0.0.1"
WS_PORT="8001"
LOG_DIR="path_to_log"
MANIFEST="path_to/skill_init_states/task-xxxx/state_manifest.json"
EVAL_CONFIG=""
PROMPT=""
N_EPISODES=1
SKILL_IDS=""   # comma-separated skill ids; empty = evaluate START_ID..END_ID in order (e.g. "2,5,6")
SKIP_EXISTING=false   # true = skip skills whose result file already exists


# ── Parse named arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK="$2"; shift 2 ;;
        --port)
            WS_PORT="$2"; shift 2 ;;
        --host)
            WS_HOST="$2"; shift 2 ;;
        --log_dir)
            LOG_DIR="$2"; shift 2 ;;
        --manifest)
            MANIFEST="$2"; shift 2 ;;
        --eval_config)
            EVAL_CONFIG="$2"; shift 2 ;;
        --prompt)
            PROMPT="$2"; shift 2 ;;
        --n_episodes)
            N_EPISODES="$2"; shift 2 ;;
        --skill_ids)
            SKILL_IDS="$2"; shift 2 ;;
        --skip_existing)
            SKIP_EXISTING=true; shift 1 ;;
        *)
            echo "[ERROR] Unknown option: $1"
            echo "Usage: bash $0 [start_id] [end_id] [--task NAME] [--port N] [--host H] [--log_dir PATH] [--manifest PATH] [--eval_config PATH]"
            exit 1 ;;
    esac
done

# ── END_ID=-1: read the maximum skill_id from the manifest ────────────────
if [ "${END_ID}" = "-1" ]; then
    if [ ! -f "${MANIFEST}" ]; then
        echo "[ERROR] END_ID=-1 requires reading ${MANIFEST}, but the file does not exist"
        exit 1
    fi
    END_ID=$(python -c "import json; m=json.load(open('${MANIFEST}')); print(max(s['skill_idx'] for s in m['skills']))")
    echo "[INFO] END_ID=${END_ID} read automatically from the manifest"
fi

mkdir -p "${LOG_DIR}"
RUN_LOG_DIR="${LOG_DIR}/run_logs"
mkdir -p "${RUN_LOG_DIR}"

echo "==========================================================="
echo "Running skills ${START_ID} -> ${END_ID}"
echo "Task:        ${TASK}"
echo "Host:Port    ${WS_HOST}:${WS_PORT}"
echo "Log dir:     ${LOG_DIR}"
echo "Manifest:    ${MANIFEST}"
echo "Eval config: ${EVAL_CONFIG:-<auto>}"
echo "==========================================================="

# build optional arguments
EXTRA_ARGS=()
[ -n "${EVAL_CONFIG}" ] && EXTRA_ARGS+=(--eval_config "${EVAL_CONFIG}")
[ -n "${PROMPT}" ]      && EXTRA_ARGS+=(--prompt "${PROMPT}")
EXTRA_ARGS+=(--n_episodes "${N_EPISODES}")

# build the list of skill ids to evaluate
if [ -n "${SKILL_IDS}" ]; then
    # --skill_ids given (comma-separated): convert to an array
    IFS=',' read -ra SKILL_ID_LIST <<< "${SKILL_IDS}"
else
    # otherwise generate START_ID..END_ID in order
    SKILL_ID_LIST=($(seq "${START_ID}" "${END_ID}"))
fi

for SKILL_ID in "${SKILL_ID_LIST[@]}"; do
    STAMP=$(date +"%Y%m%d_%H%M%S")
    RUN_LOG="${RUN_LOG_DIR}/skill_${SKILL_ID}_${STAMP}.log"
    RESULT_FILE="${LOG_DIR}/${TASK}_skill_$(printf '%02d' "${SKILL_ID}").json"

    # --skip_existing: skip if the result file already exists
    if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${RESULT_FILE}" ]; then
        echo ""
        echo "----- [$(date '+%F %T')] SKIP skill_id=${SKILL_ID} (result already exists: ${RESULT_FILE})-----"
        continue
    fi

    echo ""
    echo "----- [$(date '+%F %T')] Starting skill_id=${SKILL_ID} -----"
    echo "Log: ${RUN_LOG}"

    python -m benchmark.skill_eval \
        --task "${TASK}" \
        --skill_id "${SKILL_ID}" \
        --websocket_host "${WS_HOST}" --websocket_port "${WS_PORT}" \
        --log_dir "${LOG_DIR}" \
        --manifest "${MANIFEST}" \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "${RUN_LOG}"

    STATUS=${PIPESTATUS[0]}
    if [ "${STATUS}" -ne 0 ]; then
        echo "[WARN] skill_id=${SKILL_ID} exited with status ${STATUS}. Continuing to next skill."
    else
        echo "[OK] skill_id=${SKILL_ID} finished."
    fi
done

echo ""
echo "==========================================================="
echo "All skills finished: ${SKILL_IDS:-${START_ID}..${END_ID}}"
echo "==========================================================="
