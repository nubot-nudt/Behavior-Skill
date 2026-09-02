#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_auto_eval.sh — example multi-GPU evaluation driver (top layer)
#
# Example script for step-by-step (skill-by-skill) task evaluation on
# multiple GPUs; adapt it freely to your own setup.
#
# Pipeline:
#   1. one policy websocket server is started per GPU (one port each)
#   2. the script waits until all policy servers are ready
#   3. one evaluation worker is started per GPU (episodes split round-robin);
#      each (episode, run) is delegated to run_all_skills.sh (same directory),
#      which iterates over the skills and drives benchmark.skill_eval
#   4. after all workers finish, the policy servers are stopped
#
# Plugging in your own policy:
#   the policy service location (POLICY_ROOT, top of this script) and the
#   launch command (Step 1) are embedded as a pi0.5 example — replace them
#   with your own implementation. Any websocket server exposing the policy
#   API works; see benchmark/policies for the client side of the protocol.
#
# Prerequisites:
#   - run this script with the evaluator environment active (the benchmark
#     package importable, i.e. `python -m benchmark.skill_eval` runs, with
#     OmniGibson installed)
#   - make sure the policy server repo is runnable (dependencies installed,
#     checkpoint paths valid)
#
# Usage (either A or B):
#
# A) via a task_config JSON:
#   bash benchmark/scripts/run_auto_eval.sh \
#     --task_config      /path/to/eval_episodes.json \
#     [--task_name       make_microwave_popcorn] \
#     --policy_config    <policy_config_name> \
#     --policy_dir       /path/to/checkpoint/ \
#     --log_root         /path/to/output/f0 \
#     --num_runs         10 \
#     --base_port        8001
#
#   task_config JSON format:
#   {
#     "tasks": [
#       {
#         "task_name": "make_microwave_popcorn",
#         "task_prompt": "In the kitchen, ...",
#         "episodes": ["00400010", "00400020", "00400450"],
#         "states_root": "skill_init_states/task-0040",
#         "configs_root": "skill_eval_configs"
#       }
#     ]
#   }
#   - --task_name is optional and selects a task from the JSON (default: first)
#   - episodes must come from the JSON; if --episodes is also given, the JSON wins
#   - states_root / configs_root / task_prompt fall back to the CLI values
#     when missing in the JSON
#   - relative states_root / configs_root in the JSON are resolved against
#     the JSON file's own directory (e.g. data/eval_episodes.json pairs with
#     data/skill_init_states/ and data/skill_eval_configs/)
#
# B) with explicit arguments:
#   bash benchmark/scripts/run_auto_eval.sh \
#     --task_name        make_microwave_popcorn \
#     --task_prompt      "In the kitchen, ..." \
#     --policy_config    pi05_b1k-pt50_finegrain_0 \
#     --policy_dir       /path/to/checkpoint/49999 \
#     --episodes         00400010,00400020,00400450 \
#     --states_root      /path/to/subtask_states/task-0040 \
#     --log_root         /path/to/output/f0 \
#     --configs_root     /path/to/skill_eval_configs \
#     --num_runs         10 \
#     --base_port        8001
#
# Notes:
#   - GPU count is auto-detected; one policy server is started per GPU
#     (ports increase from --base_port)
#   - final config path = --configs_root / --task_name
#   - final log path    = --log_root / --task_name, results are stored as:
#       {log_root}/{task_name}/episode_{ep}/run_{XX}/
#   - each GPU hosts one evaluation worker; episodes are split round-robin
#   - evaluation starts only after ALL policy servers are ready
#   - --policy_prompt_only: pass the task prompt to the policy server only,
#     not to the evaluator
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Path resolution (based on the script's own location, independent of cwd) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"   # parent of benchmark/

# ── Policy service location ─────────────────────────────────────────────
# Repo containing the policy server launched in Step 1 (pi0.5 serve_b1k.py
# example). Replace this with the path to YOUR policy server repo.
POLICY_ROOT="path to your policy server repo"

# ── Defaults ─────────────────────────────────────────────────────────────────
TASK_CONFIG=""       # task config JSON; if given, task info is read from it
TASK_NAME=""
TASK_PROMPT=""
POLICY_CONFIG=""
POLICY_DIR=""
EPISODES=""
STATES_ROOT=""
LOG_ROOT=""
CONFIGS_ROOT=""
NUM_RUNS=10
START_RUN=1
POLICY_PROMPT_ONLY=false
EVAL_GPUS=0          # >0: the last N GPUs are dedicated to evaluation, the rest host policy servers
BASE_PORT=8001
SKILL_IDS=""          # comma-separated skill ids; empty = evaluate all (e.g. "2,5,6")
SKIP_EXISTING=false   # true = skip skills whose result file already exists

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_config)        TASK_CONFIG="$2";       shift 2 ;;
        --task_name)          TASK_NAME="$2";         shift 2 ;;
        --task_prompt)        TASK_PROMPT="$2";       shift 2 ;;
        --policy_config)      POLICY_CONFIG="$2";     shift 2 ;;
        --policy_dir)         POLICY_DIR="$2";        shift 2 ;;
        --episodes)           EPISODES="$2";          shift 2 ;;
        --states_root)        STATES_ROOT="$2";       shift 2 ;;
        --log_root)           LOG_ROOT="$2";          shift 2 ;;
        --configs_root)       CONFIGS_ROOT="$2";      shift 2 ;;
        --num_runs)           NUM_RUNS="$2";          shift 2 ;;
        --start_run)          START_RUN="$2";         shift 2 ;;
        --policy_prompt_only) POLICY_PROMPT_ONLY=true; shift 1 ;;
        --eval_gpus)          EVAL_GPUS="$2";         shift 2 ;;
        --base_port)          BASE_PORT="$2";         shift 2 ;;
        --skill_ids)          SKILL_IDS="$2";         shift 2 ;;
        --skip_existing)      SKIP_EXISTING=true;     shift 1 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

# ── task_config mode: read task info from the JSON ───────────────────────────
# When --task_config is given, task_name/task_prompt/episodes/states_root/
# configs_root are taken from the JSON first (--task_name only selects the
# entry, defaulting to the first one); fields missing in the JSON fall back
# to the CLI values of the same name.
if [[ -n "${TASK_CONFIG}" ]]; then
    if [[ ! -f "${TASK_CONFIG}" ]]; then
        echo "[ERROR] task_config file does not exist: ${TASK_CONFIG}"
        exit 1
    fi

    # Select the task: look up by --task_name, otherwise take the first one
    if [[ -n "${TASK_NAME}" ]]; then
        TASK_IDX=$(python3 -c "
import json, sys
d = json.load(open('${TASK_CONFIG}'))
for i, t in enumerate(d['tasks']):
    if t['task_name'] == '${TASK_NAME}':
        print(i); sys.exit(0)
print(-1)")
        if [[ "${TASK_IDX}" == "-1" ]]; then
            echo "[ERROR] task '${TASK_NAME}' not found in ${TASK_CONFIG}"
            exit 1
        fi
    else
        TASK_IDX=0
        echo "[INFO] No --task_name given, defaulting to the 1st task in task_config"
    fi

    # episodes must come from task_config
    if [[ -n "${EPISODES}" ]]; then
        echo "[WARN] Both --task_config and --episodes given; episodes from task_config take precedence"
    fi
    EPISODES=$(python3 -c "
import json
t = json.load(open('${TASK_CONFIG}'))['tasks'][${TASK_IDX}]
eps = t.get('episodes') or []
print(','.join(str(e) for e in eps))")
    if [[ -z "${EPISODES}" ]]; then
        echo "[ERROR] episodes of task [${TASK_IDX}] in task_config is empty or missing"
        exit 1
    fi

    # task_name always comes from the JSON (--task_name is only a selector here)
    TASK_NAME=$(python3 -c "import json; print(json.load(open('${TASK_CONFIG}'))['tasks'][${TASK_IDX}]['task_name'])")

    # remaining fields: JSON values override the CLI values when present
    _V=$(python3 -c "import json; print(json.load(open('${TASK_CONFIG}'))['tasks'][${TASK_IDX}].get('task_prompt', ''))")
    if [[ -n "${_V}" ]]; then TASK_PROMPT="${_V}"; fi
    _V=$(python3 -c "import json; print(json.load(open('${TASK_CONFIG}'))['tasks'][${TASK_IDX}].get('states_root', ''))")
    if [[ -n "${_V}" ]]; then STATES_ROOT="${_V}"; fi
    _V=$(python3 -c "import json; print(json.load(open('${TASK_CONFIG}'))['tasks'][${TASK_IDX}].get('configs_root', ''))")
    if [[ -n "${_V}" ]]; then CONFIGS_ROOT="${_V}"; fi

    # Resolve relative states_root / configs_root against the JSON's own
    # directory (the JSON and the data directories ship together under data/)
    _TC_DIR="$(cd "$(dirname "${TASK_CONFIG}")" && pwd)"
    if [[ "${STATES_ROOT}" != /* ]]; then STATES_ROOT="${_TC_DIR}/${STATES_ROOT}"; fi
    if [[ "${CONFIGS_ROOT}" != /* ]]; then CONFIGS_ROOT="${_TC_DIR}/${CONFIGS_ROOT}"; fi

    echo "[INFO] task_config mode: ${TASK_CONFIG} (task ${TASK_IDX}: ${TASK_NAME})"
fi

# ── Required-argument validation ─────────────────────────────────────────────
# task_config mode has already filled the variables above; this check applies
# uniformly to both modes
for var in TASK_NAME TASK_PROMPT POLICY_CONFIG POLICY_DIR EPISODES STATES_ROOT LOG_ROOT CONFIGS_ROOT; do
    if [[ -z "${!var}" ]]; then
        echo "[ERROR] Missing required argument: --${var,,}"
        exit 1
    fi
done
if [[ ! -d "${POLICY_ROOT}" ]]; then
    echo "[ERROR] POLICY_ROOT directory does not exist: ${POLICY_ROOT}"
    echo "        (edit POLICY_ROOT at the top of this script to point to your policy repo)"
    exit 1
fi

# ── GPU detection ───────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
else
    echo "[WARN] nvidia-smi not found, defaulting to 1 GPU"
    NUM_GPUS=1
fi
echo "[INFO] Detected GPUs: ${NUM_GPUS}"

# ── Policy / eval GPU allocation ─────────────────────────────────────────────
# EVAL_GPUS=0 (default): policy and eval share the same GPUs
# EVAL_GPUS=N (N>0):     the last N GPUs are dedicated to eval, the first
#                        (total - N) GPUs host the policy servers
if [[ ${EVAL_GPUS} -eq 0 ]]; then
    NUM_POLICY_GPUS=${NUM_GPUS}
    NUM_EVAL_GPUS=${NUM_GPUS}
    EVAL_GPU_OFFSET=0
else
    if [[ ${EVAL_GPUS} -ge ${NUM_GPUS} ]]; then
        echo "[ERROR] --eval_gpus (${EVAL_GPUS}) must be smaller than the total GPU count (${NUM_GPUS}); at least 1 GPU is required for the policy servers"
        exit 1
    fi
    NUM_POLICY_GPUS=$((NUM_GPUS - EVAL_GPUS))
    NUM_EVAL_GPUS=${EVAL_GPUS}
    EVAL_GPU_OFFSET=${NUM_POLICY_GPUS}
    echo "[INFO] GPU allocation: policy GPUs 0-$((NUM_POLICY_GPUS-1)), eval GPUs ${EVAL_GPU_OFFSET}-$((NUM_GPUS-1))"
fi

# ── Episode validation & split ──────────────────────────────────────────────
# Skip episodes whose state manifest or eval config does not exist.
IFS=',' read -ra ALL_EPISODES <<< "$EPISODES"
VALID_EPISODES=()
for EP in "${ALL_EPISODES[@]}"; do
    MANIFEST_EP="${STATES_ROOT}/episode_${EP}/state_manifest.json"
    CONFIG_EP="${CONFIGS_ROOT}/${TASK_NAME}/episode_${EP}.yaml"
    if [[ -f "${MANIFEST_EP}" && -f "${CONFIG_EP}" ]]; then
        VALID_EPISODES+=("${EP}")
    else
        echo "[WARN] Skipping episode ${EP}: missing ${MANIFEST_EP} or ${CONFIG_EP}"
    fi
done
TOTAL_EPISODES=${#VALID_EPISODES[@]}
if [[ ${TOTAL_EPISODES} -eq 0 ]]; then
    echo "[ERROR] No valid episodes (state manifest / eval config missing) under:"
    echo "        states_root:  ${STATES_ROOT}"
    echo "        configs_root: ${CONFIGS_ROOT}/${TASK_NAME}"
    exit 1
fi
echo "[INFO] Valid episodes: ${TOTAL_EPISODES}, split round-robin across ${NUM_POLICY_GPUS} evaluation workers"

# ── Log directories ──────────────────────────────────────────────────────────
LOG_DIR="${LOG_ROOT}/auto_eval_logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
POLICY_LOG_DIR="${LOG_DIR}/policy_servers"
EVAL_LOG_DIR="${LOG_DIR}/eval_workers"
mkdir -p "${POLICY_LOG_DIR}" "${EVAL_LOG_DIR}"
echo "[INFO] Log directory: ${LOG_DIR}"

# ── Cleanup on exit/interrupt: kill leftover background jobs ─────────────────
POLICY_PIDS=()
EVAL_PIDS=()
_cleanup() {
    for _pid in ${EVAL_PIDS[@]+"${EVAL_PIDS[@]}"}; do
        kill "${_pid}" 2>/dev/null || true
        pkill -P "${_pid}" 2>/dev/null || true
    done
    for _pid in ${POLICY_PIDS[@]+"${POLICY_PIDS[@]}"}; do
        kill "${_pid}" 2>/dev/null || true
        pkill -P "${_pid}" 2>/dev/null || true
    done
}
trap _cleanup EXIT

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: launch one policy server per policy GPU
# ─────────────────────────────────────────────────────────────────────────────
# The command below is the pi0.5 example — replace it with the launch command
# of your own policy server (see "Plugging in your own policy" at the top).
for ((i=0; i<NUM_POLICY_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    GPU_LOG="${POLICY_LOG_DIR}/policy_gpu${i}_port${PORT}.log"

    echo "[INFO] Launching policy server GPU=${i} PORT=${PORT} -> log: ${GPU_LOG}"

    (
        export CUDA_VISIBLE_DEVICES=${i}
        cd "${POLICY_ROOT}"
        # activate the Python environment of the policy server here if needed
        # (e.g. source .venv/bin/activate, conda activate ...)

        echo '=== policy server launch command ==='
        printf 'CUDA_VISIBLE_DEVICES=%s cd %s &&\n' "${i}" "${POLICY_ROOT}"
        printf 'python scripts/serve_b1k.py --task_name=%s --task_prompt=%s --port=%s --control_mode=receeding_horizon --max-len=32 --fine-grained-level=0 policy:checkpoint --policy.config=%s --policy.dir=%s\n' \
            "${TASK_NAME}" "${TASK_PROMPT}" "${PORT}" "${POLICY_CONFIG}" "${POLICY_DIR}"
        echo '======================================'

        python scripts/serve_b1k.py \
            --task_name="${TASK_NAME}" \
            --task_prompt="${TASK_PROMPT}" \
            --port "${PORT}" \
            --control_mode=receeding_horizon \
            --max-len=32 \
            --fine-grained-level=0 \
            policy:checkpoint \
            --policy.config="${POLICY_CONFIG}" \
            --policy.dir="${POLICY_DIR}"
    ) > "${GPU_LOG}" 2>&1 &

    POLICY_PIDS+=($!)
    echo "[INFO] policy server PID=${POLICY_PIDS[-1]} started in background"
done

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: wait in parallel until all policy servers are ready
# (watch the log files so evaluation does not hit websocket connection errors)
# ─────────────────────────────────────────────────────────────────────────────
echo "[INFO] Waiting for ${NUM_POLICY_GPUS} policy servers to become ready (parallel checks)..."

WAIT_TIMEOUT=1800  # wait at most 30 minutes (large model loading is slow)

# One background watcher per policy server, polling the log for the ready flag
WAIT_PIDS=()
for ((i=0; i<NUM_POLICY_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    GPU_LOG="${POLICY_LOG_DIR}/policy_gpu${i}_port${PORT}.log"
    (
        ELAPSED=0
        # policy servers print "server listening on 0.0.0.0:<PORT>" once ready
        READY_PATTERN="server listening on"
        while true; do
            if [[ -f "${GPU_LOG}" ]] && grep -q "${READY_PATTERN}" "${GPU_LOG}" 2>/dev/null; then
                echo "[INFO] policy server PORT=${PORT} is ready (${ELAPSED}s)"
                exit 0
            fi
            if [[ ${ELAPSED} -ge ${WAIT_TIMEOUT} ]]; then
                echo "[WARN] Timed out waiting for policy PORT=${PORT} (${WAIT_TIMEOUT}s), starting evaluation anyway..."
                exit 0
            fi
            sleep 5
            ELAPSED=$((ELAPSED + 5))
        done
    ) &
    WAIT_PIDS+=($!)
done

# Wait for all port watchers to finish
for pid in "${WAIT_PIDS[@]}"; do
    wait "${pid}"
done
echo "[INFO] All policy server checks finished, starting evaluation"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: launch one evaluation worker per GPU (episodes split round-robin)
# ─────────────────────────────────────────────────────────────────────────────
# Each worker runs the full episodes x runs loop for its share of the episodes,
# delegating every (episode, run) combination to run_all_skills.sh, which
# iterates over the skills of that episode.
for ((k=0; k<NUM_POLICY_GPUS; k++)); do
    EVAL_GPU_IDX=$((EVAL_GPU_OFFSET + k % NUM_EVAL_GPUS))
    PORT=$((BASE_PORT + k))
    EVAL_LOG="${EVAL_LOG_DIR}/eval_gpu${EVAL_GPU_IDX}_port${PORT}.log"

    # round-robin split: worker k takes episodes k, k+NUM_POLICY_GPUS, ...
    GPU_EPISODES=()
    for ((j=k; j<TOTAL_EPISODES; j+=NUM_POLICY_GPUS)); do
        GPU_EPISODES+=("${VALID_EPISODES[j]}")
    done

    if [[ ${#GPU_EPISODES[@]} -eq 0 ]]; then
        echo "[INFO] eval worker ${k}: no episodes assigned, skipping"
        continue
    fi

    EPISODE_STR=$(IFS=','; echo "${GPU_EPISODES[*]}")
    echo "[INFO] Launching eval worker ${k}: eval_GPU=${EVAL_GPU_IDX} PORT=${PORT} episodes=[${EPISODE_STR}] -> log: ${EVAL_LOG}"

    (
        # eval workers inherit the current environment; only the GPU is
        # pinned per worker
        export CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX}
        cd "${REPO_ROOT}"

        for EP in "${GPU_EPISODES[@]}"; do
            MANIFEST="${STATES_ROOT}/episode_${EP}/state_manifest.json"
            EVAL_CONFIG="${CONFIGS_ROOT}/${TASK_NAME}/episode_${EP}.yaml"

            for ((RUN=${START_RUN}; RUN<${START_RUN}+${NUM_RUNS}; RUN++)); do
                RUN_TAG=$(printf '%02d' "${RUN}")
                RUN_LOG_DIR="${LOG_ROOT}/${TASK_NAME}/episode_${EP}/run_${RUN_TAG}"

                # Optional arguments for run_all_skills.sh
                EXTRA_ARGS=()
                if [[ -n "${SKILL_IDS}" ]]; then
                    EXTRA_ARGS+=(--skill_ids "${SKILL_IDS}")
                fi
                if [[ "${SKIP_EXISTING}" = "true" ]]; then
                    EXTRA_ARGS+=(--skip_existing)
                fi
                # --policy_prompt_only: the task prompt goes to the policy
                # server only, not to the evaluator
                if [[ "${POLICY_PROMPT_ONLY}" != "true" ]]; then
                    EXTRA_ARGS+=(--prompt "${TASK_PROMPT}")
                fi

                echo ""
                echo "----- [worker ${k}] episode=${EP} run=${RUN_TAG} -----"
                echo "  log_dir:     ${RUN_LOG_DIR}"
                echo "  manifest:    ${MANIFEST}"
                echo "  eval_config: ${EVAL_CONFIG}"

                # A failed (episode, run) does not abort the worker; the
                # remaining combinations are still evaluated
                if ! bash "${SCRIPT_DIR}/run_all_skills.sh" 0 -1 \
                        --task "${TASK_NAME}" \
                        --port "${PORT}" \
                        --log_dir "${RUN_LOG_DIR}" \
                        --manifest "${MANIFEST}" \
                        --eval_config "${EVAL_CONFIG}" \
                        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
                    echo "[WARN] [worker ${k}] episode=${EP} run=${RUN_TAG} exited non-zero, continuing"
                fi
            done
        done
    ) > "${EVAL_LOG}" 2>&1 &

    EVAL_PIDS+=($!)
    echo "[INFO] eval worker PID=${EVAL_PIDS[-1]} started in background"
done

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: wait for all evaluation workers to finish
# ─────────────────────────────────────────────────────────────────────────────
echo "[INFO] Waiting for all evaluation workers to finish..."

ALL_OK=true
for pid in "${EVAL_PIDS[@]}"; do
    if wait "${pid}"; then
        echo "[INFO] eval worker PID=${pid} finished"
    else
        echo "[WARN] eval worker PID=${pid} exited non-zero"
        ALL_OK=false
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: stop all policy servers
# ─────────────────────────────────────────────────────────────────────────────
echo "[INFO] Stopping all policy servers..."
for pid in ${POLICY_PIDS[@]+"${POLICY_PIDS[@]}"}; do
    if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
        pkill -P "${pid}" 2>/dev/null || true
        echo "[INFO] policy server PID=${pid} stopped"
    fi
done

if ${ALL_OK}; then
    echo "[INFO] All evaluations finished. Logs: ${LOG_DIR}"
    echo "[INFO] Per-run results are stored under: ${LOG_ROOT}/${TASK_NAME}/episode_<EP>/run_<XX>/"
    echo "[INFO] To aggregate metrics across runs and episodes, run e.g.:"
    echo "    python -m benchmark.metrics.task_summary --log_root '${LOG_ROOT}/${TASK_NAME}' --task '${TASK_NAME}' --configs_root '${CONFIGS_ROOT}/${TASK_NAME}'"
else
    echo "[WARN] Some evaluation workers exited non-zero; check the logs: ${LOG_DIR}"
    exit 1
fi
