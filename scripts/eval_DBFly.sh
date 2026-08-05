
# 当前脚本所在目录
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# scripts 的上一级目录，即项目根目录
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

SIM_ROOT="$ROOT_DIR/env_unzip"
SIM_PORT=30000
MASTER_PORT=60001
GPU_ID=0

MAX_RETRY=-1
SLEEP_SECONDS=10

retry_count=0


kill_port() {
    port=$1
    echo "[INFO] Killing processes on port ${port}..."

    pids=$(lsof -ti:${port})

    if [ -n "$pids" ]; then
        echo "[INFO] Found PIDs on port ${port}: ${pids}"
        kill -9 ${pids}
        sleep 1
    else
        echo "[INFO] No process found on port ${port}"
    fi
}


prepare_simulator() {
    echo "=========================================="
    echo "[INFO] Prepare simulator at $(date)"
    echo "=========================================="

    kill_port ${SIM_PORT}
    kill_port $((SIM_PORT + 1))
    kill_port $((SIM_PORT + 2))

    echo "[INFO] Starting AirVLN simulator server on port ${SIM_PORT}..."


    (
        cd "$ROOT_DIR/airsim_plugin" || exit 1

        nohup python AirVLNSimulatorServerTool.py \
            --port ${SIM_PORT} \
            --root_path ${SIM_ROOT} \
            > simulator_${SIM_PORT}.log 2>&1 &

        sim_pid=$!
        echo "[INFO] Simulator PID: ${sim_pid}"
    )


    sleep 10
}


while true
do
    prepare_simulator

    echo "=========================================="
    echo "[INFO] Start eval at $(date)"
    echo "[INFO] Retry count: ${retry_count}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u $ROOT_DIR/src/vlnce_src/eval_IFC-VLN.py \
        --run_type eval \
        --name  qwen3_vl \
        --gpu_id 0 \
        --simulator_tool_port ${SIM_PORT} \
        --DDP_MASTER_PORT ${MASTER_PORT} \
        --batchSize 1 \
        --maxWaypoints 100 \
        --dataset_path $ROOT_DIR/dataset/test \
        --eval_save_path  $ROOT_DIR/result/test \
        --log_path  $ROOT_DIR/log_files \
        --model_path   $ROOT_DIR/models/DBFly \
        --map_spawn_area_json_path $ROOT_DIR/meta/map_spawnarea_info.json \
        --obj_desc_json_path $ROOT_DIR/meta/instruction.json

    exit_code=$?

    echo "[INFO] Eval exited with code: ${exit_code}"
    echo "[INFO] End time: $(date)"

    if [ ${exit_code} -eq 0 ]; then
        echo "[INFO] Eval finished normally. Stop restarting."
        break
    fi

    if [ ${exit_code} -eq 130 ]; then
        echo "[INFO] Interrupted by Ctrl+C. Stop restarting."
        break
    fi

    retry_count=$((retry_count + 1))

    if [ ${MAX_RETRY} -ne -1 ] && [ ${retry_count} -ge ${MAX_RETRY} ]; then
        echo "[ERROR] Reached max retry count: ${MAX_RETRY}. Stop restarting."
        break
    fi

    echo "[WARNING] Eval crashed abnormally. Restart after ${SLEEP_SECONDS} seconds..."
    sleep ${SLEEP_SECONDS}
done
