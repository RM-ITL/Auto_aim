#!/bin/bash
# ============================================================
# ITL_AutoAim 节点启动脚本
#
# 默认面向哨兵直连自瞄入口：
#   ./scripts/start_node.sh
#   ./scripts/start_node.sh pipeline sentry_node
#
# 普通车入口示例：
#   ./scripts/start_node.sh pipeline standard3_node
#
# 哨兵桥接模式示例：
#   ./scripts/start_node.sh pipeline sentry_node src/config/sentry.yaml --ros-args -p run_mode:=bridge
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -------------------- 配置区 --------------------
WORKSPACE_DIR="${WORKSPACE_DIR:-${WORKSPACE_DIR_DEFAULT}}"
SERIAL_DEV="${SERIAL_DEV:-/dev/gimbal}"   # 仅普通车/legacy 直连模式使用
CAMERA_USB_VID="${CAMERA_USB_VID:-2bdf}"
MAX_WAIT="${MAX_WAIT:-30}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
WAIT_FOR_SERIAL="${WAIT_FOR_SERIAL:-auto}"  # auto|always|never

PACKAGE="${1:-pipeline}"
NODE="${2:-sentry_node}"
DEFAULT_CONFIG="${WORKSPACE_DIR}/src/config/sentry.yaml"

EXTRA_ARGS=()
if [[ $# -ge 3 ]]; then
    EXTRA_ARGS=("${@:3}")
elif [[ "${NODE}" == "sentry_node" ]]; then
    EXTRA_ARGS=("${DEFAULT_CONFIG}")
fi
# ------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ==================== 设备检测 ====================

wait_for_camera() {
    local elapsed=0
    log "正在检测 USB 相机 (VID: ${CAMERA_USB_VID})..."
    while [ $elapsed -lt $MAX_WAIT ]; do
        if lsusb | grep -qi "${CAMERA_USB_VID}"; then
            log "相机已检测到"
            return 0
        fi
        sleep $CHECK_INTERVAL
        elapsed=$((elapsed + CHECK_INTERVAL))
        log "等待相机中... (${elapsed}/${MAX_WAIT}s)"
    done
    log "错误: 超时未检测到相机"
    return 1
}

wait_for_serial() {
    local elapsed=0
    log "正在检测串口设备 (${SERIAL_DEV})..."
    while [ $elapsed -lt $MAX_WAIT ]; do
        if [ -e "$SERIAL_DEV" ]; then
            log "串口设备 ${SERIAL_DEV} 已就绪"
            return 0
        fi
        sleep $CHECK_INTERVAL
        elapsed=$((elapsed + CHECK_INTERVAL))
        log "等待串口中... (${elapsed}/${MAX_WAIT}s)"
    done
    log "错误: 超时未检测到串口 ${SERIAL_DEV}"
    return 1
}

should_wait_for_serial() {
    local extra_args_joined="${EXTRA_ARGS[*]:-}"
    case "${WAIT_FOR_SERIAL}" in
        always) return 0 ;;
        never) return 1 ;;
        auto)
            if [[ "${NODE}" == "sentry_node" && "${extra_args_joined}" == *"run_mode:=bridge"* ]]; then
                return 1
            fi
            if [[ "${NODE}" == "sentry_node" || "${NODE}" == "standard3_node" || "${NODE}" == "hero_node" ]]; then
                return 0
            fi
            return 1
            ;;
        *)
            log "错误: WAIT_FOR_SERIAL 只能是 auto / always / never"
            return 2
            ;;
    esac
}

# ==================== 环境检查 ====================

check_workspace() {
    local setup_file="${WORKSPACE_DIR}/install/setup.bash"
    if [ ! -f "$setup_file" ]; then
        log "错误: 未找到 ${setup_file}，请先编译工作空间"
        return 1
    fi
    log "加载 ROS2 工作空间环境..."
    set +u
    source "$setup_file"
    set -u
    log "ROS2 环境已加载"
}

# ==================== 主流程 ====================

main() {
    log "========== ITL_AutoAim 启动脚本 =========="
    log "工作区: ${WORKSPACE_DIR}"
    log "目标: ros2 run ${PACKAGE} ${NODE} ${EXTRA_ARGS[*]:-}"

    # 1. 设备检测
    wait_for_camera &
    local cam_pid=$!

    local failed=0
    wait $cam_pid || failed=1

    if should_wait_for_serial; then
        wait_for_serial || failed=1
    else
        log "当前入口不要求本仓持有串口，跳过串口检测"
    fi

    if [ $failed -ne 0 ]; then
        log "设备检测未通过，终止启动"
        exit 1
    fi

    # 2. 加载 ROS2 环境
    check_workspace || exit 1

    # 3. 启动节点
    # 切换到工作空间目录，确保 spdlog 日志写入正确的 logs/ 目录
    cd "${WORKSPACE_DIR}"

    log "正在启动节点: ros2 run ${PACKAGE} ${NODE} ${EXTRA_ARGS[*]:-}"
    exec ros2 run "${PACKAGE}" "${NODE}" "${EXTRA_ARGS[@]}"
}

main
