#!/bin/bash
# ============================================================
# Auto-Aim 节点启动脚本
# 用法: ./scripts/start_node.sh [包名] [节点名]
# 示例: ./scripts/start_node.sh test_pipeline deep_node
#       ./scripts/start_node.sh pipeline sentry_node
# ============================================================

set -euo pipefail

# -------------------- 配置区 --------------------
WORKSPACE_DIR="/home/guo/ITL_Auto_aim"
SERIAL_DEV="/dev/ttyACM0"          # 云台串口设备
CAMERA_USB_VID="2bdf"              # 海康/MindVision 相机 USB Vendor ID
MAX_WAIT=30                        # 设备等待超时(秒)
CHECK_INTERVAL=2                   # 检测间隔(秒)

# 默认启动的包和节点
PACKAGE="${1:-pipeline}"
NODE="${2:-standard3_node}"
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
    log "========== Auto-Aim 启动脚本 =========="
    log "目标: ros2 run ${PACKAGE} ${NODE}"

    # 1. 设备检测（并行等待两个设备）
    wait_for_camera &
    local cam_pid=$!
    wait_for_serial &
    local serial_pid=$!

    local failed=0
    wait $cam_pid || failed=1
    wait $serial_pid || failed=1

    if [ $failed -ne 0 ]; then
        log "设备检测未通过，终止启动"
        exit 1
    fi

    # 2. 加载 ROS2 环境
    check_workspace || exit 1

    # 3. 启动节点
    # 切换到工作空间目录，确保 spdlog 日志写入正确的 logs/ 目录
    cd "${WORKSPACE_DIR}"

    log "正在启动节点: ros2 run ${PACKAGE} ${NODE}"
    exec ros2 run "${PACKAGE}" "${NODE}"
}

main
