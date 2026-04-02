#!/bin/bash
# ==================================================
# 串口设备映射自动安装脚本
# 用法: sudo bash setup_gimbal_udev.sh
# 功能: 自动检测当前连接的下位机，生成并安装 udev 规则
# ==================================================

set -e

SYMLINK_NAME="gimbal"
RULES_FILE="/etc/udev/rules.d/99-gimbal.rules"

# 必须 root 执行
if [ "$EUID" -ne 0 ]; then
    echo "[错误] 请使用 sudo 运行: sudo bash $0"
    exit 1
fi

# 扫描所有 ttyACM / ttyUSB 设备
echo "正在扫描串口设备..."
devices=()
for dev in /dev/ttyACM* /dev/ttyUSB*; do
    [ -e "$dev" ] && devices+=("$dev")
done

if [ ${#devices[@]} -eq 0 ]; then
    echo "[错误] 未检测到任何串口设备，请先插上下位机"
    exit 1
fi

# 列出所有设备信息供选择
echo ""
echo "检测到以下串口设备:"
echo "-------------------------------------------"
for i in "${!devices[@]}"; do
    dev="${devices[$i]}"
    vid=$(udevadm info --name="$dev" --attribute-walk 2>/dev/null | grep 'ATTRS{idVendor}' | head -1 | sed 's/.*=="\(.*\)"/\1/')
    pid=$(udevadm info --name="$dev" --attribute-walk 2>/dev/null | grep 'ATTRS{idProduct}' | head -1 | sed 's/.*=="\(.*\)"/\1/')
    product=$(udevadm info --name="$dev" --attribute-walk 2>/dev/null | grep 'ATTRS{product}' | head -1 | sed 's/.*=="\(.*\)"/\1/')
    echo "  [$i] $dev  (VID:PID = $vid:$pid  $product)"
done
echo "-------------------------------------------"

# 只有一个设备直接选中，多个设备让用户选
if [ ${#devices[@]} -eq 1 ]; then
    choice=0
    echo "只有一个设备，自动选择: ${devices[0]}"
else
    echo ""
    read -p "请输入下位机对应的编号 [0-$((${#devices[@]}-1))]: " choice
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -ge ${#devices[@]} ]; then
        echo "[错误] 无效选择"
        exit 1
    fi
fi

selected="${devices[$choice]}"
VID=$(udevadm info --name="$selected" --attribute-walk 2>/dev/null | grep 'ATTRS{idVendor}' | head -1 | sed 's/.*=="\(.*\)"/\1/')
PID=$(udevadm info --name="$selected" --attribute-walk 2>/dev/null | grep 'ATTRS{idProduct}' | head -1 | sed 's/.*=="\(.*\)"/\1/')

if [ -z "$VID" ] || [ -z "$PID" ]; then
    echo "[错误] 无法读取设备 $selected 的 VID/PID"
    exit 1
fi

echo ""
echo "选中设备: $selected"
echo "VID:PID = $VID:$PID"
echo "将创建映射: /dev/$SYMLINK_NAME -> $selected"

# 生成并写入规则
RULE="SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"$VID\", ATTRS{idProduct}==\"$PID\", SYMLINK+=\"$SYMLINK_NAME\", MODE=\"0666\""

echo ""
echo "写入规则到 $RULES_FILE ..."
echo "# 下位机串口映射 (VID:$VID PID:$PID -> /dev/$SYMLINK_NAME)" > "$RULES_FILE"
echo "$RULE" >> "$RULES_FILE"

# 重载并触发
udevadm control --reload-rules
udevadm trigger

# 等待 symlink 生效
sleep 1

# 验证
if [ -L "/dev/$SYMLINK_NAME" ]; then
    target=$(readlink -f "/dev/$SYMLINK_NAME")
    echo ""
    echo "======================================"
    echo "  安装成功！"
    echo "  /dev/$SYMLINK_NAME -> $target"
    echo "======================================"
    echo ""
    echo "请确认 yaml 配置文件中 com_port 设为: \"/dev/$SYMLINK_NAME\""
else
    echo ""
    echo "[警告] /dev/$SYMLINK_NAME 未出现，请尝试重新插拔USB后检查"
fi
