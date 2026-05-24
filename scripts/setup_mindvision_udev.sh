#!/bin/bash
# ==================================================
# MindVision USB camera permission setup script
# Usage: sudo bash setup_mindvision_udev.sh
# Function: install udev rule for ordinary-user access to MindVision SUA133GC
# ==================================================

set -e

RULES_FILE="/etc/udev/rules.d/99-mindvision.rules"
VID="f622"
PID="d132"
PRODUCT_NAME="MindVision SUA133GC"

# Must run as root because /etc/udev/rules.d and udevadm reload require privileges.
if [ "$EUID" -ne 0 ]; then
    echo "[error] Please run with sudo: sudo bash $0"
    exit 1
fi

echo "Installing udev rule for $PRODUCT_NAME (VID:PID = $VID:$PID)"
echo "Rule file: $RULES_FILE"

RULE="SUBSYSTEM==\"usb\", ATTR{idVendor}==\"$VID\", ATTR{idProduct}==\"$PID\", MODE=\"0666\""

echo "# $PRODUCT_NAME ordinary-user USB access (VID:$VID PID:$PID)" > "$RULES_FILE"
echo "$RULE" >> "$RULES_FILE"

udevadm control --reload-rules
udevadm trigger

echo ""
echo "======================================"
echo "  MindVision udev rule installed."
echo "======================================"
echo ""
echo "If the camera is already plugged in, unplug and replug it if permissions do not update immediately."
echo "Verify with:"
echo "  lsusb | grep -i MindVision"
echo "  ls -l /dev/bus/usb/*/*"
echo "Expected permission for the MindVision device: crw-rw-rw-"
