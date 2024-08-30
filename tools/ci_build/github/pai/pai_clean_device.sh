#!/bin/bash
set -ex

usage() { echo "Usage: $0 [-n <agent name>] [-d <target device>] [-r <driver render>]" 1>&2; exit 1; }

while getopts "n:d:r:" parameter_Option
do case "${parameter_Option}"
in
n) AGENT_NAME=${OPTARG};;
d) TARGET_DEVICE=${OPTARG};;
r) DRIVER_RENDER=${OPTARG};;
*) usage ;;
esac
done

echo "Agent Name: $AGENT_NAME, Target Device: $TARGET_DEVICE, Driver Render: $DRIVER_RENDER"

echo -e "\n ---- Execute rocm-smi"
rocm-smi

echo -e "\n ---- Execute rocm-smi --showpids"
rocm-smi --showpids

echo -e "\n ---- Execute rocm-smi --showpidgpus"
rocm-smi --showpidgpus

echo -e "\n ---- Execute rocm-smi --showpids detail"
rocm-smi --showpids | awk '$1 ~/[0-9]+/{if((NR>6)) {print $1}}' | xargs -I {} ps {}

echo -e "\n ---- Execute rocm-smi --showmeminfo"
rocm-smi --showmeminfo vram vis_vram gtt

echo -e "\n ---- Clean up processes that use the target device $TARGET_DEVICE"
GPU_USED_BY_PIDS=$(rocm-smi --showpidgpus)
PID_NUMBERS_LINES=$(echo "$GPU_USED_BY_PIDS" | grep -n "DRM device" | cut -d ":" -f 1)
PID_NUMBERS_LINES_ARRAY=($PID_NUMBERS_LINES)

for ((i = 0; i < ${#PID_NUMBERS_LINES_ARRAY[@]}; i++)); do
    PID_NUMBER_LINE=${PID_NUMBERS_LINES_ARRAY[$i]}
    PID_NUMBER=$(echo "$GPU_USED_BY_PIDS" | awk '{print $2}' | sed -n "${PID_NUMBER_LINE}p")
    GPU_USED_BY_PID_LINE=$((PID_NUMBER_LINE + 1))
    GPU_USED_BY_PID=$(echo "$GPU_USED_BY_PIDS" | sed -n "${GPU_USED_BY_PID_LINE}p" | sed -e 's/^[ ]*//g' | sed -e 's/[ ]*$//g')
    if [ "$GPU_USED_BY_PID" == "$TARGET_DEVICE" ]; then
        echo "kill pid: $PID_NUMBER, using gpu: $GPU_USED_BY_PID"
        kill -9 "$PID_NUMBER"
    fi
done
