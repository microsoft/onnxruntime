#!/bin/bash
set -ex

agentName=$1
target_device=$2
echo "agent name $agentName"
echo "agent target device : $target_device"

echo -e "\n ---- rocm-smi"
rocm-smi

echo -e "\n ---- rocm-smi --showpids"
rocm-smi --showpids

echo -e "\n ---- rocm-smi --showpidgpus"
rocm-smi --showpidgpus

echo -e "\n ---- rocm-smi --showpids detail"
rocm-smi --showpids | awk '$1 ~/[0-9]+/{if((NR>6)) {print $1}}' | xargs -I {} ps {}

echo -e "\n ---- rocm-smi --showmeminfo"
rocm-smi --showmeminfo vram vis_vram gtt

echo -e "\n ---- Clean up the process that is using the target device"
gpu_details=$(rocm-smi --showpidgpus)
pid_lines=$(echo "$gpu_details" | grep -n "DRM device" | cut -d ":" -f 1)
pid_lines_array=($pid_lines)

for ((i = 0; i < ${#pid_lines_array[@]}; i++)); do
    pid_line=${pid_lines_array[$i]}
    pid=$(echo "$gpu_details" | awk '{print $2}' | sed -n "${pid_line}p")
    gpu_line=$((pid_line + 1))
    pid_gpu=$(echo "$gpu_details" | sed -n "${gpu_line}p" | sed -e 's/^[ ]*//g' | sed -e 's/[ ]*$//g')
    if [ "$pid_gpu" == "$target_device" ]; then
        echo "kill pid: $pid, gpu: $pid_gpu"
        kill -9 $pid
    fi
done
