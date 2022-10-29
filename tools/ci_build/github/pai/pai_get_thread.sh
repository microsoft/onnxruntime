#!/bin/bash 

agentName=$1
finalCharacter=${agentName: -1}
echo "agent name $agentName"
echo "agent name final character : $finalCharacter"
targetRender=$((finalCharacter+128))

echo -e "\n ---- rocm-smi"
rocm-smi

echo -e "\n ---- rocm-smi --showpids"
rocm-smi --showpids

echo -e "\n ---- rocm-smi --showpidgpus"
rocm-smi --showpidgpus 

echo -e "\n ---- rocm-smi --showpids detail"
rocm-smi --showpids | awk '$1 ~/[0-9]+/{if((NR>6)) {print $1}}' | xargs -I {} ps {}

echo -e "\n ---- rocm-smi --showmeminfo"
rocm-smi  --showmeminfo vram vis_vram gtt

echo -e "\n ---- show all renders"
lsof /dev/dri/renderD*

echo -e "\n ---- show specific render"
lsof /dev/dri/renderD${targetRender}

echo -e "\n ---- show specific render pids detail"
lsof /dev/dri/renderD${targetRender} | grep "mem" | awk '{print $2}' | xargs -I {} ps {}