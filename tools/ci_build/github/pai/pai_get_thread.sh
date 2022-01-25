#!/bin/bash 

agentName=$1
finalCharacter=${agentName: -1}
echo "agent name $agentName"
echo "agent name final character : $finalCharacter"

echo "============ rocm-smi =============="
rocm-smi
echo "============ rocm-smi --showpids =============="
rocm-smi --showpids