#!/bin/bash
# Configure Vulkan environment for WebGPU based on available drivers
export DISPLAY=${DISPLAY:-:99}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-dir}
export VK_FORCE_HEADLESS=${VK_FORCE_HEADLESS:-1}

# Set ICD preference: NVIDIA first, then Mesa fallback
nvidia_icd="/usr/share/vulkan/icd.d/nvidia_icd.json"
mesa_icd="/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"

if [ -f "$nvidia_icd" ] && command -v nvidia-smi >/dev/null 2>&1; then
    export VK_ICD_FILENAMES="$nvidia_icd"
    echo "Configured for NVIDIA Vulkan driver"
elif [ -f "$mesa_icd" ]; then
    export VK_ICD_FILENAMES="$mesa_icd"
    echo "Configured for Mesa LLVMpipe fallback"
else
    echo "No Vulkan drivers found - WebGPU will use CPU mode"
fi

# Execute the original command
exec "$@"
