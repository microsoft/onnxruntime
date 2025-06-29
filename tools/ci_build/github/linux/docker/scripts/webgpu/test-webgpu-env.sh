#!/bin/bash
# Test WebGPU environment by creating a minimal Dawn/WebGPU instance
echo "=== WebGPU Environment Test ==="

# Set up environment
export DISPLAY=${DISPLAY:-:99}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-dir}
export VK_FORCE_HEADLESS=1

# Configure Vulkan ICD preference
nvidia_icd="/usr/share/vulkan/icd.d/nvidia_icd.json"
mesa_icd="/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"

if [ -f "$nvidia_icd" ] && command -v nvidia-smi >/dev/null 2>&1; then
    export VK_ICD_FILENAMES="$nvidia_icd"
    echo "Using NVIDIA Vulkan driver for WebGPU test"
elif [ -f "$mesa_icd" ]; then
    export VK_ICD_FILENAMES="$mesa_icd"
    echo "Using Mesa Vulkan driver for WebGPU test"
else
    echo "No suitable Vulkan drivers found"
fi

echo "Environment variables:"
echo "  DISPLAY: $DISPLAY"
echo "  XDG_RUNTIME_DIR: $XDG_RUNTIME_DIR"
echo "  VK_ICD_FILENAMES: $VK_ICD_FILENAMES"
echo "  VK_FORCE_HEADLESS: $VK_FORCE_HEADLESS"

# Test if we can create the runtime directory
if [ ! -d "$XDG_RUNTIME_DIR" ]; then
    echo "Creating runtime directory: $XDG_RUNTIME_DIR"
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
fi

# Test Vulkan loader functionality if available
if command -v vulkaninfo >/dev/null 2>&1; then
    echo "=== Vulkan Loader Test ==="
    timeout 15 vulkaninfo --summary 2>&1 | head -30
    echo "=== End Vulkan Test ==="
else
    echo "vulkaninfo not available for testing"
fi

echo "=== Environment test complete ==="
