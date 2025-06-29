#!/bin/bash
# Simple validation that doesnt run Vulkan - just checks driver availability
echo "=== Vulkan Environment Check ==="

# Check GPU availability first
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✓ NVIDIA GPU utilities available"
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "  (GPU query failed)"
else
    echo "ℹ NVIDIA GPU utilities not available"
fi

# Check for Vulkan ICDs
echo "Checking Vulkan drivers..."

nvidia_icd="/usr/share/vulkan/icd.d/nvidia_icd.json"
if [ -f "$nvidia_icd" ]; then
    echo "✓ NVIDIA Vulkan ICD found: $nvidia_icd"
    # Try to read the ICD file to see if it points to valid library
    if grep -q "libGLX_nvidia" "$nvidia_icd" 2>/dev/null; then
        echo "  - NVIDIA ICD appears configured for hardware acceleration"
    fi
else
    echo "✗ NVIDIA Vulkan ICD not found"
fi

# Check for Mesa/software ICDs
mesa_found=false
for mesa_icd in "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json" "/usr/share/vulkan/icd.d/mesa_icd.x86_64.json"; do
    if [ -f "$mesa_icd" ]; then
        echo "✓ Mesa Vulkan ICD found: $mesa_icd"
        mesa_found=true
        break
    fi
done

if [ "$mesa_found" = "false" ]; then
    echo "✗ Mesa/LLVMpipe Vulkan ICD not found"
fi

# List all available ICDs
echo "All Vulkan ICDs in /usr/share/vulkan/icd.d/:"
ls -la /usr/share/vulkan/icd.d/ 2>/dev/null || echo "  Directory not found"

echo "=== Environment Summary ==="
if [ -f "$nvidia_icd" ]; then
    echo "WebGPU will attempt hardware acceleration via NVIDIA Vulkan"
elif [ "$mesa_found" = "true" ]; then
    echo "WebGPU will fall back to software rendering via Mesa Vulkan"
else
    echo "WebGPU may fall back to CPU-only mode (no Vulkan drivers found)"
fi

# Test if vulkaninfo can enumerate devices (critical for WebGPU)
echo "=== Vulkan Device Enumeration Test ==="
if command -v vulkaninfo >/dev/null 2>&1; then
    echo "Testing Vulkan device enumeration..."
    # Set environment for testing
    export DISPLAY=${DISPLAY:-:99}
    export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-dir}
    export VK_FORCE_HEADLESS=1

    if [ -f "$nvidia_icd" ] && command -v nvidia-smi >/dev/null 2>&1; then
        export VK_ICD_FILENAMES="$nvidia_icd"
        echo "Testing with NVIDIA ICD..."
    elif [ "$mesa_found" = "true" ]; then
        export VK_ICD_FILENAMES="$mesa_icd"
        echo "Testing with Mesa ICD..."
    fi

    # Try to get device information (timeout after 10 seconds)
    timeout 10 vulkaninfo --summary 2>/dev/null | head -20 && echo "✓ Vulkan device enumeration successful" || echo "✗ Vulkan device enumeration failed"
else
    echo "vulkaninfo not available - cannot test device enumeration"
fi

echo "=== End Check ==="
