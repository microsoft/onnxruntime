#!/bin/bash
echo "=== Vulkan Environment Debug ==="
echo "DISPLAY: $DISPLAY"
echo "XDG_RUNTIME_DIR: $XDG_RUNTIME_DIR"
echo "VK_ICD_FILENAMES: $VK_ICD_FILENAMES"
echo "Available ICDs:"
ls -la /usr/share/vulkan/icd.d/ 2>/dev/null || echo "No ICD directory found"
echo "=== End Debug ==="
