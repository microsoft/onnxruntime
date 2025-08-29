# ONNX Runtime Installation Guide

## Installation Options

ONNX Runtime provides several installation options depending on your platform and use case.

## Windows Installation

### WinML Installs (Recommended for Windows)

**WinML (Windows Machine Learning)** is the **preferred installation method** for Windows developers. WinML provides an optimized, future-safe solution that simplifies Windows deployment.

#### Key Benefits

- **Integrated Windows Experience**: Designed specifically for Windows applications
- **Automatic Hardware Optimization**: Dynamically selects the best execution provider based on available hardware
- **Same ONNX Runtime APIs**: Full compatibility with existing ONNX Runtime code
- **Simplified Deployment**: Handles packaging and distribution complexities

#### Installation Methods

**Via NuGet (C#/.NET)**:
```bash
Install-Package Microsoft.AI.MachineLearning
```

**Via WinGet**:
```bash
winget install Microsoft.WindowsML
```

**Via Microsoft Store**: Search for "Windows Machine Learning" in the Microsoft Store.

#### Getting Started with WinML

1. Install using one of the methods above
2. Follow the [Windows Getting Started Guide](../get-started/with-windows.md)
3. Review the [official WinML documentation](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)

#### WinML vs. DirectML

> **Note**: DirectML execution provider is deprecated. WinML is the recommended replacement that provides better integration and future support.

### C/C++ WinML Installs

For C/C++ development with WinML:

1. Download the latest WinML headers and libraries from the [releases page](https://github.com/microsoft/onnxruntime/releases)
2. Include the WinML headers in your project
3. Link against the appropriate WinML libraries

### Python with WinML

WinML can be used from Python through the standard ONNX Runtime Python package, which automatically uses WinML optimizations on Windows:

```bash
pip install onnxruntime
```

For GPU acceleration:
```bash
pip install onnxruntime-directml  # Uses WinML backend on Windows
```

## Cross-Platform Installation

### Python
```bash
# CPU version
pip install onnxruntime

# GPU version (CUDA)
pip install onnxruntime-gpu
```

### C/C++
Download pre-built libraries from the [releases page](https://github.com/microsoft/onnxruntime/releases) or build from source.

### JavaScript/Node.js
```bash
npm install onnxruntime-node
```

## Links and References

- [Windows Getting Started Guide](../get-started/with-windows.md)
- [WinML Principles](../../WinML_principles.md)
- [Official WinML Overview](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)
- [ONNX Runtime GitHub Releases](https://github.com/microsoft/onnxruntime/releases)