# Getting Started with ONNX Runtime on Windows

## Recommended Approach: WinML

For Windows developers, **WinML (Windows Machine Learning)** is the **preferred and future-safe execution path** for ONNX Runtime deployments. WinML provides an optimized experience specifically designed for Windows environments.

### Why WinML?

WinML offers several key advantages for Windows developers:

- **Same ONNX Runtime APIs**: Uses the familiar ONNX Runtime APIs you already know
- **Automatic Hardware Optimization**: Dynamically selects the best execution provider (EP) based on available hardware
- **Simplified Deployment**: Streamlined setup and distribution for Windows applications
- **Windows Integration**: Built-in Windows support with optimal performance
- **Future-Safe**: Actively maintained as Microsoft's primary Windows ML solution

### Quick Start with WinML

1. **Installation**: See the [WinML installation guide](../install/index.md#winml-installs)
2. **Documentation**: Visit the [official WinML overview](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)
3. **Examples**: Check out WinML samples in the repository

### WinML Relationship to ONNX Runtime

WinML is built on top of ONNX Runtime and provides:

- **API Compatibility**: Uses the same ONNX Runtime APIs for seamless integration
- **Smart EP Selection**: Automatically chooses the optimal execution provider (CPU, GPU, NPU, etc.) based on your hardware
- **Windows Optimization**: Leverages Windows-specific optimizations and hardware acceleration
- **Simplified Packaging**: Handles distribution and deployment complexities for Windows applications

## Alternative Approaches

### DirectML (Deprecated)

> **⚠️ DEPRECATION NOTICE**: DirectML execution provider is deprecated. Please migrate to WinML for new projects.

If you're currently using DirectML, we recommend migrating to WinML. See the [DirectML documentation](../execution-providers/DirectML-ExecutionProvider.md) for migration guidance.

### Standard ONNX Runtime

For cross-platform scenarios or when you need specific execution provider control, you can still use standard ONNX Runtime. However, for Windows-specific applications, WinML provides better integration and performance.

## Next Steps

1. [Install WinML](../install/index.md#winml-installs)
2. Review the [WinML principles](../../WinML_principles.md)
3. Explore the official [Windows ML documentation](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)
4. Try the sample applications in the repository