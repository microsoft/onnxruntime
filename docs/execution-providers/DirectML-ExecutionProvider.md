# DirectML Execution Provider

> **⚠️ DEPRECATION NOTICE**: DirectML is deprecated. Please use [WinML](#winml-recommendation) for Windows-based ONNX Runtime deployments.

## Overview

The DirectML Execution Provider enables hardware acceleration for ONNX Runtime on Windows using DirectML. However, this execution provider is now deprecated in favor of WinML, which provides a more integrated and future-safe solution for Windows developers.

## WinML Recommendation

For Windows-based ONNX Runtime deployments, we strongly recommend using **WinML** instead of DirectML. WinML offers several advantages:

- **Same ONNX Runtime APIs**: WinML uses the familiar ONNX Runtime APIs you already know
- **Dynamic EP Selection**: Automatically selects the best execution provider based on available hardware
- **Simplified Deployment**: Streamlined experience specifically designed for Windows developers
- **Future-Safe**: Actively maintained and supported as the primary Windows ML solution

### Getting Started with WinML

To get started with WinML on Windows, please refer to:

- [Windows Getting Started Guide](../get-started/with-windows.md)
- [WinML Installation Instructions](../install/index.md#winml-installs)
- [Official WinML Overview](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)

## Migration from DirectML

If you're currently using DirectML, migrating to WinML is straightforward since it uses the same ONNX Runtime APIs. The main change is in how you configure and initialize the runtime environment.

For specific migration guidance and examples, please consult the WinML documentation linked above.