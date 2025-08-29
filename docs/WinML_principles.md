# Contributing to Windows ML

Windows Machine Learning is a high-performance, reliable API for deploying hardware-accelerated ML inferences on Windows devices. WinML is the **preferred and future-safe execution path** for Windows developers using ONNX Runtime.

## WinML and ONNX Runtime

WinML is built on ONNX Runtime and provides Windows developers with:

- **Same ONNX Runtime APIs**: Uses the familiar ONNX Runtime APIs you already know, ensuring easy migration and compatibility
- **Dynamic Execution Provider Selection**: Automatically selects the best execution provider (EP) based on available hardware (CPU, GPU, NPU, etc.)
- **Simplified Windows Deployment**: Streamlines deployment for Windows developers by handling distribution and packaging complexities
- **Windows-Optimized Performance**: Leverages Windows-specific optimizations and hardware acceleration capabilities

Please visit the [Windows ML documentation](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview) to learn more about Windows ML.


## Windows ML Base Principles

**We design and optimize for all Windows devices.**

  Our goal is to provide developers with a platform that enables new experiences that run well on all Windows devices. WinML is designed as the unified solution that gives developers confidence that their applications will work optimally for all Windows customers.

**We maintain and curate the Windows ML APIs.**

  The API is designed to ensure consistency of developer’s experience across the Windows platform. We provide long-term servicing and support, and we are committed to ensuring application’s compatibility as we evolve the API.

**Windows ML is a core component of Windows.**

  The Windows ML code is packaged and distributed with each new release of Windows. To provide consumers with high-quality products, Microsoft is responsible for distributing Windows ML and related binaries as part of Windows or standalone distributable packages.


## Open for Community Contributions

We encourage community contributions to Windows ML to enhance users’ experience on Windows. We use the principles above to guide how we look at and evaluate all contributions.

Ensure your feature request follows all these principles to help the review process and include information about the customer problem(s) the feature request addresses.

Note: minor issues or bugs can be addressed more quickly using the [bug/performance issue request](https://github.com/microsoft/onnxruntime/issues/new/choose) rather than feature request.

## Start your Feature Request

If you'd like to contribute to Windows ML and engage with the community to get feedback, please review to the contributing [process details](https://github.com/microsoft/onnxruntime/blob/main/CONTRIBUTING.md) and submit a new feature request [here](https://github.com/microsoft/onnxruntime/issues/new/choose).
