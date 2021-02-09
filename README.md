<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20CPU%20CI%20Pipeline?label=Windows+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=9)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20CI%20Pipeline?label=Windows+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=10)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20CI%20Pipeline?label=Linux+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=11)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20CI%20Pipeline?label=Linux+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=12)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20CI%20Pipeline?label=MacOS+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=13)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining-linux-ci-pipeline?label=Linux+CPU+Training)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=86)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining-linux-gpu-ci-pipeline?label=Linux+GPU+Training)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=84)

**ONNX Runtime** is a cross-platform **inference and training machine-learning accelerator** compatible with deep learning frameworks, PyTorch and TensorFlow/Keras, as well as classical machine learning libraries such as scikit-learn, and more. **[aka.ms/onnxruntime](https://aka.ms/onnxruntime)**

ONNX Runtime uses the portable [ONNX](https://onnx.ai) computation graph format, backed by execution providers optimized for operating systems, drivers and hardware.

Many users can benefit from ONNX Runtime, including those looking to:

* Improve inference performance for a wide variety of ML models
* Reduce time and cost of training large models
* Train in Python but deploy into a C#/C++/Java app
* Run on different hardware and operating systems
* Support models created in several different frameworks

[ONNX Runtime inference](./onnxruntime) APIs are stable and production-ready since the [1.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.0.0) in October 2019 and can enable faster customer experiences and lower costs.

[ONNX Runtime training](./orttraining) feature was introduced in May 2020 in preview. This feature supports acceleration of PyTorch training on multi-node NVIDIA GPUs for transformer models. Additional updates for this feature are coming soon.


***

# Contents

* **[Get Started](#get-started)**
* **[Data/Telemetry](#DataTelemetry)**
* **[Contributions and Feedback](#contributions-and-feedback)**
* **[License](#license)**

***

# Get Started
**http://onnxruntime.ai/**
* [Install](https://www.onnxruntime.ai/docs/get-started/install.html)
* [Inference](https://www.onnxruntime.ai/docs/get-started/inference.html)
* [Training](https://www.onnxruntime.ai/docs/get-started/training.html)
* [Documentation](https://www.onnxruntime.ai/docs/)
* [Samples and Tutorials](https://www.onnxruntime.ai/docs/tutorials/)
* [Frequently Asked Questions](./docs/FAQ.md)


# Data/Telemetry

This project may collect usage data and send it to Microsoft to help improve our products and services. See the [privacy statement](docs/Privacy.md) for more details.

# Contributions and Feedback

We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

For any feedback or to report a bug, please file a [GitHub Issue](https://github.com/Microsoft/onnxruntime/issues).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# License

This project is licensed under the [MIT License](LICENSE).
