<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

**ONNX Runtime** is a cross-platform **inference and training machine-learning accelerator** compatible with deep learning frameworks, PyTorch and TensorFlow/Keras, as well as classical machine learning libraries such as scikit-learn, and more.

ONNX Runtime uses the portable [ONNX](https://onnx.ai) computation graph format, backed by execution providers optimized for operating systems, drivers and hardware.

Common use cases for ONNX Runtime:

* Improve inference performance for a wide variety of ML models
* Reduce time and cost of training large models
* Train in Python but deploy into a C#/C++/Java app
* Run with optimized performance on different hardware and operating systems
* Support models created in several different frameworks

[ONNX Runtime inference](https://www.onnxruntime.ai/docs/get-started/inference.html) APIs are stable and production-ready since the [1.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.0.0) in October 2019 and can enable faster customer experiences and lower costs.

[ONNX Runtime training](https://www.onnxruntime.ai/docs/get-started/training.html) feature was introduced in May 2020 in preview. This feature supports acceleration of PyTorch training on multi-node NVIDIA GPUs for transformer models. Additional updates for this feature are coming soon.


## Get Started

**http://onnxruntime.ai/**
* [Install](https://www.onnxruntime.ai/docs/get-started/install.html)
* [Inference](https://www.onnxruntime.ai/docs/get-started/inference.html)
* [Training](https://www.onnxruntime.ai/docs/get-started/training.html)
* [Documentation](https://www.onnxruntime.ai/docs/)
* [Samples and Tutorials](https://www.onnxruntime.ai/docs/tutorials/)
* [Build Instructions](https://www.onnxruntime.ai/docs/how-to/build.html)
* [Frequently Asked Questions](./docs/FAQ.md)

## Build Pipeline Status
|System|CPU|GPU|EPs|
|---|---|---|---|
|Windows|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20CPU%20CI%20Pipeline?label=Windows+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=9)|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20CI%20Pipeline?label=Windows+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=10)|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20TensorRT%20CI%20Pipeline?label=Windows+GPU+TensorRT)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=47)|
|Linux|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20CI%20Pipeline?label=Linux+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=11)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20Minimal%20Build%20E2E%20CI%20Pipeline?label=Linux+CPU+Minimal+Build)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=64)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20x64%20NoContribops%20CI%20Pipeline?label=Linux+CPU+x64+No+Contrib+Ops)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=110)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/centos7_cpu?label=Linux+CentOS7)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=78)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining-linux-ci-pipeline?label=Linux+CPU+Training)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=86)|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20CI%20Pipeline?label=Linux+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=12)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20TensorRT%20CI%20Pipeline?label=Linux+GPU+TensorRT)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=45)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining-distributed?label=Distributed+Training)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=140)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining-linux-gpu-ci-pipeline?label=Linux+GPU+Training)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=84)|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20NUPHAR%20CI%20Pipeline?label=Linux+NUPHAR)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=110)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20OpenVINO%20CI%20Pipeline%20v2?label=Linux+OpenVINO)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=108)|
|Mac|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20CI%20Pipeline?label=MacOS+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=13)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20NoContribops%20CI%20Pipeline?label=MacOS+NoContribops)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=65)|||
|Android|||[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Android%20CI%20Pipeline?label=Android)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=53)|
|iOS|||[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/iOS%20CI%20Pipeline?label=iOS)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=134)|
|WebAssembly|||[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20WebAssembly%20CI%20Pipeline?label=WASM)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=161)|


## Data/Telemetry

This project may collect usage data and send it to Microsoft to help improve our products and services. See the [privacy statement](docs/Privacy.md) for more details.

## Contributions and Feedback

We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

For feature requests or bug reports, please file a [GitHub Issue](https://github.com/Microsoft/onnxruntime/issues).

For general discussion or questions, please use [Github Discussions](https://github.com/microsoft/onnxruntime/discussions).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

This project is licensed under the [MIT License](LICENSE).
