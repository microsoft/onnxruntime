<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

**ONNX Runtime is a cross-platform inference and training machine-learning accelerator**.

**ONNX Runtime inference** can enable faster customer experiences and lower costs, supporting models from deep learning frameworks such as PyTorch and TensorFlow/Keras as well as classical machine learning libraries such as scikit-learn, LightGBM, XGBoost, etc. ONNX Runtime is compatible with different hardware, drivers, and operating systems, and provides optimal performance by leveraging hardware accelerators where applicable alongside graph optimizations and transforms. [Learn more &rarr;](https://www.onnxruntime.ai/docs/#onnx-runtime-for-inferencing)

**ONNX Runtime training** can accelerate the model training time on multi-node NVIDIA GPUs for transformer models with a one-line addition for existing PyTorch training scripts. [Learn more &rarr;](https://www.onnxruntime.ai/docs/#onnx-runtime-for-training)


## Get Started

**http://onnxruntime.ai/**
* [Overview](https://www.onnxruntime.ai/docs/)
* [Tutorials](https://www.onnxruntime.ai/docs/tutorials/)
  * [Inferencing](https://www.onnxruntime.ai/docs/tutorials/inferencing/)
  * [Training](https://www.onnxruntime.ai/docs/tutorials/training/)
* [How To](https://www.onnxruntime.ai/docs/how-to)
  * [Install](https://www.onnxruntime.ai/docs/how-to/install.html)
  * [Build](https://www.onnxruntime.ai/docs/how-to/build/)
  * [Tune performance](https://www.onnxruntime.ai/docs/how-to/tune-performance.html)
  * [Quantize models](https://www.onnxruntime.ai/docs/how-to/quantization.html)
  * [Deploy on mobile](https://www.onnxruntime.ai/docs/how-to/deploy-on-mobile.html)
  * [Use custom ops](https://www.onnxruntime.ai/docs/how-to/add-custom-op.html)
  * [Add a new EP](https://www.onnxruntime.ai/docs/how-to/add-execution-provider.html)
* [Reference](https://www.onnxruntime.ai/docs/reference)
  * [API documentation](https://www.onnxruntime.ai/docs/reference/api/)
  * [Execution Providers](https://www.onnxruntime.ai/docs/reference/execution-providers/)
  * [Releases and servicing](https://www.onnxruntime.ai/docs/reference/releases-servicing.html)
  * [Citing](https://www.onnxruntime.ai/docs/reference/citing.html)
* [Additional resources](https://www.onnxruntime.ai/docs/resources/)

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

Windows distributions of this project may collect usage data and send it to Microsoft to help improve our products and services. See the [privacy statement](docs/Privacy.md) for more details.

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
