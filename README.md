<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

**ONNX Runtime is a cross-platform machine-learning inference accelerator**.

**ONNX Runtime inference** can enable faster customer experiences and lower costs, supporting models from deep learning frameworks such as PyTorch and TensorFlow/Keras as well as classical machine learning libraries such as scikit-learn, LightGBM, XGBoost, etc. ONNX Runtime is compatible with different hardware, drivers, and operating systems, and provides optimal performance by leveraging hardware accelerators where applicable alongside graph optimizations and transforms. [Learn more &rarr;](https://www.onnxruntime.ai/docs/#onnx-runtime-for-inferencing)

## Get Started & Resources

* **General Information**: [onnxruntime.ai](https://onnxruntime.ai)

* **Usage documentation and tutorials**: [onnxruntime.ai/docs](https://onnxruntime.ai/docs)

* **YouTube video tutorials**: [youtube.com/@ONNXRuntime](https://www.youtube.com/@ONNXRuntime)

* [**Upcoming Release Roadmap**](https://onnxruntime.ai/roadmap)

* **Companion sample repositories**:
  - ONNX Runtime Inferencing: [microsoft/onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples)

* **Plugin EP repositories**:
  - ONNX Runtime QNN Plugin EP: [onnxruntime/onnxruntime-qnn](https://github.com/onnxruntime/onnxruntime-qnn)

## Releases

The current release and past releases can be found here: https://github.com/microsoft/onnxruntime/releases.

For details on the upcoming release, including release dates, announcements, features, and guidance on submitting feature requests, please visit the release roadmap: https://onnxruntime.ai/roadmap.

### Training retirement

ONNX Runtime no longer supports training. Training source, ORTModule, training APIs, checkpoints, and training packages
have been removed. Training build switches now report an error when enabled; legacy CMake settings explicitly set to
`OFF` remain accepted. The training-coupled `ort_opschema_lib` target and external custom-op schema registration have also
been retired. Models using retired training operators are no longer supported. The training-specific
`ExecutionOrder.MEMORY_EFFICIENT` execution order now reports a retirement error.

Pretrained inference models, standard ONNX operator behavior (including supported `training_mode` semantics), ordinary
inference custom operators, DLPack, ATen fallback, and inference collectives remain supported. The shipped
`OrtApi::GetTrainingApi` compatibility slot remains in place and always returns `nullptr`; no inference C API fields have
been removed or reordered.

## Data/Telemetry

This project may collect usage data and send it to Microsoft to help improve our products and services. See the [privacy statement](docs/Privacy.md) for more details.

## Contributions and Feedback

We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

For feature requests or bug reports, please file a [GitHub Issue](https://github.com/Microsoft/onnxruntime/issues).

For general discussion or questions, please use [GitHub Discussions](https://github.com/microsoft/onnxruntime/discussions).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

This project is licensed under the [MIT License](LICENSE).
