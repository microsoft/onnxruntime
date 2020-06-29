# ONNX Runtime Roadmap
ONNX Runtime is an active, fast-paced project backed by a strong team of Microsoft engineers and data scientists along with a worldwide community of partners and contributors. This roadmap summarizes the pending investments identified by the team to continually grow 
ONNX Runtime as a robust, versatile, and high performance inference engine for DNN and traditional ML models.

## High Level Goals
ONNX Runtime is a runtime accelerator that supports interoperable ML and DNN models based on the [ONNX](https://onnx.ai/) spec. For key technical design objectives and considerations, see [ONNX Runtime Inference High Level Design](./InferenceHighLevelDesign.md).

We recognize the challenges involved in operationalizing ML models performantly in an agile way, and we understand that high volume production services can be highly performance-sensitive and often need to support a variety of compute targets (we experience these first-hand at Microsoft across our vast array of products and services). 

As such, our investments are directly in support of solving those challenges, focusing on areas such as:
* Platform coverage
* Extensibility and customization
* Performance (latency, memory, throughput, scale, etc)
* Model coverage
* Quality and ease of use - including backwards compatibility of models (older opsets) and APIs

In addition to our OSS participation, we also internally use this technology in core products at Microsoft, with over 80 models in production providing an average of 2x+ performance improvement. 

## Investments
In support of the high level goals outlined above, the investment areas listed below represent our active and backlog projects, 
which are largely driven by community demand and anticipated usage opportunities. We will work through our prioritized backlog as 
quickly as possible, and if there are any specific features or enhancements you need, we gladly welcome community contributions for 
these efforts or any of the [enhancements suggested on Github](https://github.com/microsoft/onnxruntime/issues?q=is%3Aopen+is%3Aissue+label%3Aenhancement). If you have a specific suggestion or unsupported use case, please let us 
know by filing a [Github issue](https://github.com/microsoft/onnxruntime/issues).

* **Platform coverage** - [Expanded platform compatibility](#expanded-platform-compatibility)
* **Extensibility and customization** - [Accelerators and Execution Providers](#accelerators-and-execution-providers)
* **Performance** - [Continued performance optimizations](#continued-performance-optimizations) 
* **Model coverage** - [Expanded model compatibility](#expanded-model-compatibility)  
* **Quality and ease of use** - [Increased integration with popular ML products](#increased-integration-with-popular-products)
---

### Expanded platform compatibility
ONNX Runtime already supports a wide range of architectures, platforms, and languages, and this will continue to be an active investment area to broaden the availability of the engine for varied usage. 
Additionally, we understand that lightweight devices and local applications may have constraints for package size, so there is active awareness to opportunistically minimize binary size.

#### Architectures
Supported
* X64
* X86
* ARM64
* ARM32 (Limited)

#### Platforms
Supported
* Windows 7+
* Linux (various)
* Mac OS X
* Android (community contribution, Preview)

*Future*
* *iOS*

#### Languages
Supported languages are listed in [API Documentation](../README.md#api-documentation). The core team is not actively working on other language bindings at this time. If there is a missing API, please file a request in [Issues](https://github.com/microsoft/onnxruntime/issues). Community contributions are welcome for other languages.

### Accelerators and Execution Providers
#### New EPs
To achieve the best performance on a growing set of compute targets across cloud and the intelligent edge, we invest in and partner with hardware partners and community members to add new execution providers. The flexible pluggability of ONNX Runtime is critical to support a broad range of scenarios and compute options.

Supported

Supported EPs are listed [here](../README.md#supported-accelerators). Upcoming EPs include:
* Xilinx FPGA


#### CUDA operator coverage
To maximize performance potential, we will be continually adding additional CUDA implementations for supported operators.

#### Simplify EP contributions
In addition to new execution providers, we aim to make it easy for community partners to contribute in a non-disruptive way. To support this, we are investing in improvements to the execution provider interface for easily registering new execution providers and separating out EPs from the core runtime engine.

### Continued Performance Optimizations
Performance is a key focus for ONNX Runtime. From latency to memory utilization to CPU usage, we are constantly seeking strategies to deliver the best performance. Although DNNs are rapidly driving research areas for innovation, we acknowledge that in practice, many companies and developers are still using traditional ML frameworks for reasons ranging from expertise to privacy to legality. As such, ONNX Runtime is focused on improvements and support for both DNNs and traditional ML. 

#### Examples of projects the team is working on:
* Improvements to batch processing for scikit-learn models
* More quantization support
* Improved multithreading (e.g. smarter work sharding, user supplied thread pools, etc)
* Graph optimizations
* Intelligent graph partitioning to maximize the value of different accelerators

#### Optimizations for mobile and IoT Edge devices
IoT provides growing opportunity to execute ML workloads on the edge of the network, where the data is collected. However, the devices used for ML execution have different hardware specifications. To support compatibility with this group of devices, we will invest in strategies to optimize ONNX model execution across the breadth of IoT endpoints using different hardware configurations with CPUs, GPUs and custom NN ASICs.

### Expanded model compatibility
The ONNX spec focuses on ML model interoperability rather than coverage of all operators from all frameworks. 
We aim to continuously improve coverage to support popular as well as new state-of-the-art models.

#### Spec coverage 
As more operators are added to the ONNX spec, ONNX Runtime will provide implementations (default CPU and GPU-CUDA) of each to stay in compliance with the latest ONNX spec.

This includes:
* Sparse Tensor support

#### Investments in popular converters
We work with the OSS and ONNX community to ensure popular frameworks can export or be converted to ONNX format. 
* [PyTorch export](https://pytorch.org/docs/stable/onnx.html)
* [Tensorflow-ONNX](https://github.com/onnx/tensorflow-onnx)
* [Keras-ONNX](https://github.com/onnx/keras-onnx)
* [Sklearn-ONNX](https://github.com/onnx/sklearn-onnx)
* [ONNXMLTools](https://github.com/onnx/onnxmltools/tree/master/onnxmltools/convert) (CoreML, XGBoost, LibSVM, LightGBM, SparkML)
* [ML.NET](https://github.com/dotnet/machinelearning)

#### Improved error handling
To decrease the risk of model inferencing failures, we will improve the error handling and fallback strategies for missing types or unsupported operators. For EPs that have missing or incorrect implementations for ONNX operators, we aim to fallback or fail as gracefully as possible.

#### Community-driven feature additions
Focusing on practicality, we take a scenario driven approach to adding additional capabilities to ONNX Runtime.

### Increased integration with popular products
We understand that data scientists and ML engineers work with many different products and toolsets to bring complex machine learning 
algorithms to life through innovative user-facing applications. We want to ensure ONNX Runtime works as seamlessly as possible with 
these. If you've identified any integration ideas or opportunities and have questions or need assistance, we encourage use of Github Issues as a discussion forum.

Some of these products include:
* [AzureML](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-onnx): simplify the process to train, convert, and deploy ONNX models to Azure
* [Model Interpretability](https://docs.microsoft.com/en-us/azure/machine-learning/service/machine-learning-interpretability-explainability): explainability for ONNX models
* [ML.NET](https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/object-detection-onnx): inference ONNX models in .NET
* [PyTorch](https://pytorch.org/docs/stable/onnx.html): improve coverage for exporting trained models to ONNX
* [Windows](https://docs.microsoft.com/en-us/windows/ai/windows-ml/index): run ONNX models on Windows devices using the built-in Windows ML APIs. Windows ML APIs will be included in the ONNX Runtime builds and binaries to enable Windows developers to get OS-independent updates
* [SQL Database Edge](https://docs.microsoft.com/en-us/azure/sql-database-edge/deploy-onnx): predict with ONNX models in SQL Database Edge, an optimized relational database engine geared for IoT and IoT Edge deployments 

Have an idea or feature request? [Contribute](https://github.com/microsoft/onnxruntime/blob/master/CONTRIBUTING.md) or [let us know](https://github.com/microsoft/onnxruntime/blob/master/.github/ISSUE_TEMPLATE/feature_request.md)!
