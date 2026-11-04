**Please refer to these repositories for ONNX Runtime samples:**

- ONNX Runtime Inferencing: [microsoft/onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples)
- ONNX Runtime Training: [microsoft/onnxruntime-training-examples](https://github.com/microsoft/onnxruntime-training-examples)

Native stateful Python execution has a local
[tiny-model example](python/stateful_execution.py) covering feedback loops,
sampler-free prefill/decode bootstrap and deterministic DDIM. See
[Stateful Python execution](../docs/Stateful_Python.md) for requirements and commands.

For real local CPU INT4/CUDA FP16 text generation, see the
[Phi-3 stateful demo](../docs/Stateful_Phi3_Demo.md) and its
[CLI/helpers](python/phi3_demo.py). Model acquisition is explicit and outside Git.

For CUDA-only local SD1.5 text-to-image with native DDIM and mandatory safety
checking, see the [SD1.5 stateful demo](../docs/Stateful_SD15_Demo.md) and its
[CLI/helpers](python/sd15_demo.py). CPU EP fixtures remain available offline.

The [native qualification helper](python/stateful_qualification.py) records
completed-work latency, separate step/resource observations and checked report
artifacts from explicit local packages. See the
[delivered-foundation qualification report](../docs/Stateful_Qualification.md)
for the fixed Linux CPU/CUDA workloads and current qualification status.
