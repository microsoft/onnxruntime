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
