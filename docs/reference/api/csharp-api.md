---
title: C# API
parent: API docs
grand_parent: Reference
---

# ONNX Runtime C# API
{: .no_toc }

The ONNX runtime provides a C# .NET binding for running inference on ONNX models in any of the .NET standard platforms.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Supported Versions
.NET standard 1.1

## Builds

| Artifact | Description | Supported Platforms |
|-----------|-------------|---------------------|
| [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) | CPU (Release) |Windows, Linux,  Mac, X64, X86 (Windows-only), ARM64 (Windows-only)...more details: [compatibility](../../resources/compatibility.md) |
| [Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu) | GPU - CUDA (Release) | Windows, Linux, Mac, X64...more details: [compatibility](../../resources/compatibility.md) |
| [Microsoft.ML.OnnxRuntime.DirectML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.directml) | GPU - DirectML (Release) | Windows 10 1709+ |
| [ort-nightly](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly) | CPU, GPU (Dev) | Same as Release versions |


## API Reference
[C# API Reference](./csharp-api-reference.html)

## Reuse input/output tensor buffers

In some scenarios, you may want to reuse input/output tensors. This often happens when you want to chain 2 models (ie. feed one's output as input to another), or want to accelerate inference speed during multiple inference runs.

### Chaining: Feed model A's output(s) as input(s) to model B

```cs
InferenceSession session1, session2;  // let's say 2 sessions are initialized

Tensor<float> t1;  // let's say data is fed into the Tensor objects
var inputs1 = new List<NamedOnnxValue>()
              {
                  NamedOnnxValue.CreateFromTensor<float>("name1", t1)
              };
// session1 inference
using (var outputs1 = session1.Run(inputs1))
{
    // get intermediate value
    var input2 = outputs1.First();
    
    // modify the name of the ONNX value
    input2.Name = "name2";

    // create input list for session2
    var inputs2 = new List<NamedOnnxValue>() { input2 };

    // session2 inference
    using (var results = session2.Run(inputs2))
    {
        // manipulate the results
    }
}
```

### Multiple inference runs with fixed sized input(s) and output(s)

If the model have fixed sized inputs and outputs of numeric tensors, you can use "FixedBufferOnnxValue" to accelerate the inference speed. By using "FixedBufferOnnxValue", the container objects only need to be allocated/disposed one time during multiple InferenceSession.Run() calls. This avoids some overhead which may be beneficial for smaller models where the time is noticeable in the overall running time.

An example can be found at `TestReusingFixedBufferOnnxValueNonStringTypeMultiInferences()`:
* [Microsoft.ML.OnnxRuntime.Tests/InferenceTest.cs#L1047](https://github.com/microsoft/onnxruntime/tree/master/csharp/test/Microsoft.ML.OnnxRuntime.Tests/InferenceTest.cs#L1047)

## Running on GPU (Optional)
If using the GPU package, simply use the appropriate SessionOptions when creating an InferenceSession.

```cs
int gpuDeviceId = 0; // The GPU device ID to execute on
var session = new InferenceSession("model.onnx", SessionOptions.MakeSessionOptionWithCudaProvider(gpuDeviceId));
```
## Samples

See [Tutorials: Basics - C#](../../tutorials/inferencing/api-basics.md#c-2)
