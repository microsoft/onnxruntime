---
title: C#
parent: Get Started
toc: true
nav_order: 4
---
# Get started with ORT for C#
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install the Nuget Packages with the .NET CLI

```bash
dotnet add package Microsoft.ML.OnnxRuntime --version 1.16.0
dotnet add package System.Numerics.Tensors --version 0.1.0
```

## Import the libraries

```csharp
using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
```

## Create method for inference

This is an [Azure Function](https://azure.microsoft.com/services/functions/) example that uses ORT with C# for inference on an NLP model created with SciKit Learn.

```csharp
 public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", "post", Route = null)] HttpRequest req,
            ILogger log, ExecutionContext context)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string review = req.Query["review"];

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            review ??= data.review;
            Debug.Assert(!string.IsNullOrEmpty(review), "Expecting a string with a content");

            // Get path to model to create inference session.
            const string modelPath = "./model.onnx";

            // Create an InferenceSession from the Model Path.
            // Creating and loading sessions are expensive per request.
            // They better be cached
            using var session = new InferenceSession(modelPath);

            // create input tensor (nlp example)
            using var inputOrtValue = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, new long[] { 1, 1 });
            inputOrtValue.StringTensorSetElementAt(review, 0);

            // Create input data for session. Request all outputs in this case.
            var inputs = new Dictionary<string, OrtValue>
            {
                { "input", inputOrtValue }
            };

            using var runOptions = new RunOptions();

            // We are getting a sequence of maps as output. We are interested in the first element (map) of the sequence.
            // That result is a Sequence of Maps, and we only need the first map from there.
            using var outputs = session.Run(runOptions, inputs, session.OutputNames);
            Debug.Assert(outputs.Count > 0, "Expecting some output");

            // We want the last output, which is the sequence of maps
            var lastOutput = outputs[outputs.Count - 1];

            // Optional code to check the output type
            {
                var outputTypeInfo = lastOutput.GetTypeInfo();
                Debug.Assert(outputTypeInfo.OnnxType == OnnxValueType.ONNX_TYPE_SEQUENCE, "Expecting a sequence");

                var sequenceTypeInfo = outputTypeInfo.SequenceTypeInfo;
                Debug.Assert(sequenceTypeInfo.ElementType.OnnxType == OnnxValueType.ONNX_TYPE_MAP, "Expecting a sequence of maps");
            }

            var elementsNum = lastOutput.GetValueCount();
            Debug.Assert(elementsNum > 0, "Expecting a non empty sequence");

            // Get the first map in sequence
            using var firstMap = lastOutput.GetValue(0, OrtAllocator.DefaultInstance);

            // Optional code just checking
            {
                // Maps always have two elements, keys and values
                // We are expecting this to be a map of strings to floats
                var mapTypeInfo = firstMap.GetTypeInfo().MapTypeInfo;
                Debug.Assert(mapTypeInfo.KeyType == TensorElementType.String, "Expecting keys to be strings");
                Debug.Assert(mapTypeInfo.ValueType.OnnxType == OnnxValueType.ONNX_TYPE_TENSOR, "Values are in the tensor");
                Debug.Assert(mapTypeInfo.ValueType.TensorTypeAndShapeInfo.ElementDataType == TensorElementType.Float, "Result map value is float");
            }

            var inferenceResult = new Dictionary<string, float>();
            // Let use the visitor to read map keys and values
            // Here keys and values are represented with the same number of corresponding entries
            // string -> float
            firstMap.ProcessMap((keys, values) => {
                // Access native buffer directly
                var valuesSpan = values.GetTensorDataAsSpan<float>();

                var entryCount = (int)keys.GetTensorTypeAndShape().ElementCount;
                inferenceResult.EnsureCapacity(entryCount);
                for (int i = 0; i < entryCount; ++i)
                {
                    inferenceResult.Add(keys.GetStringElement(i), valuesSpan[i]);
                }
            }, OrtAllocator.DefaultInstance);


            // Return the inference result as json.
            return new JsonResult(inferenceResult);

        }
```
## Reuse input/output tensor buffers

In some scenarios, you may want to reuse input/output tensors. This often happens when you want to chain 2 models (ie. feed one's output as input to another), or want to accelerate inference speed during multiple inference runs.

### Chaining: Feed model A's output(s) as input(s) to model B

```cs
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace Samples
{
    class FeedModelAToModelB
    {
        static void Program()
        {
            const string modelAPath = "./modelA.onnx";
            const string modelBPath = "./modelB.onnx";
            using InferenceSession session1 = new InferenceSession(modelAPath);
            using InferenceSession session2 = new InferenceSession(modelBPath);

            // Illustration only
            float[] inputData = { 1, 2, 3, 4 };
            long[] inputShape = { 1, 4 };

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(inputData, inputShape);

            // Create input data for session. Request all outputs in this case.
            var inputs1 = new Dictionary<string, OrtValue>
            {
                { "input", inputOrtValue }
            };

            using var runOptions = new RunOptions();

            // session1 inference
            using (var outputs1 = session1.Run(runOptions, inputs1, session1.OutputNames))
            {
                // get intermediate value
                var outputToFeed = outputs1.First();

                // modify the name of the ONNX value
                // create input list for session2
                var inputs2 = new Dictionary<string, OrtValue>
                {
                    { "inputNameForModelB", outputToFeed }
                };

                // session2 inference
                using (var results = session2.Run(runOptions, inputs2, session2.OutputNames))
                {
                    // manipulate the results
                }
            }
        }
    }
}

```
### Multiple inference runs with fixed sized input(s) and output(s)

If the model have fixed sized inputs and outputs of numeric tensors,
use the preferable **OrtValue** and its API to accelerate the inference speed and minimize data transfer.
**OrtValue** class makes it possible to reuse the underlying buffer for the input and output tensors.
It pins the managed buffers and makes use of them for inference. It also provides direct access
to the native buffers for outputs. You can also preallocate `OrtValue` for outputs or create it on top
of the existing buffers.
This avoids some overhead which may be beneficial for smaller models
where the time is noticeable in the overall running time.

Keep in mind that **OrtValue** class, like many other classes in Onnruntime C# API is **IDisposable**.
It needs to be properly disposed to either unpin the managed buffers or release the native buffers
to avoid memory leaks.

## Running on GPU (Optional)
If using the GPU package, simply use the appropriate SessionOptions when creating an InferenceSession.

```cs
int gpuDeviceId = 0; // The GPU device ID to execute on
using var gpuSessionOptoins = SessionOptions.MakeSessionOptionWithCudaProvider(gpuDeviceId);
using var session = new InferenceSession("model.onnx", gpuSessionOptoins);
```
# ONNX Runtime C# API
{: .no_toc }

The ONNX runtime provides a C# .NET binding for running inference on ONNX models in any of the .NET standard platforms.

## Supported Versions
.NET standard 1.1

## Builds

| Artifact | Description | Supported Platforms |
|-----------|-------------|---------------------|
| [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) | CPU (Release) |Windows, Linux,  Mac, X64, X86 (Windows-only), ARM64 (Windows-only)...more details: [compatibility](../reference/compatibility.md) |
| [Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu) | GPU - CUDA (Release) | Windows, Linux, Mac, X64...more details: [compatibility](../reference/compatibility.md) |
| [Microsoft.ML.OnnxRuntime.DirectML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.directml) | GPU - DirectML (Release) | Windows 10 1709+ |
| [ort-nightly](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly) | CPU, GPU (Dev), CPU (On-Device Training) | Same as Release versions |
| [Microsoft.ML.OnnxRuntime.Training](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) | CPU On-Device Training (Release) |Windows, Linux,  Mac, X64, X86 (Windows-only), ARM64 (Windows-only)...more details: [compatibility](../reference/compatibility.md) |


## API Reference
[C# API Reference](../api/csharp/api)

## Samples

See [Tutorials: Basics - C#](../tutorials/api-basics)


## Learn More
- [C# Tutorials](../tutorials/)
- [C# API Reference](../api/csharp/api)