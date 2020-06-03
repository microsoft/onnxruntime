# ONNX Runtime C# API
The ONNX runtime provides a C# .Net binding for running inference on ONNX models in any of the .Net standard platforms. The API is .Net standard 1.1 compliant for maximum portability. This document describes the API. 

## NuGet Package
The Microsoft.ML.OnnxRuntime Nuget package includes the precompiled binaries for ONNX runtime, and includes libraries for Windows and Linux platforms with X64 CPUs. The APIs conform to .Net Standard 1.1.

## Sample Code

The unit tests contain several examples of loading models, inspecting input/output node shapes and types, as well as constructing tensors for scoring. 

* [../csharp/test/Microsoft.ML.OnnxRuntime.Tests/InferenceTest.cs#L166](../csharp/test/Microsoft.ML.OnnxRuntime.Tests/InferenceTest.cs#L166)

## Getting Started
Here is simple tutorial for getting started with running inference on an existing ONNX model for a given input data. The model is typically trained using any of the well-known training frameworks and exported into the ONNX format. To start scoring using the model, open a session using the `InferenceSession` class, passing in the file path to the model as a parameter.

```cs
var session = new InferenceSession("model.onnx");
```

Once a session is created, you can execute queries using the `Run` method of the  `InferenceSession` object. Currently, only `Tensor` type of input and outputs  are supported. The results of the `Run` method are represented as a collection of .Net `Tensor` objects (as defined in [System.Numerics.Tensor](https://www.nuget.org/packages/System.Numerics.Tensors)).

```cs
Tensor<float> t1, t2;  // let's say data is fed into the Tensor objects
var inputs = new List<NamedOnnxValue>()
             {
                 NamedOnnxValue.CreateFromTensor<float>("name1", t1),
                 NamedOnnxValue.CreateFromTensor<float>("name2", t2)
             };
using (var results = session.Run(inputs))
{
    // manipulate the results
}
```

You can load your input data into Tensor<T> objects in several ways. A simple example is to create the Tensor from arrays.

```cs
float[] sourceData;  // assume your data is loaded into a flat float array
int[] dimensions;    // and the dimensions of the input is stored here
Tensor<float> t1 = new DenseTensor<float>(sourceData, dimensions);
```

Here is a [complete sample code](../csharp/sample/Microsoft.ML.OnnxRuntime.InferenceSample) that runs inference on a pretrained model.

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
* [../csharp/test/Microsoft.ML.OnnxRuntime.Tests/InferenceTest.cs#L1047](../csharp/test/Microsoft.ML.OnnxRuntime.Tests/InferenceTest.cs#L1047)

## Running on GPU (Optional)
If using the GPU package, simply use the appropriate SessionOptions when creating an InferenceSession.

```cs
int gpuDeviceId = 0; // The GPU device ID to execute on
var session = new InferenceSession("model.onnx", SessionOptions.MakeSessionOptionWithCudaProvider(gpuDeviceId));
```

## API Reference
### InferenceSession
```cs
class InferenceSession: IDisposable
```
The runtime representation of an ONNX model

#### Constructor
```cs
InferenceSession(string modelPath);
InferenceSession(string modelPath, SessionOptions options);
```
    
#### Properties
```cs
IReadOnlyDictionary<NodeMetadata> InputMetadata;    
```
Data types and shapes of the input nodes of the model.    

```cs
IReadOnlyDictionary<NodeMetadata> OutputMetadata; 
```
Data types and shapes of the output nodes of the model.

#### Methods
```cs
IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs);
```
Runs the model with the given input data to compute all the output nodes and returns the output node values. Both input and output are collection of NamedOnnxValue, which in turn is a name-value pair of string names and Tensor values. The outputs are IDisposable variant of NamedOnnxValue, since they wrap some unmanaged objects.

```cs
IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs, IReadOnlyCollection<string> desiredOutputNodes);
```
Runs the model on given inputs for the given output nodes only.

### System.Numerics.Tensor
The primary .Net object that is used for holding input-output of the model inference. Details on this newly introduced data type can be found in its [open-source implementation](https://github.com/dotnet/corefx/tree/master/src/System.Numerics.Tensors). The binaries are available as a [.Net NuGet package](https://www.nuget.org/packages/System.Numerics.Tensors).

### NamedOnnxValue
```cs
class NamedOnnxValue;
```
Represents a name-value pair of string names and any type of value that ONNX runtime supports as input-output data. Currently, only Tensor objects are supported as input-output values.

#### Constructor
No public constructor available.

#### Properties
```cs
string Name;   // get or set the name
```

#### Methods
```cs
static NamedOnnxValue CreateFromTensor<T>(string name, Tensor<T>);
```
Creates a NamedOnnxValue from a name and a Tensor<T> object.

```cs
Tensor<T> AsTensor<T>();
```
Accesses the value as a Tensor<T>. Returns null if the value is not a Tensor<T>.     

### DisposableNamedOnnxValue
```cs
class DisposableNamedOnnxValue: NamedOnnxValue, IDisposable;
```
This is a disposable variant of NamedOnnxValue, used for holding output values which contains objects allocated in unmanaged memory. 

### FixedBufferOnnxValue
```cs
class FixedBufferOnnxValue: IDisposable;
```
Class `FixedBufferOnnxValue` enables the availability to pin the tensor buffer. This helps to minimize overhead within each inference run.

`FixedBufferOnnxValue` can be used as either input or output. However, if used as output, it has to be a numeric tensor.

`FixedBufferOnnxValue` implements `IDisposable`, so make sure it get disposed after use.
#### Methods
```cs
static FixedBufferOnnxValue CreateFromTensor<T>(Tensor<T>);
```
Creates a FixedBufferOnnxValue from a name and a Tensor<T> object.


### IDisposableReadOnlyCollection
```cs
interface IDisposableReadOnlyCollection: IReadOnlyCollection, IDisposable
```
Collection interface to hold disposable values. Used for output of Run method.

### SessionOptions
```cs
class SessionOptions: IDisposable;
```
A collection of properties to be set for configuring the OnnxRuntime session

#### Constructor
```cs
SessionOptions();
```
Constructs a SessionOptions will all options at default/unset values.

#### Properties
```cs
static SessionOptions Default;   //read-only
```
Accessor to the default static option object

#### Methods
```cs
SetSessionGraphOptimizationLevel(GraphOptimizationLevel graph_transformer_level);
```
See [ONNX_Runtime_Graph_Optimizations.md] for more details.

```cs
SetSessionExecutionMode(ExecutionMode execution_mode);
```
 * ORT_SEQUENTIAL - execute operators in the graph sequentially.
 * ORT_PARALLEL - execute operators in the graph in parallel.   
See [ONNX_Runtime_Perf_Tuning.md] for more details.

### NodeMetadata
Container of metadata for a model graph node, used for communicating the shape and type of the input and output nodes.

#### Properties
```cs
int[] Dimensions;  
```
Read-only shape of the node, when the node is a Tensor. Undefined if the node is not a Tensor.
    
```cs
System.Type ElementType;
```
Type of the elements of the node, when node is a Tensor. Undefined for non-Tensor nodes.

```cs
bool IsTensor;
```
Whether the node is a Tensor

### Exceptions
```cs
class OnnxRuntimeException: Exception;
```

The type of Exception that is thrown in most of the error conditions related to Onnx Runtime.



