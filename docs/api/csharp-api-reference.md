---
nav_exclude: true 
---

# C# API Reference

### OrtEnv
```cs
class OrtEnv
```
Holds some methods which can be used to tune the ONNX Runtime's runime environment

#### Constructor
No public constructor available.

#### Methods
```cs
static OrtEnv Instance();
```
Returns an instance of the singlton class `OrtEnv`.    

```cs
void EnableTelemetryEvents();
```
Enables platform-specific telemetry collection where applicable. Please see [Privacy](https://github.com/microsoft/onnxruntime/blob/master/docs/Privacy.md) for more details.    

```cs
void DisableTelemetryEvents();
```
Disables platform-specific telemetry collection. Please see [Privacy](https://github.com/microsoft/onnxruntime/blob/master/docs/Privacy.md) for more details.    

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
See [How to tune performance](../../how-to/tune-performance.md) for more details.

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

### ModelMetadata
```cs
class ModelMetadata
```
Encapsulates some metadata about the ONNX model.

#### Constructor
No public constructor available.

The `ModelMetadata` instance for an ONNX model may be obtained by querying the `ModelMetadata` property of an `InferenceSession` instance.
    
#### Properties
```cs
string ProducerName;
```
Holds the producer name of the ONNX model.

```cs
string GraphName;
```
Holds the graph name of the ONNX model.

```cs
string Domain;
```
Holds the opset domain of the ONNX model.

```cs
string Description;
```
Holds the description of the ONNX model.

```cs
long Version;
```
Holds the version of the ONNX model.

```cs
Dictionary<string, string> CustomMetadataMap;
```
Holds a dictionary containing key-value pairs of custom metadata held by the ONNX model.
