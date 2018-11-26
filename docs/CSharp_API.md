# ONNX Runtime C# API
The ONNX runtime provides a C# .Net binding for running inference on ONNX models in any of the .Net standard platforms. The API is .Net standard 1.1 compliant for maximum portability. This document describes the API. 

## NuGet Package
There is a NuGet package Microsoft.ML.OnnxRuntime available for .Net consumers, which includes the prebuilt binaries for ONNX runtime.  The API is portable across all platforms and architectures supported by the .Net standard, although currently the NuGet package contains the prebuilt binaries for Windows 10 platform on x64 CPUs only.

## Getting Started
Here is simple tutorial for getting started with running inference on an existing ONNX model for a given input data (a.k.a query). Say the model is trained using any of the well-known training frameworks and exported as an ONNX model into a file named `model.onnx`. The runtime incarnation of a model is an `InferenceSession` object. You simply construct an `InferenceSession` object using the model file as parameter --
    
    var session = new InferenceSession("model.onnx");

Once a session is created, you can run queries on the session using your input data, using the `Run` method of the  `InferenceSession`. Both input and output of `Run` method are represented as collections of .Net `Tensor` objects (as defined in [System.Numerics.Tensor](https://www.nuget.org/packages/System.Numerics.Tensors)) -
    
    Tensor<float> t1, t2;  // let's say data is fed into the Tensor objects
    var inputs = new List<NamedOnnxValue>()
                 {
                    NamedOnnxValue.CreateFromTensor<float>("name1", t1),
                    NamedOnnxValue.CreateFromTensor<float>("name2", t2)
                 };
    IReadOnlyCollection<NamedOnnxValue> results = session.Run(inputs);

You can load your input data into Tensor<T> objects in several ways. A simple example is to create the Tensor from arrays -
    float[] sourceData;  // assume your data is loaded into a flat float array
    int[] dimensions;    // and the dimensions of the input is stored here
    Tensor<float> t1 = new DenseTensor<float>(sourceData, dimensions);    

Here is a [complete sample code](https://github.com/Microsoft/onnxruntime/tree/master/csharp/sample/Microsoft.ML.OnnxRuntime.InferenceSample) that runs inference on a pretrained model.


## API Reference
### InferenceSession
    class InferenceSession: IDisposable
The runtime representation of an ONNX model

#### Constructor
    InferenceSession(string modelPath);
    InferenceSession(string modelPath, SesionOptions options);
    
#### Properties
    IReadOnlyDictionary<NodeMetadata> InputMetadata;    
Data types and shapes of the input nodes of the model.    
    IReadOnlyDictionary<NodeMetadata> OutputMetadata; 
Data types and shapes of the output nodes of the model.

#### Methods
    IReadOnlyCollection<NamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs);
Runs the model with the given input data to compute all the output nodes and returns the output node values. Both input and output are collection of NamedOnnxValue, which in turn is a name-value pair of string names and Tensor values.

    IReadOnlyCollection<NamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs, IReadOnlyCollection<string> desiredOutputNodes);
Runs the model on given inputs for the given output nodes only.

### System.Numerics.Tensor
The primary .Net object that is used for holding input-output of the model inference. Details on this newly introduced data type can be found in its [open-source implementation](https://github.com/dotnet/corefx/tree/master/src/System.Numerics.Tensors). The binaries are available as a [.Net NuGet package](https://www.nuget.org/packages/System.Numerics.Tensors).

### NamedOnnxValue
    class NamedOnnxValue;
Represents a name-value pair of string names and any type of value that ONNX runtime supports as input-output data. Currently, only Tensor objects are supported as input-output values.

#### Constructor
    No public constructor available.

#### Properties
    string Name;   // read only

#### Methods
    static NamedOnnxValue CreateFromTensor<T>(string name, Tensor<T>);
Creates a NamedOnnxValue from a name and a Tensor<T> object.

    Tensor<T> AsTensor<T>();
Accesses the value as a Tensor<T>. Returns null if the value is not a Tensor<T>.     


### SessionOptions
    class SessionOptions: IDisposable;
A collection of properties to be set for configuring the OnnxRuntime session

#### Constructor
    SessionOptions();
Constructs a SessionOptions will all options at default/unset values.

#### Properties
    static SessionOptions Default;   //read-only
Accessor to the default static option object

#### Methods
    AppendExecutionProvider(ExecutionProvider provider);
Appends execution provider to the session. For any operator in the graph the first execution provider that implements the operator will be user. ExecutionProvider is defined as the following enum -

    enum ExecutionProvider
    {
        Cpu,
        MklDnn
    }

### NodeMetadata
Container of metadata for a model graph node, used for communicating the shape and type of the input and output nodes.

#### Properties
    int[] Dimensions;  
Read-only shape of the node, when the node is a Tensor. Undefined if the node is not a Tensor.
    
    System.Type ElementType;
Type of the elements of the node, when node is a Tensor. Undefined for non-Tensor nodes.

    bool IsTensor;
Whether the node is a Tensor

### Exceptions
    class OnnxRuntimeException: Exception;

The type of Exception that is thrown in most of the error conditions related to Onnx Runtime.




