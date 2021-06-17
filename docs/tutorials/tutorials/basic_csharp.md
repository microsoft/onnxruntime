---
nav_exclude: true 
---

# C# Tutorial: Basic

Here is simple tutorial for getting started with running inference on an existing ONNX model for a given input data. The model is typically trained using any of the well-known training frameworks and exported into the ONNX format. 

To start scoring using the model, open a session using the `InferenceSession` class, passing in the file path to the model as a parameter.


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
