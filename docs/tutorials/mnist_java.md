---
nav_exclude: true 
---

# Character recognition with MNIST in Java
{: .no_toc }

Here is simple tutorial for getting started with running inference on an existing ONNX model for a given input data. The model is typically trained using any of the well-known training frameworks and exported into the ONNX format.

Note the code presented below uses syntax available from Java 10 onwards. The Java 8 syntax is similar but more verbose.
To start a scoring session, first create the `OrtEnvironment`, then open a session using the `OrtSession` class, passing in the file path to the model as a parameter.

```java
    var env = OrtEnvironment.getEnvironment();
    var session = env.createSession("model.onnx",new OrtSession.SessionOptions());
```

Once a session is created, you can execute queries using the `run` method of the `OrtSession` object. 
At the moment we support `OnnxTensor` inputs, and models can produce `OnnxTensor`, `OnnxSequence` or `OnnxMap` outputs. The latter two are more likely when scoring models produced by frameworks like scikit-learn.
The run call expects a `Map<String,OnnxTensor>` where the keys match input node names stored in the model. These can be viewed by calling `session.getInputNames()` or `session.getInputInfo()` on an instantiated session.
The run call produces a `Result` object, which contains a `Map<String,OnnxValue>` representing the output. The `Result` object is `AutoCloseable` and can be used in a try-with-resources statement to 
prevent references from leaking out. Once the `Result` object is closed, all it's child `OnnxValue`s are closed too.

```java
    OnnxTensor t1,t2;
    var inputs = Map.of("name1",t1,"name2",t2);
    try (var results = session.run(inputs)) {
    // manipulate the results
   }
```

You can load your input data into OnnxTensor objects in several ways. The most efficient way is to use a `java.nio.Buffer`, but it's possible to use multidimensional arrays too. If constructed using arrays the arrays must not be ragged.

```java
    FloatBuffer sourceData;  // assume your data is loaded into a FloatBuffer
    long[] dimensions;       // and the dimensions of the input are stored here
    var tensorFromBuffer = OnnxTensor.createTensor(env,sourceData,dimensions);

    float[][] sourceArray = new float[28][28];  // assume your data is loaded into a float array 
    var tensorFromArray = OnnxTensor.createTensor(env,sourceArray);
```

Here is a [complete sample program](https://github.com/microsoft/onnxruntime/blob/master/java/src/test/java/sample/ScoreMNIST.java) that runs inference on a pretrained MNIST model.

## Running on a GPU or with another provider (Optional)

To enable other execution providers like GPUs simply turn on the appropriate flag on SessionOptions when creating an OrtSession.

```java
    int gpuDeviceId = 0; // The GPU device ID to execute on
    var sessionOptions = new OrtSession.SessionOptions();
    sessionOptions.addCUDA(gpuDeviceId);
    var session = environment.createSession("model.onnx", sessionOptions);
```

The execution providers are preferred in the order they were enabled.
