---
title: Java API
parent: API docs
grand_parent: Reference
---

# ONNX Runtime Java API
{: .no_toc }

The ONNX runtime provides a Java binding for running inference on ONNX models on a JVM.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## Supported Versions
Java 8 or newer

## Builds
Release artifacts are published to **Maven Central** for use as a dependency in most Java build tools. The artifacts are built with support for some popular plaforms.

![Version Shield](https://img.shields.io/maven-central/v/com.microsoft.onnxruntime/onnxruntime)

| Artifact  | Description | Supported Platforms |
|-----------|-------------|---------------------|
| [com.microsoft.onnxruntime:onnxruntime](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime) | CPU | Windows x64, Linux x64, macOS x64 |
| [com.microsoft.onnxruntime:onnxruntime_gpu](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu) | GPU (CUDA) | Windows x64, Linux x64 |

For building locally, please see the [Java API development documentation](https://github.com/microsoft/onnxruntime/tree/master/java/README.md) for more details.

For customization of the loading mechanism of the shared library, please see [advanced loading instructions](https://github.com/microsoft/onnxruntime/tree/master/java/README.md#advanced-loading).

## API Reference

The Javadoc is available [here](https://javadoc.io/doc/com.microsoft.onnxruntime/onnxruntime).

## Sample

An example implementation is located in
[src/test/java/sample/ScoreMNIST.java](https://github.com/microsoft/onnxruntime/tree/master/java/src/test/java/sample/ScoreMNIST.java).

Once compiled the sample code expects the following arguments `ScoreMNIST
[path-to-mnist-model] [path-to-mnist] [scikit-learn-flag]`.  MNIST is expected
to be in libsvm format. If the optional scikit-learn flag is supplied the model
is expected to be produced by skl2onnx (so expects a flat feature vector, and
produces a structured output), otherwise the model is expected to be a CNN from
pytorch (expecting a `[1][1][28][28]` input, producing a vector of
probabilities).  Two example models are provided in [testdata](https://github.com/microsoft/onnxruntime/tree/master/java/testdata),
`cnn_mnist_pytorch.onnx` and `lr_mnist_scikit.onnx`. The first is a LeNet5 style
CNN trained using PyTorch, the second is a logistic regression trained using scikit-learn.

The unit tests contain several examples of loading models, inspecting input/output node shapes and types, as well as constructing tensors for scoring. 

* [https://github.com/microsoft/onnxruntime/tree/master/java/src/test/java/ai/onnxruntime/InferenceTest.java#L66](https://github.com/microsoft/onnxruntime/tree/master/java/src/test/java/ai/onnxruntime/InferenceTest.java#L66)

## Get Started

Here is simple tutorial for getting started with running inference on an existing ONNX model for a given input data. The model is typically trained using any of the well-known training frameworks and exported into the ONNX format.

Note the code presented below uses syntax available from Java 10 onwards. The Java 8 syntax is similar but more verbose.

To start a scoring session, first create the `OrtEnvironment`, then open a session using the `OrtSession` class, passing in the file path to the model as a parameter.

```java
    var env = OrtEnvironment.getEnvironment();
    var session = env.createSession("model.onnx",new OrtSession.SessionOptions());
```

Once a session is created, you can execute queries using the `run` method of the `OrtSession` object. At the moment we support `OnnxTensor` inputs, and models can produce `OnnxTensor`, `OnnxSequence` or `OnnxMap` outputs. The latter two are more likely when scoring models produced by frameworks like scikit-learn.

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

Here is a [complete sample program](https://github.com/microsoft/onnxruntime/tree/master/java/src/test/java/sample/ScoreMNIST.java) that runs inference on a pretrained MNIST model.

## Run on a GPU or with another provider (optional)

To enable other execution providers like GPUs simply turn on the appropriate flag on SessionOptions when creating an OrtSession.

```java
    int gpuDeviceId = 0; // The GPU device ID to execute on
    var sessionOptions = new OrtSession.SessionOptions();
    sessionOptions.addCUDA(gpuDeviceId);
    var session = environment.createSession("model.onnx", sessionOptions);
```

The execution providers are prioritized in the order they are enabled.
