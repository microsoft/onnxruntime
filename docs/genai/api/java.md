---
title: Java API
description: Java API reference for ONNX Runtime generate() API
has_children: false
parent: API docs
grand_parent: Generate API (Preview)
nav_order: 4
---

# ONNX Runtime generate() Java API

_Note: this API is in preview and is subject to change._

{: .no_toc }

* TOC placeholder
{:toc}

## Overview

This document describes the Java API for ONNX Runtime GenAI.  
Below are the main classes and methods, with code snippets and descriptions for each.

---

## Install and import

The Java API is delivered by the `ai.onnxruntime.genai` Java package. Package publication is pending. To build the package from source, see the [build from source guide](../howto/build-from-source.md).

```java
import ai.onnxruntime.genai.*;
```

---

## Model class

### Constructor

Initializes a new model from the given model path.

```java
public Model(String modelPath) throws GenAIException
```

---

### createGeneratorParams

Creates a GeneratorParams instance for executing the model.

```java
public GeneratorParams createGeneratorParams() throws GenAIException
```

---

### createTokenizer

Creates a Tokenizer instance for this model.

```java
public Tokenizer createTokenizer() throws GenAIException
```

---

### generate

Generates output sequences using the provided generator parameters.

```java
public Sequences generate(GeneratorParams generatorParams) throws GenAIException
```

---

## Config class

### Constructor

Initializes a new configuration object from a config path.

```java
public Config(String configPath) throws GenAIException
```

---

### clearProviders

Clears all providers from the configuration.

```java
public void clearProviders() throws GenAIException
```

---

### appendProvider

Appends a provider to the configuration.

```java
public void appendProvider(String provider) throws GenAIException
```

---

### setProviderOption

Sets a provider option in the configuration.

```java
public void setProviderOption(String provider, String name, String value) throws GenAIException
```

---

### overlay

Overlays a JSON string onto the configuration.

```java
public void overlay(String json) throws GenAIException
```

---

## Tokenizer class

### Constructor

Initializes a tokenizer for the given model.

```java
public Tokenizer(Model model) throws GenAIException
```

---

### encode

Encodes a string into a sequence of token ids.

```java
public Sequences encode(String string) throws GenAIException
```

---

### encodeBatch

Encodes an array of strings into a sequence of token ids for each input.

```java
public Sequences encodeBatch(String[] strings) throws GenAIException
```

---

### decode

Decodes a sequence of token ids into text.

```java
public String decode(int[] sequence) throws GenAIException
```

---

### decodeBatch

Decodes a batch of sequences of token ids into text.

```java
public String[] decodeBatch(Sequences sequences) throws GenAIException
```

---

### createStream

Creates a TokenizerStream object for streaming tokenization.

```java
public TokenizerStream createStream() throws GenAIException
```

---

## TokenizerStream class

### decode

Decodes a single token in the stream and returns the generated string chunk.

```java
public String decode(int token) throws GenAIException
```

---

## GeneratorParams class

### Constructor

Initializes generator parameters for the given model.

```java
public GeneratorParams(Model model) throws GenAIException
```

---

### setSearchOption (double)

Sets a numeric search option.

```java
public void setSearchOption(String optionName, double value) throws GenAIException
```

---

### setSearchOption (boolean)

Sets a boolean search option.

```java
public void setSearchOption(String optionName, boolean value) throws GenAIException
```

---

### setInput (Sequences)

Sets the prompt(s) for model execution using sequences.

```java
public void setInput(Sequences sequences) throws GenAIException
```

---

### setInput (int[])

Sets the prompt(s) token ids for model execution.

```java
public void setInput(int[] tokenIds, int sequenceLength, int batchSize) throws GenAIException
```

---

## Generator class

### Constructor

Constructs a Generator object with the given model and generator parameters.

```java
public Generator(Model model, GeneratorParams generatorParams) throws GenAIException
```

---

### isDone

Checks if the generation process is done.

```java
public boolean isDone()
```

---

### computeLogits

Computes the logits for the next token in the sequence.

```java
public void computeLogits() throws GenAIException
```

---

### generateNextToken

Generates the next token in the sequence.

```java
public void generateNextToken() throws GenAIException
```

---

### getSequence

Retrieves a sequence of token ids for the specified sequence index.

```java
public int[] getSequence(long sequenceIndex) throws GenAIException
```

---

### getLastTokenInSequence

Retrieves the last token in the sequence for the specified sequence index.

```java
public int getLastTokenInSequence(long sequenceIndex) throws GenAIException
```

---

## Sequences class

### numSequences

Gets the number of sequences in the collection.

```java
public long numSequences()
```

---

### getSequence

Gets the sequence at the specified index.

```java
public int[] getSequence(long sequenceIndex)
```

---

## Tensor class

### Constructor

Constructs a Tensor with the given data, shape, and element type.

```java
public Tensor(ByteBuffer data, long[] shape, ElementType elementType) throws GenAIException
```

---

## Result class

### isSuccess

Indicates if the operation was successful.

```java
public boolean isSuccess()
```

---

### getError

Gets the error message from a failed operation.

```java
public String getError()
```

---

## Utils class

### setLogBool

Sets a boolean logging option.

```java
public static void setLogBool(String name, boolean value)
```

---

### setLogString

Sets a string logging option.

```java
public static void setLogString(String name, String value)
```

---

### setCurrentGpuDeviceId

Sets the current GPU device ID.

```java
public static void setCurrentGpuDeviceId(int deviceId)
```

---

### getCurrentGpuDeviceId

Gets the current GPU device ID.

```java
public static int getCurrentGpuDeviceId()
```

---