---
title: C# API
description: C# API reference for ONNX Runtime generate() API
has_children: false
parent: API docs
grand_parent: Generate API (Preview)
nav_order: 2
---

# ONNX Runtime generate() C# API

_Note: this API is in preview and is subject to change._

{: .no_toc }

* TOC placeholder
{:toc}

## Overview

This document describes the C# API for ONNX Runtime GenAI.  
Below are the main classes and methods, with code snippets and descriptions for each.

---

## Model class

### Constructor

Initializes a new model from the given model path.

```csharp
public Model(string modelPath)
```

---

### Generate

Generates output sequences using the provided generator parameters.

```csharp
public Sequences Generate(GeneratorParams generatorParams)
```

---

## Config class

### Constructor

Initializes a new configuration object from a config path.

```csharp
public Config(string configPath)
```

---

### ClearProviders

Clears all providers from the configuration.

```csharp
public void ClearProviders()
```

---

### AppendProvider

Appends a provider to the configuration.

```csharp
public void AppendProvider(string provider)
```

---

### SetProviderOption

Sets a provider option in the configuration.

```csharp
public void SetProviderOption(string provider, string name, string value)
```

---

### Overlay

Overlays a JSON string onto the configuration.

```csharp
public void Overlay(string json)
```

---

## Tokenizer class

### Constructor

Initializes a tokenizer for the given model.

```csharp
public Tokenizer(Model model)
```

---

### Encode

Encodes a string and returns the encoded sequences.

```csharp
public Sequences Encode(string str)
```

---

### EncodeBatch

Encodes a batch of strings and returns the encoded sequences.

```csharp
public Sequences EncodeBatch(string[] strings)
```

---

### Decode

Decodes a sequence of tokens into a string.

```csharp
public string Decode(ReadOnlySpan<int> sequence)
```

---

### DecodeBatch

Decodes a batch of sequences into an array of strings.

```csharp
public string[] DecodeBatch(Sequences sequences)
```

---

### ApplyChatTemplate

Applies a chat template to messages and tools.

```csharp
public string ApplyChatTemplate(string template, string messages, string tools, bool addGenerationPrompt)
```

---

### CreateStream

Creates a tokenizer stream for incremental decoding.

```csharp
public TokenizerStream CreateStream()
```

---

## TokenizerStream class

### Decode

Decodes a single token in the stream and returns the generated string chunk.

```csharp
public string Decode(int token)
```

---

## GeneratorParams class

### Constructor

Initializes generator parameters for the given model.

```csharp
public GeneratorParams(Model model)
```

---

### SetSearchOption (double)

Sets a numeric search option.

```csharp
public void SetSearchOption(string searchOption, double value)
```

---

### SetSearchOption (bool)

Sets a boolean search option.

```csharp
public void SetSearchOption(string searchOption, bool value)
```

---

### TryGraphCaptureWithMaxBatchSize

Attempts to enable graph capture mode with a maximum batch size.

```csharp
public void TryGraphCaptureWithMaxBatchSize(int maxBatchSize)
```

---

### SetInputIDs

Sets the input IDs for the generator parameters.

```csharp
public void SetInputIDs(ReadOnlySpan<int> inputIDs, ulong sequenceLength, ulong batchSize)
```

---

### SetInputSequences

Sets the input sequences for the generator parameters.

```csharp
public void SetInputSequences(Sequences sequences)
```

---

### SetModelInput

Sets an additional model input.

```csharp
public void SetModelInput(string name, Tensor value)
```

---

## Generator class

### Constructor

Initializes a generator from the given model and generator parameters.

```csharp
public Generator(Model model, GeneratorParams generatorParams)
```

---

### IsDone

Checks if generation is complete.

```csharp
public bool IsDone()
```

---

### ComputeLogits

Computes the logits for the current state.

```csharp
public void ComputeLogits()
```

---

### GenerateNextToken

Generates the next token.

```csharp
public void GenerateNextToken()
```

---

### GetSequence

Returns the generated sequence at the given index.

```csharp
public ReadOnlySpan<int> GetSequence(ulong index)
```

---

### SetActiveAdapter

Sets the active adapter on this Generator instance.

```csharp
public void SetActiveAdapter(Adapters adapters, string adapterName)
```

**Parameters**

- `adapters`: the previously created `Adapters` object
- `adapterName`: the name of the adapter to activate

**Return value**

`void`

**Exception**

Throws on error.

---

## Result class

### Error

Gets the error message from a failed operation.

```csharp
public string Error { get; }
```

---

### Success

Indicates if the operation was successful.

```csharp
public bool Success { get; }
```

---

## Sequences class

### NumSequences

Gets the number of sequences.

```csharp
public ulong NumSequences { get; }
```

---

### Indexer

Gets the sequence at the specified index.

```csharp
public ReadOnlySpan<int> this[ulong sequenceIndex]
```

---

## Tensor class

### Constructor

Initializes a tensor from a buffer.

```csharp
public Tensor(Array data, long[] shape, ElementType elementType)
```

---

### Data

Gets the underlying data buffer.

```csharp
public Array Data { get; }
```

---

### Shape

Gets the shape of the tensor.

```csharp
public long[] Shape { get; }
```

---

### ElementType

Gets the element type of the tensor.

```csharp
public ElementType ElementType { get; }
```

---

## Utils class

### SetLogBool

Sets a boolean logging option.

```csharp
public static void SetLogBool(string name, bool value)
```

---

### SetLogString

Sets a string logging option.

```csharp
public static void SetLogString(string name, string value)
```

---

### SetCurrentGpuDeviceId

Sets the current GPU device ID.

```csharp
public static void SetCurrentGpuDeviceId(int deviceId)
```

---

### GetCurrentGpuDeviceId

Gets the current GPU device ID.

```csharp
public static int GetCurrentGpuDeviceId()
```

---