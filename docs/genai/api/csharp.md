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

## Model class

### Constructor

```csharp
public Model(string modelPath)
```

### Generate method

```csharp
public Sequences Generate(GeneratorParams generatorParams)
```

## Tokenizer class

### Constructor

```csharp
public Tokenizer(Model model)
```

### Encode method

```csharp
public Sequences Encode(string str)
```

### Encode batch method

```csharp
public Sequences EncodeBatch(string[] strings)
```

### ApplyChatTemplate method

```csharp
public string ApplyChatTemplate(string template_str, string messages, bool add_generation_prompt)
```

### Decode method

```csharp
public string Decode(ReadOnlySpan<int> sequence)
```

### Decode batch method

```csharp
public string[] DecodeBatch(Sequences sequences)
```

### Create stream method

```csharp
public TokenizerStream CreateStream()
```

## TokenizerStream class

### Decode method

```csharp
public string Decode(int token)
```

## GeneratorParams class

### Constructor

```csharp
public GeneratorParams(Model model)
```

### Set search option (double)

```csharp
public void SetSearchOption(string searchOption, double value)
```

### Set search option (bool) method

```csharp
public void SetSearchOption(string searchOption, bool value)
```

### Try graph capture with max batch size

```csharp
 public void TryGraphCaptureWithMaxBatchSize(int maxBatchSize)
```

### Set input ids method

```csharp
public void SetInputIDs(ReadOnlySpan<int> inputIDs, ulong sequenceLength, ulong batchSize)
```

### Set input sequences method

```csharp
public void SetInputSequences(Sequences sequences)
```

### Set model inputs

```csharp
public void SetModelInput(string name, Tensor value)
```


## Generator class

### Constructor

```csharp
public Generator(Model model, GeneratorParams generatorParams)
```

### Is done method

```csharp
public bool IsDone()
```

### Compute logits

```csharp
public void ComputeLogits()
```

### Generate next token method

```csharp
public void GenerateNextToken()
```

### Get sequence

```csharp
public ReadOnlySpan<int> GetSequence(ulong index)
```

### Set active adapter

Sets the active adapter on this Generator instance.

```csharp
using var model = new Model(modelPath);
using var genParams = new GeneratorParams(model);
using var generator = new Generator(model, genParams);
using var adapters = new Adapters(model);
string adapterName = "..."

generator.SetActiveAdapter(adapters, adapterName);
```

#### Parameters

* `adapters`: the previously created `Adapter` object
* `adapterName`: the name of the adapter to activate

#### Return value

`void`

#### Exception

Throws on error.

## Sequences class

### Num sequences member

```csharp
public ulong NumSequences { get { return _numSequences; } }
```

### [] operator

```csharp
public ReadOnlySpan<int> this[ulong sequenceIndex]
```

## Adapter class

This API is used to load and switch fine-tuned adapters, such as LoRA adapters.

### Constructor

Construct an instance of an Adapter class.

```csharp
using var model = new Model(modelPath);

using var adapters = new Adapters(model);
```

#### Parameters

* `model`: a previously constructed model class

### Load Adapter method

Loads an adapter file from disk.

```csharp
string adapterPath = Path()
string adapterName = ...

adapters.LoadAdapter(adapterPath, adapterName);
```

#### Parameters

* `adapterPath`: the path to the adapter file on disk
* `adapterName`: a string identifier used to refer to the adapter in subsequent methods

#### Return value

`void`

### Unload Adapter method

Unloads an adapter file from memory.

```csharp
adapters.UnLoadAdapter(adapterName);
```

#### Parameters

* `adapterName`: the name of the adapter to unload

#### Return value

`void`

#### Execption

Throws an exception on error.


